"""
nan/models.py — combined models, configuration, data loading, and penalties.

Includes:
    - Global configuration (device, output directories)
    - Fashion-MNIST data loaders with resolution/class-count hooks and synthetic fallbacks
    - Diversity regularizers/penalties to discourage neuron collapse
    - Autoencoder layer architectures (NN, ANN, static NAN, spiking NAN) and classifiers
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ==============================================================================
# 1. Global Configuration
# ==============================================================================
OUT = os.environ.get("NAN_OUT_DIR", "experiments")
DATA_ROOT = os.environ.get("NAN_DATA_DIR", "./data")
os.makedirs(OUT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fashion-MNIST class labels (index -> human-readable name). Same 10-class,
# 28x28 grayscale layout as MNIST, so every downstream shape is identical; only
# the semantics of each label change (garments instead of digits).
FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

def outdir(subdir=None):
    """Return the output directory, creating a subdirectory under OUT if given."""
    d = os.path.join(OUT, subdir) if subdir else OUT
    os.makedirs(d, exist_ok=True)
    return d

# ==============================================================================
# 2. Data Loader
# ==============================================================================
def get_loaders(resolution=14, num_classes=10, batch_size=128, root=None):
    """Fashion-MNIST loaders with the resolution / class-count knobs.

    resolution is the "N-axis"; num_classes subsets the garment categories
    (see FASHION_CLASSES for the label -> name mapping). Falls back to a
    synthetic stand-in if Fashion-MNIST can't be downloaded (offline sandbox),
    so the code still runs — but real runs use true Fashion-MNIST.

    Fashion-MNIST is a drop-in replacement for MNIST: same 60k/10k split, same
    28x28 single-channel images, same 10 classes. That means the encoder input
    dimension (resolution*resolution), the classifier head width (num_classes),
    and every reshape in the visualisers stay exactly as they were for MNIST —
    only the underlying pixels and the meaning of each class change.
    """
    root = root if root is not None else DATA_ROOT
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.Resize((resolution, resolution)),
                             transforms.ToTensor()])
    def prep(ds):
        X, y = [], []
        for img, lab in ds:
            if lab < num_classes:
                X.append(img.view(-1)); y.append(lab)
        return TensorDataset(torch.stack(X), torch.tensor(y))
    try:
        tr = prep(datasets.FashionMNIST(root, train=True, download=True, transform=tf))
        te = prep(datasets.FashionMNIST(root, train=False, download=True, transform=tf))
    except Exception as e:
        print(f"[get_loaders] real Fashion-MNIST unavailable ({type(e).__name__}); synthetic stand-in.")
        d = resolution * resolution
        def synth(n):
            y = torch.randint(0, num_classes, (n,))
            X = (torch.rand(n, d) * 0.1 + (y.float().unsqueeze(1) / num_classes)).clamp(0, 1)
            return TensorDataset(X, y)
        tr, te = synth(2000), synth(1000)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(te, batch_size=batch_size, shuffle=False),
            resolution * resolution)

# ==============================================================================
# 3. Diversity Penalties
# ==============================================================================
def penalty_decoder(layer, x=None):
    """Push the per-neuron DECODER weight vectors apart (decorrelate them).
    Acts on parameters: discourages neurons from reconstructing the same thing."""
    unit_vectors = F.normalize(layer.decoder_weight, dim=1)   # one unit vec/neuron
    similarity = unit_vectors @ unit_vectors.t()              # (neurons, neurons)
    num_neurons = similarity.size(0)
    off_diagonal = similarity[~torch.eye(num_neurons, dtype=torch.bool,
                                         device=layer.decoder_weight.device)]
    return (off_diagonal ** 2).mean()                         # 0 when orthogonal


def penalty_activation(layer, x):
    """
    based on Barlow-Twins redundancy reduction: decorrelate the ACTIVATIONS over a 
    batch. Acts on the representation (what neurons actually output). Needs 
    the input x.
    """
    activations = layer(x)                                    # (batch, neurons)
    activations = ((activations - activations.mean(0, keepdim=True))
                   / (activations.std(0, keepdim=True) + 1e-8))
    correlation = (activations.t() @ activations) / activations.size(0)
    num_neurons = correlation.size(0)
    off_diagonal = correlation[~torch.eye(num_neurons, dtype=torch.bool,
                                          device=x.device)]
    return (off_diagonal ** 2).mean()                         # push off-diag to 0


# ------------------------------------------------------------------------------
# Group-aware decoder penalties
#
# penalty_decoder above averages squared cosine over EVERY pair of neurons. In a
# grouped layer that silently aggregates two different things: pairs inside a
# group, which share a reconstruction target and are supposed to divide it, and
# pairs in different groups, which reconstruct independently and have no reason
# to agree or disagree. The single number cannot tell you which of the two the
# penalty is acting on, and pushing them apart is not obviously the same
# instruction in both cases.
#
# These split it. `decoder_within` penalises only same-group pairs; the
# instruction is "the members of a group should cover different parts of their
# shared target". `decoder_between` penalises only cross-group pairs; the
# instruction is "different groups should end up encoding different things".
# ------------------------------------------------------------------------------
def _group_masks(layer):
    """(same, diff) boolean (H, H) masks, diagonal excluded, or None.

    Two neurons count as sharing a group if their responsibility rows overlap
    anywhere, which is what makes this work for ring routing as well as hard
    partitions: with overlapping groups a pair can be adjacent (sharing one
    decoder) without being identical.
    """
    if not hasattr(layer, "responsibilities"):
        return None
    with torch.no_grad():
        R = layer.responsibilities()                     # (H, G)
        share = (R > 0).float() @ (R > 0).float().t()    # (H, H), >0 if shared
    n = share.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=share.device)
    same = (share > 0) & ~eye
    diff = (share == 0) & ~eye
    return same, diff


def _decoder_cos2(layer):
    u = F.normalize(layer.decoder_weight, dim=1)
    return (u @ u.t()) ** 2


def penalty_decoder_within(layer, x=None):
    """Squared cosine averaged over same-group decoder pairs only."""
    masks = _group_masks(layer)
    c = _decoder_cos2(layer)
    if masks is None:                                    # NAN / ANN: no groups
        return penalty_decoder(layer, x)
    same, _ = masks
    return c[same].mean() if same.any() else c.sum() * 0.0


def penalty_decoder_between(layer, x=None):
    """Squared cosine averaged over cross-group decoder pairs only."""
    masks = _group_masks(layer)
    c = _decoder_cos2(layer)
    if masks is None:
        return penalty_decoder(layer, x)
    _, diff = masks
    return c[diff].mean() if diff.any() else c.sum() * 0.0


@torch.no_grad()
def decoder_overlap_split(layer):
    """Diagnostic: the two components the plain `decoder` penalty aggregates.

    Recorded on every run, whatever penalty is in use, so the question "is the
    single penalty acting on within-group or between-group structure?" can be
    answered from the existing sweeps rather than needing new ones.
    """
    if not hasattr(layer, "decoder_weight"):
        return None
    c = _decoder_cos2(layer)
    n = c.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=c.device)
    out = {"decoder_cos2_all": float(c[~eye].mean())}
    masks = _group_masks(layer)
    if masks is None:
        out.update(decoder_cos2_within=None, decoder_cos2_between=None,
                   frac_pairs_within=None)
        return out
    same, diff = masks
    out.update(
        decoder_cos2_within=float(c[same].mean()) if same.any() else None,
        decoder_cos2_between=float(c[diff].mean()) if diff.any() else None,
        frac_pairs_within=float(same.float().sum() / (~eye).float().sum()))
    return out


PENALTIES = {"decoder": penalty_decoder,
             "decoder_within": penalty_decoder_within,
             "decoder_between": penalty_decoder_between,
             "activation": penalty_activation}

# ==============================================================================
# 4. Neural Network & Autoencoder Layers
# ==============================================================================
class NeuronAELayer(nn.Module):
    """NAN hidden layer: every neuron is its own autoencoder.

    Each neuron encodes the input vector down to a single scalar (its activation)
    and decodes that scalar back to a full reconstruction of the input. The
    forward pass returns the activations (passed to the next layer); local_loss
    returns how badly each neuron reconstructs its input, averaged.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.encoder = nn.Linear(in_features, out_features)
        # One decoder PER neuron: row j maps neuron j's scalar -> full input.
        # Shapes are (num_neurons, in_features).
        self.decoder_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.decoder_bias = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.decoder_weight, a=5 ** 0.5)

    def forward(self, x):
        # the scalar "code" each neuron emits; this is the normal layer output
        return torch.sigmoid(self.encoder(x))

    def local_loss(self, x):
        code = torch.sigmoid(self.encoder(x))          # (batch, num_neurons)
        # reconstruct the input from EACH neuron's scalar, independently:
        #   reconstruction[b, j, :] = code[b,j] * decoder_weight[j] + decoder_bias[j]
        reconstruction = (code.unsqueeze(-1) * self.decoder_weight.unsqueeze(0)
                          + self.decoder_bias.unsqueeze(0))   # (batch, neurons, in)
        target = x.unsqueeze(1).expand_as(reconstruction)     # broadcast input
        return F.mse_loss(reconstruction, target)


# ------------------------------------------------------------------------------
# SPIKING neuron-AE layer. Same INTERFACE as NeuronAELayer:
#   - exposes .encoder (the Linear), .decoder_weight, .decoder_bias
#   - forward(x) -> per-neuron scalar code (here the firing RATE)
#   - local_loss(x) -> per-neuron reconstruction MSE from that rate
# ------------------------------------------------------------------------------
class _SurrogateSpike(torch.autograd.Function):
    """A spike (hard threshold) with a fake-but-usable gradient.

    A real spike is a step function, whose derivative is zero everywhere and so
    blocks gradient descent. The standard fix: emit the true 0/1 spike on the
    forward pass, but on the backward pass pretend the step was a smooth curve
    so a gradient still flows. This is the 'surrogate gradient' trick.
    """
    @staticmethod
    def forward(ctx, membrane_minus_threshold):
        ctx.save_for_backward(membrane_minus_threshold)
        return (membrane_minus_threshold >= 0).float()      # real spike: 1 or 0

    @staticmethod
    def backward(ctx, grad_output):
        (shifted_membrane,) = ctx.saved_tensors
        # derivative of a steep smooth sigmoid, peaked at the threshold
        surrogate_grad = 1.0 / (10.0 * torch.abs(shifted_membrane) + 1.0) ** 2
        return grad_output * surrogate_grad


spike = _SurrogateSpike.apply


class SpikingNeuronAELayer(nn.Module):
    """Spiking version of NeuronAELayer. Same interface, but each neuron's
    scalar 'code' is its FIRING RATE over `n_timesteps` leaky-integrate-and-fire
    steps, rather than a single static activation."""
    def __init__(self, in_features, out_features,
                 n_timesteps=20, leak=0.9, threshold=1.0):
        super().__init__()
        self.num_neurons = out_features
        self.n_timesteps = n_timesteps
        self.leak = leak                                # membrane decay per step
        self.threshold = threshold                      # voltage needed to fire
        self.encoder = nn.Linear(in_features, out_features)
        self.decoder_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.decoder_bias = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.decoder_weight, a=5 ** 0.5)

    def firing_rate(self, x):
        """Simulate the neurons for n_timesteps; return mean spikes per neuron."""
        batch_size = x.size(0)
        input_current = self.encoder(x)                 # constant drive each step
        membrane = torch.zeros(batch_size, self.num_neurons, device=x.device)
        spike_count = torch.zeros(batch_size, self.num_neurons, device=x.device)
        for _ in range(self.n_timesteps):
            membrane = self.leak * membrane + input_current        # integrate
            fired = spike(membrane - self.threshold)               # 1 where fired
            membrane = membrane * (1.0 - fired)                    # reset on fire
            spike_count = spike_count + fired
        return spike_count / self.n_timesteps           # rate in [0, 1]

    def forward(self, x):
        return self.firing_rate(x)                      # the neuron's scalar code

    def local_loss(self, x):
        rate = self.firing_rate(x)                      # (batch, num_neurons)
        reconstruction = (rate.unsqueeze(-1) * self.decoder_weight.unsqueeze(0)
                          + self.decoder_bias.unsqueeze(0))
        target = x.unsqueeze(1).expand_as(reconstruction)
        return F.mse_loss(reconstruction, target)


class PlainNNLayer(nn.Module):
    """Plain hidden layer, no autoencoding (Bull's 'NN'): the traditional baseline."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.encoder = nn.Linear(in_features, out_features)
        # Unused placeholder decoder params, frozen, so the diagnostics that look
        # for .decoder_weight still find an attribute. Never trained, because this
        # layer's reconstruction loss is identically zero.
        self.decoder_weight = nn.Parameter(torch.zeros(out_features, in_features),
                                           requires_grad=False)
        self.decoder_bias = nn.Parameter(torch.zeros(out_features, in_features),
                                         requires_grad=False)

    def forward(self, x):
        return torch.sigmoid(self.encoder(x))

    def local_loss(self, x):
        # no autoencoder -> a true zero with no gradient path; the trainer detects
        # this and simply skips the autoencoding cycle for a plain NN.
        return torch.zeros((), device=x.device)


class LayerAELayer(nn.Module):
    """Single SHARED decoder over the whole hidden layer (Bull's 'ANN')."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.encoder = nn.Linear(in_features, out_features)
        self.decoder = nn.Linear(out_features, in_features)   # hidden -> input
        # Expose the decoder weights under the same name the NAN layers use, so
        # the diagnostics read something. Note: here a row corresponds to an
        # INPUT dimension, not a neuron, so the decoder-similarity metric isn't
        # meaningful for ANN — the activation metrics are the ones to trust.
        self.decoder_weight = self.decoder.weight
        self.decoder_bias = self.decoder.bias

    def forward(self, x):
        return torch.sigmoid(self.encoder(x))

    def local_loss(self, x):
        hidden_code = torch.sigmoid(self.encoder(x))   # (batch, num_neurons)
        reconstruction = self.decoder(hidden_code)     # (batch, in_features)
        return F.mse_loss(reconstruction, x)


# Selectable architectures
ARCHS = {
    "static":  NeuronAELayer,          # NAN, static neurons (paper's proposal)
    "spiking": SpikingNeuronAELayer,   # NAN, spiking neurons (your extension)
    "ann":     LayerAELayer,           # ANN, one shared decoder layer
    "nn":      PlainNNLayer,           # NN, no autoencoder (traditional baseline)
}


class HiddenLayerClassifier(nn.Module):
    """A one-hidden-layer classifier where the hidden layer can be any of the
    four architectures (NN / ANN / NAN-static / NAN-spiking), chosen by `arch`."""
    def __init__(self, in_dim, hidden=128, num_classes=10, arch="static"):
        super().__init__()
        self.hidden = ARCHS[arch](in_dim, hidden)   # the chosen hidden layer type
        self.head = nn.Linear(hidden, num_classes)  # linear readout to class logits

    def forward(self, x, want_local=False):
        recon_loss = self.hidden.local_loss(x) if want_local else None
        logits = self.head(self.hidden(x))
        return (logits, recon_loss) if want_local else logits


# Backwards-compatible alias (older call sites used this name).
NAN_Classifier = HiddenLayerClassifier
