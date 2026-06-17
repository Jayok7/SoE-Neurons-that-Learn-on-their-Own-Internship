"""
================================================================================
stage7_experiment_harness.py
--------------------------------------------------------------------------------
The experiment harness that turns the working code into presentable results.

It adds, on top of stage6:
  1. INSTRUMENTED training: logs test acc, task loss, recon loss, and all three
     collapse metrics at EVERY epoch (not just the end), so you get curves.
  2. MULTI-SEED runs with mean +/- standard-error bands on every curve.
  3. Two SWEEPS: div_weight (anti-collapse strength) and local_weight
     (task-vs-reconstruction trade-off), each producing accuracy + effective
     rank vs the swept variable.
  4. Sequential schedule VARIANTS: freeze vs fine-tune the encoder in Phase 2,
     crossed with/without the diversity penalty (the 2x2 discussed).
  5. Two PENALTY FORMULATIONS: decoder-weight decorrelation (original) and
     activation decorrelation (Barlow-Twins style), selectable by name.
  6. RESULTS SAVED to disk (JSON) so you can re-plot without re-training, and
     PLOTS written to PNG.

Run modes (set MODE at the bottom):
  "demo"        - tiny, fast, just proves the harness runs
  "curves"      - per-epoch curves, interleaved vs sequential vs fixed
  "div_sweep"   - accuracy + eff-rank vs diversity weight
  "local_sweep" - accuracy + eff-rank vs local (reconstruction) weight
  "seq_variants"- the freeze/finetune x penalty 2x2

Author scaffolding: Jamal Idrissou
================================================================================
"""

import json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

OUT = "experiments"
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==============================================================================
# Model (unchanged neuron-AE layer + classifier)
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
# SPIKING neuron-AE layer (directive compliance). Same INTERFACE as
# NeuronAELayer so the harness, penalties and diagnostics work unchanged:
#   - exposes .encoder (the Linear), .dec_w, .dec_b   (used by diagnostics/penalties)
#   - forward(x) -> per-neuron scalar code (here the firing RATE)
#   - local_loss(x) -> per-neuron reconstruction MSE from that rate
# The only conceptual change vs the static layer: each neuron's scalar "code" is
# its mean firing rate over T leaky-integrate-and-fire timesteps, and the spike
# is made differentiable with a surrogate gradient.
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


# map a string to the hidden-layer class so experiments can switch architecture
# ------------------------------------------------------------------------------
# The paper's OTHER two architectures, so all three of Bull's variants are
# selectable. All share the same interface (.encoder, .dec_w, .dec_b, forward,
# local_loss) so the harness/diagnostics/penalties work on them unchanged.
#
#   NN  : plain MLP hidden layer. NO autoencoder at all -> local_loss == 0, so
#         with the alternating trainer it only ever runs task cycles. This is
#         the honest "traditional model" baseline (distinct from a NAN with
#         local_weight=0, which still carries unused decoder parameters).
#   ANN : ONE shared decoder layer (H -> input) hanging off the whole hidden
#         layer — the standard "one decoder layer" autoencoder in the paper.
#         Contrast with NAN, where each neuron has its OWN 1->input decoder.
# ------------------------------------------------------------------------------
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


# selectable architectures. "static"/"spiking" are NANs (per-neuron decoders);
# "ann" is the shared-decoder autoencoder; "nn" is the plain baseline.
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


# ==============================================================================
# Diversity penalties — TWO formulations, selectable by name
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
    """Barlow-Twins style: decorrelate the ACTIVATIONS over a batch.
    Acts on the representation (what neurons actually output). Needs the input x."""
    activations = layer(x)                                    # (batch, neurons)
    activations = ((activations - activations.mean(0, keepdim=True))
                   / (activations.std(0, keepdim=True) + 1e-8))
    correlation = (activations.t() @ activations) / activations.size(0)
    num_neurons = correlation.size(0)
    off_diagonal = correlation[~torch.eye(num_neurons, dtype=torch.bool,
                                          device=x.device)]
    return (off_diagonal ** 2).mean()                         # push off-diag to 0


PENALTIES = {"decoder": penalty_decoder, "activation": penalty_activation}


# ==============================================================================
# Diagnostics (same three metrics as stage6)
# ==============================================================================
@torch.no_grad()
def metrics(model, loader, probe_batch):
    """Compute test accuracy plus three 'neuron collapse' diagnostics.

    probe_batch: a fixed batch of inputs used to measure how diverse the neurons
    are. The three collapse metrics answer the supervisor's question 'do all the
    neurons end up learning the same thing?':
        decoder_similarity : how alike the neurons' decoder weights are (NAN only)
        activation_corr    : how correlated the neurons' outputs are
        effective_rank     : how many neurons do genuinely independent work
    """
    model.eval()

    # ---- test accuracy ----
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    accuracy = 100.0 * correct / total

    layer = model.hidden

    # ---- decoder similarity (only meaningful when each neuron has its own
    #      decoder row, i.e. the NAN layers). For NN the decoder is unused, for
    #      ANN it is shared and indexed by input dimension, so we report -1
    #      ('not applicable') in those cases. ----
    num_neurons = layer.encoder.out_features
    decoder_is_per_neuron = (layer.decoder_weight.shape[0] == num_neurons
                             and layer.decoder_weight.abs().sum() > 0)
    if decoder_is_per_neuron:
        unit_vectors = F.normalize(layer.decoder_weight, dim=1)
        similarity = unit_vectors @ unit_vectors.t()
        off_diagonal = similarity[~torch.eye(num_neurons, dtype=torch.bool,
                                             device=probe_batch.device)]
        decoder_similarity = off_diagonal.abs().mean().item()
    else:
        decoder_similarity = -1.0     # not applicable for NN / ANN

    # ---- activation correlation: centre and standardise each neuron's outputs
    #      over the batch, then average the off-diagonal of the correlation
    #      matrix (1 = neurons are clones, 0 = uncorrelated). ----
    activations = layer(probe_batch)
    activations = activations - activations.mean(0, keepdim=True)
    standardised = activations / (activations.std(0, keepdim=True) + 1e-8)
    correlation = (standardised.t() @ standardised) / standardised.size(0)
    n = correlation.size(0)
    activation_corr = correlation[~torch.eye(n, dtype=torch.bool,
                                             device=probe_batch.device)].abs().mean().item()

    # ---- effective rank: how many neurons are effectively independent. We take
    #      the singular values of the activation matrix, turn them into a
    #      probability distribution, and compute exp(entropy). Ranges from 1
    #      (total collapse onto one direction) up to num_neurons (full diversity). ----
    singular_values = torch.linalg.svdvals(activations)
    spectrum = singular_values / singular_values.sum()
    entropy = -(spectrum * (spectrum + 1e-12).log()).sum()
    effective_rank = torch.exp(entropy).item()

    return {"acc": accuracy, "dec_sim": decoder_similarity,
            "act_corr": activation_corr, "eff_rank": effective_rank}


# ==============================================================================
# Instrumented training — logs metrics every epoch
# ==============================================================================
def run_interleaved(model, train_loader, test_loader, probe_batch, epochs,
                    local_weight, div_weight=0.0, penalty="decoder"):
    """Paper-style schedule: each batch is randomly EITHER a task-learning cycle
    OR an autoencoding cycle. Records the metrics once per epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    penalty_fn = PENALTIES[penalty]
    history = []
    for _epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            run_task_cycle = local_weight == 0 or torch.rand(1).item() < 0.5
            if run_task_cycle:
                optimizer.zero_grad()
                F.cross_entropy(model(x), y).backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                _, recon_loss = model(x, want_local=True)
                loss = local_weight * recon_loss
                if div_weight > 0:
                    loss = loss + div_weight * penalty_fn(model.hidden, x)
                # A plain NN has no autoencoder: its reconstruction loss is a
                # constant zero with no gradient path. There is nothing to learn
                # on an autoencoding cycle, so fall back to a task cycle rather
                # than calling backward() on a non-differentiable tensor.
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                else:
                    F.cross_entropy(model(x), y).backward()
                    optimizer.step()
        history.append(metrics(model, test_loader, probe_batch))
    return history


def run_sequential(model, train_loader, test_loader, probe_batch,
                   ae_epochs, task_epochs,
                   div_weight=0.0, penalty="decoder", finetune=False):
    """
    Sequential schedule (the variant the supervisor suggested):
      Phase 1 trains ONLY the autoencoders (optionally with a diversity penalty).
      Phase 2 trains the task. If finetune=True the encoder stays trainable at a
        low learning rate so the task signal can repair leftover collapse; if
        finetune=False the hidden layer is frozen, the strict 'is the AE-shaped
        representation good for the task?' test.
    Metrics are logged every epoch across BOTH phases, so a curve shows the
    Phase-1 -> Phase-2 transition.
    """
    penalty_fn = PENALTIES[penalty]
    history = []

    # ---- Phase 1: train the autoencoders (skipped for a plain NN, which has
    #      no autoencoder to train) ----
    first_batch = next(iter(train_loader))[0].to(DEVICE)
    has_autoencoder = model.hidden.local_loss(first_batch).requires_grad
    ae_optimizer = torch.optim.Adam(model.hidden.parameters(), lr=1e-3)
    for _epoch in range(ae_epochs):
        model.train()
        if has_autoencoder:
            for x, _y in train_loader:
                x = x.to(DEVICE)
                ae_optimizer.zero_grad()
                loss = model.hidden.local_loss(x)
                if div_weight > 0:
                    loss = loss + div_weight * penalty_fn(model.hidden, x)
                loss.backward()
                ae_optimizer.step()
        history.append({**metrics(model, test_loader, probe_batch), "phase": 1})

    # ---- Phase 2: train the task ----
    if finetune:
        # encoder keeps learning, but slowly, while the head learns at full speed
        task_optimizer = torch.optim.Adam([
            {"params": model.head.parameters(), "lr": 1e-3},
            {"params": model.hidden.parameters(), "lr": 1e-4},
        ])
    else:
        for param in model.hidden.parameters():
            param.requires_grad_(False)             # freeze the AE representation
        task_optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)

    for _epoch in range(task_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            task_optimizer.zero_grad()
            F.cross_entropy(model(x), y).backward()
            task_optimizer.step()
        history.append({**metrics(model, test_loader, probe_batch), "phase": 2})
    return history


# ==============================================================================
# Multi-seed wrapper: returns per-epoch mean and standard error for each metric
# ==============================================================================
def multiseed(make_and_run, seeds, in_dim, hidden, num_classes):
    """Run an experiment once per seed and aggregate the per-epoch metrics into
    a mean and standard-error curve.

    make_and_run(seed) -> a list (one entry per epoch) of metric dicts.
    Returns {metric_name: {'mean': [...per epoch...], 'sem': [...]}}.
    """
    runs_per_seed = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        runs_per_seed.append(make_and_run(seed))

    metric_names = [name for name in runs_per_seed[0][0].keys() if name != "phase"]
    aggregated = {}
    for name in metric_names:
        # shape (num_seeds, num_epochs): this metric's value at each epoch per seed
        values = np.array([[epoch[name] for epoch in run] for run in runs_per_seed])
        aggregated[name] = {
            "mean": values.mean(0).tolist(),
            "sem": (values.std(0) / np.sqrt(len(seeds))).tolist(),
        }
    # carry the phase markers (for the sequential schedule) straight through
    aggregated["phase"] = [epoch.get("phase", 0) for epoch in runs_per_seed[0]]
    return aggregated


# ==============================================================================
# Data
# ==============================================================================
def get_loaders(resolution=14, num_classes=10, batch_size=128, root="./data"):
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
        tr = prep(datasets.MNIST(root, train=True, download=True, transform=tf))
        te = prep(datasets.MNIST(root, train=False, download=True, transform=tf))
    except Exception as e:
        print(f"[get_loaders] real MNIST unavailable ({type(e).__name__}); synthetic stand-in.")
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
# Plotting (saves PNGs; no display needed)
# ==============================================================================
def plot_curves(curve_dict, metric, ylabel, fname, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.5))
    for label, agg in curve_dict.items():
        m = np.array(agg[metric]["mean"]); s = np.array(agg[metric]["sem"])
        ep = np.arange(1, len(m) + 1)
        plt.plot(ep, m, label=label, linewidth=2)
        plt.fill_between(ep, m - s, m + s, alpha=0.2)
    plt.xlabel("epoch"); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    path = os.path.join(OUT, fname); plt.savefig(path, dpi=130); plt.close()
    print(f"  saved {path}")


def plot_sweep(xs, ys_dict, xlabel, fname, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    am, asem = ys_dict["acc_mean"], ys_dict["acc_sem"]
    rm, rsem = ys_dict["rank_mean"], ys_dict["rank_sem"]
    ax1.errorbar(xs, am, yerr=asem, marker="o", color="tab:blue", label="test acc")
    ax2.errorbar(xs, rm, yerr=rsem, marker="s", color="tab:red", label="eff. rank")
    ax1.set_xlabel(xlabel); ax1.set_ylabel("test accuracy (%)", color="tab:blue")
    ax2.set_ylabel("effective rank", color="tab:red")
    ax1.set_title(title); ax1.grid(alpha=0.3); fig.tight_layout()
    path = os.path.join(OUT, fname); plt.savefig(path, dpi=130); plt.close()
    print(f"  saved {path}")


def save_json(obj, fname):
    with open(os.path.join(OUT, fname), "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  saved {os.path.join(OUT, fname)}")


# ==============================================================================
# EXPERIMENTS
# ==============================================================================
def exp_curves(seeds, epochs, res=14, nc=10, hidden=128, arch="static"):
    """Per-epoch curves: interleaved, sequential(frozen), sequential(fixed+pen).
    arch: "static" or "spiking" — selects the hidden-layer neuron model."""
    tr, te, in_dim = get_loaders(resolution=res, num_classes=nc)
    xb = next(iter(te))[0].to(DEVICE)
    print(f"exp_curves [{arch}]: interleaved ...")
    inter = multiseed(lambda s: run_interleaved(
        NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE), tr, te, xb, epochs, 1.0),
        seeds, in_dim, hidden, nc)
    print(f"exp_curves [{arch}]: sequential (frozen, no penalty) ...")
    seq = multiseed(lambda s: run_sequential(
        NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE), tr, te, xb,
        epochs // 2, epochs - epochs // 2, div_weight=0.0, finetune=False),
        seeds, in_dim, hidden, nc)
    print(f"exp_curves [{arch}]: sequential (frozen + diversity penalty) ...")
    seqp = multiseed(lambda s: run_sequential(
        NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE), tr, te, xb,
        epochs // 2, epochs - epochs // 2, div_weight=5.0, finetune=False),
        seeds, in_dim, hidden, nc)
    curves = {"interleaved": inter, "sequential": seq, "sequential+penalty": seqp}
    save_json(curves, f"curves_{arch}.json")
    plot_curves(curves, "acc", "test accuracy (%)", f"curve_acc_{arch}.png",
                f"Test accuracy over training ({arch})")
    plot_curves(curves, "eff_rank", "effective rank", f"curve_rank_{arch}.png",
                f"Neuron diversity (effective rank) over training ({arch})")
    return curves


def exp_sweep(kind, values, seeds, epochs, res=14, nc=10, hidden=128, arch="static"):
    """Sweep div_weight (kind='div') or local_weight (kind='local').
    Records FINAL accuracy and eff-rank (mean+sem over seeds) per value.
    arch: "static" or "spiking"."""
    tr, te, in_dim = get_loaders(resolution=res, num_classes=nc)
    xb = next(iter(te))[0].to(DEVICE)
    acc_m, acc_s, rank_m, rank_s = [], [], [], []
    for v in values:
        print(f"exp_sweep[{kind}/{arch}]: value={v} ...")
        def run(s):
            m = NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE)
            if kind == "div":   # interleaved, vary diversity penalty
                return run_interleaved(m, tr, te, xb, epochs, 1.0, div_weight=v)
            else:               # vary local (reconstruction) weight, no penalty
                return run_interleaved(m, tr, te, xb, epochs, v, div_weight=0.0)
        agg = multiseed(run, seeds, in_dim, hidden, nc)
        acc_m.append(agg["acc"]["mean"][-1]);  acc_s.append(agg["acc"]["sem"][-1])
        rank_m.append(agg["eff_rank"]["mean"][-1]); rank_s.append(agg["eff_rank"]["sem"][-1])
    res_obj = {"values": values, "acc_mean": acc_m, "acc_sem": acc_s,
               "rank_mean": rank_m, "rank_sem": rank_s}
    save_json(res_obj, f"sweep_{kind}_{arch}.json")
    xlabel = "diversity weight" if kind == "div" else "local (reconstruction) weight"
    plot_sweep(values, res_obj, xlabel, f"sweep_{kind}_{arch}.png",
               f"Final accuracy & neuron diversity vs {xlabel} ({arch})")
    return res_obj


def exp_seq_variants(seeds, epochs, res=14, nc=10, hidden=128, arch="static"):
    """The 2x2: {freeze, finetune} x {no penalty, penalty}. Final metrics.
    arch: "static" or "spiking"."""
    tr, te, in_dim = get_loaders(resolution=res, num_classes=nc)
    xb = next(iter(te))[0].to(DEVICE)
    grid = {}
    for finetune in (False, True):
        for dw in (0.0, 5.0):
            name = f"{'finetune' if finetune else 'freeze'}+{'pen' if dw>0 else 'nopen'}"
            print(f"exp_seq_variants [{arch}]: {name} ...")
            agg = multiseed(lambda s: run_sequential(
                NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE), tr, te, xb,
                epochs // 2, epochs - epochs // 2, div_weight=dw, finetune=finetune),
                seeds, in_dim, hidden, nc)
            grid[name] = {"acc": agg["acc"]["mean"][-1],
                          "acc_sem": agg["acc"]["sem"][-1],
                          "eff_rank": agg["eff_rank"]["mean"][-1]}
    save_json(grid, f"seq_variants_{arch}.json")
    print(f"\n  2x2 sequential variants ({arch}, final):")
    print(f"  {'variant':20} {'acc':>8} {'+-sem':>7} {'eff_rank':>9}")
    for k, v in grid.items():
        print(f"  {k:20} {v['acc']:>8.2f} {v['acc_sem']:>7.2f} {v['eff_rank']:>9.1f}")
    return grid


# ==============================================================================
if __name__ == "__main__":
    MODE = "demo"   # "demo"|"curves"|"div_sweep"|"local_sweep"|"seq_variants"|"compare_arch"|"compare_all4"
    ARCH = "static" # "static"(NAN) | "spiking"(spiking NAN) | "ann"(shared decoder) | "nn"(plain baseline)
    t0 = time.time()
    print(f"device={DEVICE} mode={MODE} arch={ARCH}")

    if MODE == "demo":
        # tiny everything, run BOTH NAN archs so directive compliance is visible
        exp_curves(seeds=[0, 1], epochs=2, res=7, nc=10, hidden=64, arch="static")
        exp_curves(seeds=[0, 1], epochs=2, res=7, nc=10, hidden=64, arch="spiking")

    elif MODE == "curves":
        exp_curves(seeds=range(5), epochs=10, arch=ARCH)

    elif MODE == "div_sweep":
        exp_sweep("div", [0, 0.1, 0.5, 1, 2, 5, 10, 20], seeds=range(5), epochs=6, arch=ARCH)

    elif MODE == "local_sweep":
        exp_sweep("local", [0, 0.1, 0.25, 0.5, 1, 2, 4], seeds=range(5), epochs=6, arch=ARCH)

    elif MODE == "seq_variants":
        exp_seq_variants(seeds=range(5), epochs=10, arch=ARCH)

    elif MODE == "compare_arch":
        # same experiment for both NAN neuron models: static vs spiking
        for a in ("static", "spiking"):
            exp_seq_variants(seeds=range(5), epochs=10, arch=a)
            exp_sweep("div", [0, 0.5, 1, 2, 5, 10], seeds=range(5), epochs=6, arch=a)

    elif MODE == "compare_all4":
        # the paper's full architecture comparison: NN vs ANN vs NAN(static) vs
        # NAN(spiking), interleaved training, final accuracy + diversity per arch.
        tr, te, in_dim = get_loaders(resolution=14, num_classes=10)
        xb = next(iter(te))[0].to(DEVICE)
        print(f"\n  {'arch':10} {'acc':>8} {'+-sem':>7} {'eff_rank':>9} {'dec_sim':>8}")
        summary = {}
        for a in ("nn", "ann", "static", "spiking"):
            # nn uses local_weight irrelevant (its local_loss is 0); others use 1.0
            agg = multiseed(lambda s: run_interleaved(
                NAN_Classifier(in_dim, 128, 10, arch=a).to(DEVICE),
                tr, te, xb, epochs=10, local_weight=1.0), range(5), in_dim, 128, 10)
            summary[a] = {"acc": agg["acc"]["mean"][-1], "acc_sem": agg["acc"]["sem"][-1],
                          "eff_rank": agg["eff_rank"]["mean"][-1],
                          "dec_sim": agg["dec_sim"]["mean"][-1]}
            v = summary[a]
            ds = "n/a" if v["dec_sim"] < 0 else f"{v['dec_sim']:.3f}"
            print(f"  {a:10} {v['acc']:>8.2f} {v['acc_sem']:>7.2f} {v['eff_rank']:>9.1f} {ds:>8}")
        save_json(summary, "compare_all4.json")

    print(f"\ndone in {time.time()-t0:.1f}s. Outputs in ./{OUT}/")
