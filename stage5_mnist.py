"""
================================================================================
stage5_mnist.py
--------------------------------------------------------------------------------
Porting Bull's neuron-autoencoder from the NK regression task to MNIST
classification — BOTH a gradient-only version and a gradient+spiking version.

Read stage1_2_bull_faithful.py and stage3_4_gradient_and_spiking.py first.
This file deliberately REUSES those building blocks unchanged wherever possible;
the comments call out exactly what changed from NK and what did not.

WHAT CHANGES GOING FROM NK -> MNIST (and what doesn't)
------------------------------------------------------
  (1) Task: regression (1 sigmoid node, MSE) -> classification (10 logits,
      cross-entropy). This touches ONLY the head and the task-loss line.
  (2) Reconstruction target: a 20-dim {-1,+1} vector -> a (resolution^2)-dim
      pixel vector in [0,1]. The decoder is mechanically identical (scalar ->
      input_dim); only the dimension and the value range change.
  (3) Difficulty knob: NK gave us N (size) and K (ruggedness) for free. MNIST is
      ONE fixed problem, so to reproduce Bull's *conditional* finding we
      MANUFACTURE the sweep axes:
          NK's N (problem size)      -> input RESOLUTION (7x7, 14x14, 28x28)
          NK's K (interdependence)   -> task HARDNESS proxy (#classes / noise)
      That parallel is the whole point: the MNIST study has the same 2D shape as
      the NK Figure-6 study, so the two sit side by side in your write-up.

The neuron-AE MECHANISM itself (each neuron reconstructs its own input from its
single scalar code; training alternates task vs autoencoding cycles) is UNCHANGED
from the NK files. That invariance is the thing to understand.

Author scaffolding: Jamal Idrissou
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ==============================================================================
# REUSED VERBATIM FROM stage3_4 — the neuron-AE layer does NOT change for MNIST.
# (Copied here so this file runs standalone; in your repo, import it instead:
#     from stage3_4_gradient_and_spiking import NeuronAELayer, SpikingNeuronAELayer)
# ==============================================================================
class NeuronAELayer(nn.Module):
    """Gradient neuron-autoencoder layer. IDENTICAL to the NK version.
    Each of `out_features` neurons encodes x->scalar and decodes scalar->x_hat."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.encoder = nn.Linear(in_features, out_features)
        self.dec_w = nn.Parameter(torch.empty(out_features, in_features))
        self.dec_b = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.dec_w, a=5 ** 0.5)

    def forward(self, x):
        return torch.sigmoid(self.encoder(x))           # soma signal -> next layer

    def local_loss(self, x):
        a = torch.sigmoid(self.encoder(x))              # (B,N) per-neuron codes
        x_hat = a.unsqueeze(-1) * self.dec_w.unsqueeze(0) + self.dec_b.unsqueeze(0)
        return F.mse_loss(x_hat, x.unsqueeze(1).expand_as(x_hat))


class SurrogateSpike(torch.autograd.Function):
    """Hard-threshold spike forward, smooth surrogate gradient backward.
    IDENTICAL to stage3_4 — the spiking mechanism is task-agnostic."""
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        sg = 1.0 / (10.0 * torch.abs(v) + 1.0) ** 2
        return grad_output * sg


spike_fn = SurrogateSpike.apply


class SpikingNeuronAELayer(nn.Module):
    """LIF layer where each neuron autoencodes the input from its firing RATE.
    IDENTICAL to stage3_4; the firing rate is the temporal analogue of the
    static scalar code used in the gradient version."""

    def __init__(self, in_features, out_features, T=20, beta=0.9, thresh=1.0):
        super().__init__()
        self.out_features, self.T, self.beta, self.thresh = out_features, T, beta, thresh
        self.fc = nn.Linear(in_features, out_features)
        self.dec_w = nn.Parameter(torch.empty(out_features, in_features))
        self.dec_b = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.dec_w, a=5 ** 0.5)

    def simulate(self, x):
        B = x.size(0)
        current = self.fc(x)
        v = torch.zeros(B, self.out_features, device=x.device)
        spike_sum = torch.zeros(B, self.out_features, device=x.device)
        for _ in range(self.T):
            v = self.beta * v + current
            s = spike_fn(v - self.thresh)
            v = v * (1.0 - s)
            spike_sum = spike_sum + s
        return spike_sum / self.T                        # (B,N) firing rate

    def forward(self, x):
        return self.simulate(x)

    def local_loss(self, x):
        rate = self.simulate(x)
        x_hat = rate.unsqueeze(-1) * self.dec_w.unsqueeze(0) + self.dec_b.unsqueeze(0)
        return F.mse_loss(x_hat, x.unsqueeze(1).expand_as(x_hat))


# ==============================================================================
# THE ONLY GENUINELY NEW CODE: classification heads + MNIST data + training.
# ==============================================================================

# ----- 1. GRADIENT-ONLY MNIST model -------------------------------------------
class GradientNAN_MNIST(nn.Module):
    """
    Neuron-AE MLP for MNIST. The ONLY differences from the NK GradientNAN:
        - head is Linear(hidden, num_classes) instead of Linear(hidden, 1)
        - forward returns raw LOGITS (no output sigmoid); cross-entropy wants logits
    The hidden NeuronAELayer is untouched.
    """

    def __init__(self, in_dim, hidden=256, num_classes=10):
        super().__init__()
        self.hidden = NeuronAELayer(in_dim, hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x, want_local=False):
        local = self.hidden.local_loss(x) if want_local else None
        logits = self.head(self.hidden(x))               # raw logits
        return (logits, local) if want_local else logits


# ----- 2. GRADIENT + SPIKING MNIST model --------------------------------------
class SpikingNAN_MNIST(nn.Module):
    """Spiking neuron-AE for MNIST. Same head swap as above; hidden layer is the
    spiking neuron-AE. Readout is a linear map on the firing-rate vector."""

    def __init__(self, in_dim, hidden=256, num_classes=10, T=20):
        super().__init__()
        self.hidden = SpikingNeuronAELayer(in_dim, hidden, T=T)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x, want_local=False):
        local = self.hidden.local_loss(x) if want_local else None
        logits = self.head(self.hidden(x))               # logits from firing rates
        return (logits, local) if want_local else logits


# ----- 3. MNIST data with the NK->MNIST knob mapping --------------------------
def get_mnist_loaders(resolution=28, num_classes=10, batch_size=128,
                      noise=0.0, root="./data"):
    """
    Build MNIST train/test loaders implementing the two manufactured sweep axes:

        resolution  -> the N-axis (problem size). 7, 14, or 28. Images are
                       downscaled with transforms.Resize, so in_dim = resolution^2.
        num_classes -> part of the K-proxy (hardness). 2 keeps only digits 0/1
                       (easy); 10 is the full task (hard).
        noise       -> the other half of the K-proxy. Gaussian pixel noise std;
                       higher = more tangled input->label mapping.

    Returns (train_loader, test_loader, in_dim). Falls back to a synthetic
    MNIST-shaped dataset if torchvision can't download (e.g. offline sandbox),
    so the training logic is still exercisable; the real run uses true MNIST.
    """
    from torchvision import datasets, transforms

    tf = transforms.Compose([
        transforms.Resize((resolution, resolution)),     # <-- the N-axis knob
        transforms.ToTensor(),                           # -> [0,1], shape (1,res,res)
    ])

    def _subset_and_flatten(ds):
        # keep only the first `num_classes` digit labels, flatten images
        X, y = [], []
        for img, label in ds:
            if label < num_classes:
                X.append(img.view(-1))                   # (res*res,)
                y.append(label)
        X = torch.stack(X); y = torch.tensor(y)
        if noise > 0:
            X = (X + noise * torch.randn_like(X)).clamp(0, 1)   # K-proxy noise
        return TensorDataset(X, y)

    try:
        tr = datasets.MNIST(root, train=True,  download=True, transform=tf)
        te = datasets.MNIST(root, train=False, download=True, transform=tf)
        train_ds = _subset_and_flatten(tr)
        test_ds  = _subset_and_flatten(te)
    except Exception as e:
        # ---- offline fallback: synthetic data of the right shape -----------
        print(f"[get_mnist_loaders] real MNIST unavailable ({type(e).__name__}); "
              f"using synthetic stand-in so the code still runs.")
        in_dim = resolution * resolution
        def _synth(n):
            y = torch.randint(0, num_classes, (n,))
            # give each class a distinct mean pattern so the task is learnable
            X = torch.rand(n, in_dim) * 0.1 + (y.float().unsqueeze(1) / num_classes)
            X = (X + noise * torch.randn_like(X)).clamp(0, 1)
            return TensorDataset(X, y)
        train_ds, test_ds = _synth(2000), _synth(1000)

    in_dim = resolution * resolution
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, in_dim


# ----- 4. training loop: same alternating cycle as NK, classification loss ----
def train_mnist(model, train_loader, test_loader, epochs=3,
                local_weight=1.0, device="cpu"):
    """
    Bull's alternating scheme, MNIST edition. Per batch we coin-flip between:
        TASK cycle  : minimise CROSS-ENTROPY on labels  (was MSE for NK)
        LOCAL cycle : minimise neuron reconstruction MSE (UNCHANGED from NK)
    `local_weight=0` recovers a plain classifier baseline for the ablation.
    Returns final test accuracy (%).
    """
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            if local_weight == 0 or torch.rand(1).item() < 0.5:   # ---- task ----
                opt.zero_grad()
                loss = F.cross_entropy(model(images), labels)     # <-- the change
                loss.backward(); opt.step()
            else:                                                  # ---- local ----
                opt.zero_grad()
                _, local = model(images, want_local=True)
                (local_weight * local).backward(); opt.step()

    # ---- evaluate ----
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


# ==============================================================================
# 5. SWEEP HARNESS — the MNIST analogue of the NK Figure-6 study
# ==============================================================================
def resolution_difficulty_sweep(model_kind="gradient", runs=3, epochs=2,
                                 device="cpu"):
    """
    Reproduce the SHAPE of Bull's conditional finding on MNIST by sweeping:
        resolution (N-axis)  x  num_classes (K-proxy)
    comparing neuron-AE ON (local_weight=1) vs OFF (local_weight=0), averaged
    over `runs` seeds. In a full study you would add a t-test per cell and more
    runs (Bull uses 20); this is the scaffold that produces the table.

    model_kind: "gradient" -> GradientNAN_MNIST ; "spiking" -> SpikingNAN_MNIST
    """
    import numpy as np

    resolutions = [7, 14]          # add 28 for the full N-axis (slower)
    class_counts = [2, 10]         # 2 = easy, 10 = hard (K-proxy)
    Model = GradientNAN_MNIST if model_kind == "gradient" else SpikingNAN_MNIST

    print(f"\n=== {model_kind.upper()} sweep: resolution x difficulty ===")
    print(f"{'res':>4} {'classes':>8} {'AE-off acc':>11} {'AE-on acc':>10} {'delta':>7}")
    table = {}
    for res in resolutions:
        for nc in class_counts:
            accs_off, accs_on = [], []
            for seed in range(runs):
                torch.manual_seed(seed)
                tr, te, in_dim = get_mnist_loaders(resolution=res, num_classes=nc,
                                                   batch_size=128)
                off = train_mnist(Model(in_dim, hidden=128, num_classes=nc),
                                  tr, te, epochs=epochs, local_weight=0.0, device=device)
                torch.manual_seed(seed)   # same init/data ordering for fair pairing
                tr, te, in_dim = get_mnist_loaders(resolution=res, num_classes=nc,
                                                   batch_size=128)
                on = train_mnist(Model(in_dim, hidden=128, num_classes=nc),
                                 tr, te, epochs=epochs, local_weight=1.0, device=device)
                accs_off.append(off); accs_on.append(on)
            mo, mn = np.mean(accs_off), np.mean(accs_on)
            table[(res, nc)] = (mo, mn)
            print(f"{res:>4} {nc:>8} {mo:>11.2f} {mn:>10.2f} {mn-mo:>+7.2f}")
    print("\nInterpretation guide: a POSITIVE delta that grows with resolution "
          "(the N-axis) echoes Bull's 'beneficial at larger N' result. Run with "
          "runs=20 and add scipy.stats.ttest_rel(accs_on, accs_off) per cell.")
    return table


# ==============================================================================
# RUNNER
# ==============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ---- single-shot sanity check of both models -----------------------------
    tr, te, in_dim = get_mnist_loaders(resolution=14, num_classes=10, batch_size=128)

    print("\n=== GRADIENT-ONLY neuron-AE on MNIST ===")
    g = GradientNAN_MNIST(in_dim, hidden=128, num_classes=10)
    acc_g = train_mnist(g, tr, te, epochs=2, local_weight=1.0, device=device)
    print(f"  test accuracy (AE on): {acc_g:.2f}%")

    print("\n=== GRADIENT + SPIKING neuron-AE on MNIST ===")
    s = SpikingNAN_MNIST(in_dim, hidden=128, num_classes=10, T=15)
    acc_s = train_mnist(s, tr, te, epochs=2, local_weight=1.0, device=device)
    print(f"  test accuracy (AE on): {acc_s:.2f}%")

    # ---- the sweep that ties back to the NK Figure-6 study -------------------
    resolution_difficulty_sweep(model_kind="gradient", runs=2, epochs=1, device=device)
    # resolution_difficulty_sweep(model_kind="spiking", runs=2, epochs=1, device=device)
