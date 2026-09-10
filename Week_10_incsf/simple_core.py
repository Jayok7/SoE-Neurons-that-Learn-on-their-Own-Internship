"""
simple_core.py — shared foundation for the simplified Week 5 scripts.

Purpose: a minimal, rigorously-documented base for the hidden-layer test.
Everything both runner scripts need lives here so they cannot drift apart:
data, models, metrics, configuration recording, and the two 3x3 grids.

Model variants (the hidden-layer question):
    linear : NO hidden layer — input straight to the classification head.
             If this matches the MLP, the hidden layer contributes nothing.
    frozen : hidden layer present but FROZEN at random initialisation; only
             the head trains. This is the reservoir: if the trained MLP only
             matches this, the hidden layer's *training* contributes nothing
             beyond a random projection.
    mlp    : standard MLP, hidden layer trained by the task. The contributor.
    nan    : per-neuron autoencoder hidden layer (NeuronAELayer from models.py).
    ann    : shared-decoder autoencoder hidden layer (LayerAELayer).

Metrics (the only ones recorded): test accuracy, test top-2 accuracy,
test cross-entropy loss, and reconstruction MSE (None for variants with no
autoencoder). All evaluated on the full test set after every epoch.

Reconstruction display: for NAN, each neuron reconstructs the whole image
independently; the grid shows the MEAN over the per-neuron reconstructions
(the same quantity the loss is the mean error of). For ANN it is the shared
decoder's output. Each grid cell shows original | reconstruction side by
side with that image's reconstruction MSE.
"""

import os
import json
import time
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models import (DEVICE, get_loaders, NeuronAELayer, LayerAELayer,
                        FASHION_CLASSES, outdir)
except ImportError:
    from nan.models import (DEVICE, get_loaders, NeuronAELayer, LayerAELayer,
                            FASHION_CLASSES, outdir)


# ==============================================================================
# 1. Models
# ==============================================================================
class LinearClassifier(nn.Module):
    """Hidden layer OFF: input -> head. The no-hidden-layer baseline."""
    has_ae = False

    def __init__(self, in_dim, hidden_unused, num_classes, activation="sigmoid"):
        super().__init__()
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.head(x)


class MLP(nn.Module):
    """Standard one-hidden-layer MLP. freeze_hidden=True is the reservoir
    control: same architecture, same random initialisation, but the hidden
    weights never train — only the head does."""
    has_ae = False

    def __init__(self, in_dim, hidden, num_classes, activation="sigmoid",
                 freeze_hidden=False):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden)
        self.act = torch.sigmoid if activation == "sigmoid" else F.relu
        self.head = nn.Linear(hidden, num_classes)
        if freeze_hidden:
            for p in self.fc.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        return self.head(self.act(self.fc(x)))


class AEClassifier(nn.Module):
    """Autoencoder hidden layer (NAN per-neuron or ANN shared) + linear head.
    Reuses the exact layer classes from models.py so the architecture is
    identical to every earlier experiment."""
    has_ae = True

    def __init__(self, in_dim, hidden, num_classes, kind="nan",
                 activation="sigmoid"):
        super().__init__()
        assert activation == "sigmoid", \
            "NAN/ANN layers are sigmoid by construction (models.py)"
        layer_cls = NeuronAELayer if kind == "nan" else LayerAELayer
        self.kind = kind
        self.hidden = layer_cls(in_dim, hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.head(self.hidden(x))

    def local_loss(self, x):
        return self.hidden.local_loss(x)

    @torch.no_grad()
    def reconstruct(self, x):
        """Image-space reconstruction for display. NAN: mean over the
        per-neuron reconstructions. ANN: the shared decoder's output."""
        code = self.hidden(x)                                # (batch, neurons)
        if self.kind == "ann":
            return self.hidden.decoder(code)                 # (batch, in_dim)
        w, b = self.hidden.decoder_weight, self.hidden.decoder_bias
        recon = code.unsqueeze(-1) * w.unsqueeze(0) + b.unsqueeze(0)
        return recon.mean(dim=1)                             # (batch, in_dim)


VARIANTS = ("linear", "frozen", "mlp", "nan", "ann")


def build_model(variant, in_dim, hidden, num_classes, activation="sigmoid"):
    if variant == "linear":
        return LinearClassifier(in_dim, hidden, num_classes, activation)
    if variant == "frozen":
        return MLP(in_dim, hidden, num_classes, activation, freeze_hidden=True)
    if variant == "mlp":
        return MLP(in_dim, hidden, num_classes, activation)
    if variant in ("nan", "ann"):
        return AEClassifier(in_dim, hidden, num_classes, kind=variant,
                            activation=activation)
    raise ValueError(f"unknown variant {variant}")


# ==============================================================================
# 2. Metrics: accuracy, top-2 accuracy, CE loss, reconstruction MSE
# ==============================================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ce_sum = correct = top2_correct = recon_sum = n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        ce_sum += F.cross_entropy(logits, y, reduction="sum").item()
        top2 = logits.topk(2, dim=1).indices                 # (batch, 2)
        correct += (top2[:, 0] == y).sum().item()
        top2_correct += (top2 == y.unsqueeze(1)).any(dim=1).sum().item()
        if model.has_ae:
            recon_sum += model.local_loss(x).item() * x.size(0)
        n += y.size(0)
    return {"acc": 100.0 * correct / n,
            "top2_acc": 100.0 * top2_correct / n,
            "ce_loss": ce_sum / n,
            "recon_mse": (recon_sum / n) if model.has_ae else None}


def aggregate(per_seed_histories):
    """per_seed_histories: list over seeds of lists over epochs of metric
    dicts. Returns {metric: {mean: [...], sem: [...]}} (None-safe)."""
    keys = per_seed_histories[0][0].keys()
    out = {}
    for k in keys:
        if per_seed_histories[0][0][k] is None:
            out[k] = None
            continue
        v = np.array([[ep[k] for ep in run] for run in per_seed_histories])
        out[k] = {"mean": v.mean(0).tolist(),
                  "sem": (v.std(0) / np.sqrt(v.shape[0])).tolist()}
    return out


# ==============================================================================
# 3. Configuration record — everything needed to reproduce a run exactly
# ==============================================================================
def config_record(args, schedule, extra=None):
    """One dict capturing the full experimental configuration. Saved inside
    every results JSON so no run is ever ambiguous later."""
    cfg = {
        "schedule": schedule,
        "dataset": "Fashion-MNIST",
        "resolution": args.res,
        "input_dim": args.res * args.res,
        "num_classes": args.classes,
        "class_names": FASHION_CLASSES[:args.classes],
        "pixel_range": "[0,1] (torchvision ToTensor; no further normalisation)",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seeds": list(range(args.seeds)),
        "hidden_width": args.hidden,
        "activation": args.activation,
        "optimizer": f"Adam(lr={args.lr}, betas=(0.9,0.999), eps=1e-8, "
                     f"weight_decay=0)",
        "lr": args.lr,
        "loss_task": "cross-entropy (mean over batch)",
        "loss_recon": "MSE between reconstruction and input "
                      "(NAN: mean over batch x neurons x pixels; "
                      "ANN: mean over batch x pixels)",
        "init": "PyTorch defaults for Linear; NAN/ANN decoders "
                "kaiming_uniform (models.py)",
        "eval": "full test set after every epoch",
        "device": DEVICE,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        cfg.update(extra)
    return cfg


def save_json(obj, fname, subdir=None):
    path = os.path.join(outdir(subdir), fname)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  saved {path}")
    return path


# ==============================================================================
# 4. The 3x3 grids
# ==============================================================================
def _first_n_test(loader, n=9):
    """The SAME first n test images every run (deterministic, ordered
    loader), so grids are directly comparable across variants and scripts."""
    X, Y = loader.dataset.tensors
    return X[:n].to(DEVICE), Y[:n].to(DEVICE)


def classification_grid(model, loader, res, path, title, n=9):
    """3x3 grid: image + true label + top-2 predicted classes with
    confidence. Green title = top-1 correct, red = wrong."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = _first_n_test(loader, n)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)
    conf, pred = probs.topk(2, dim=1)

    fig, axes = plt.subplots(3, 3, figsize=(7.5, 8.5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i].cpu().reshape(res, res), cmap="gray",
                  interpolation="nearest")
        ok = pred[i, 0].item() == y[i].item()
        ax.set_title(
            f"true: {FASHION_CLASSES[y[i].item()]}\n"
            f"1. {FASHION_CLASSES[pred[i,0].item()]} ({100*conf[i,0]:.0f}%)\n"
            f"2. {FASHION_CLASSES[pred[i,1].item()]} ({100*conf[i,1]:.0f}%)",
            fontsize=9, color=("green" if ok else "red"))
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def reconstruction_grid(model, loader, res, path, title, n=9):
    """3x3 grid for AE variants: each cell shows original | reconstruction
    side by side, titled with that image's reconstruction MSE."""
    if not model.has_ae:
        print(f"  [reconstruction_grid] skipped — {type(model).__name__} "
              f"has no autoencoder")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, _ = _first_n_test(loader, n)
    model.eval()
    recon = model.reconstruct(x).clamp(0, 1)
    per_image_mse = ((recon - x) ** 2).mean(dim=1)

    fig, axes = plt.subplots(3, 3, figsize=(8.5, 8.0))
    sep = np.ones((res, 2))                       # thin white separator
    for i, ax in enumerate(axes.flat):
        pair = np.hstack([x[i].cpu().reshape(res, res).numpy(), sep,
                          recon[i].cpu().reshape(res, res).numpy()])
        ax.imshow(pair, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"MSE {per_image_mse[i]:.4f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title + "   (left: input, right: reconstruction)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ==============================================================================
# 5. Small shared helpers for the runner scripts
# ==============================================================================
def plot_metric(curves, metric, ylabel, path, title):
    """Overlay of one metric over epochs, one line per variant, SEM shading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if all(agg.get(metric) is None for agg in curves.values()):
        print(f"  [plot_metric] skipped {metric} — no variant records it")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, agg in curves.items():
        if agg.get(metric) is None:
            continue
        m = np.array(agg[metric]["mean"]); s = np.array(agg[metric]["sem"])
        ep = np.arange(1, len(m) + 1)
        ax.plot(ep, m, linewidth=2, label=name)
        ax.fill_between(ep, m - s, m + s, alpha=0.2)
    ax.set_xlabel("epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  saved {path}")


def print_final_table(curves):
    print(f"\n  {'variant':10} {'acc%':>8} {'top2%':>8} {'CE':>8} {'recon':>9}")
    for name, agg in curves.items():
        rm = agg.get("recon_mse")
        rm_s = f"{rm['mean'][-1]:.4f}" if rm else "     n/a"
        print(f"  {name:10} {agg['acc']['mean'][-1]:>8.2f} "
              f"{agg['top2_acc']['mean'][-1]:>8.2f} "
              f"{agg['ce_loss']['mean'][-1]:>8.4f} {rm_s:>9}")
