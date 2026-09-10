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

--------------------------------------------------------------------------
THE DETACHED HEAD  (`detach_head`, added after re-reading Bull 2025)
--------------------------------------------------------------------------
Bull's learning rule partitions the PARAMETERS between the two objectives,
not just the losses:

    "a learning cycle consists of one randomly chosen weight from either the
     hidden layer or output layer being adjusted ... Adjusting a hidden layer
     node weight is here termed an autoencoding cycle and adjusting an output
     layer node is here termed a task learning cycle, since the former is
     minimising decoder error as its objective and the latter is minimising
     regression function error."

So a hidden-layer weight is only ever accepted because it improved
RECONSTRUCTION. The task signal never selects a hidden-layer weight. The
hidden representation is built by the local objective alone and the head
simply reads whatever the neurons independently decided to encode.

Everything in Weeks 5-10 did the opposite: cross-entropy was backpropagated
through `self.head(self.hidden(x))`, so it reached the encoder on every task
batch. Under that arrangement the per-neuron autoencoder is an AUXILIARY
LOSS on top of ordinary backprop, not a local learning rule. Both are worth
studying; only the second is what the paper proposes.

    detach_head=False : Weeks 5-10 behaviour. CE shapes the encoder.
    detach_head=True  : Bull's rule. CE stops at the hidden/head boundary.
                        The head still trains normally on CE; only the
                        gradient flowing BACK into the hidden layer is cut.

Apply it to the AUTOENCODER VARIANTS ONLY. The plain MLP must keep its
task-trained hidden layer — that is the comparison Bull's Figure 7 makes
(reconstruction-only NAN hidden layer vs task-trained NN hidden layer).
Detaching the MLP too turns it into `frozen` and throws the baseline away.
`build_model` enforces this; `apply_detach_head` warns about it.

CONSEQUENCE FOR THE SCHEDULE AXIS. With the head detached, CE gradients
reach only head parameters and reconstruction gradients reach only hidden
parameters — the two objectives have DISJOINT parameter sets. So
`interleaved` and `combined` become the same computation up to update
frequency and the optimiser's per-parameter state, and `lam` stops being a
task/reconstruction trade-off (it becomes an effective learning-rate
multiplier on the encoder-decoder pair). The half-task-budget confound
carried since Week 7 also dissolves: there is no shared budget to split.
`config_record` records `detach_head` so a detached run is never mistaken
for an attached one later.
"""

import os
import json
import time
import platform
import warnings
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
    weights never train — only the head does.

    Deliberately has NO detach_head option. This variant is the task-trained
    baseline that a detached NAN has to beat; detaching it would collapse it
    onto `frozen` and remove the comparison."""
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
    identical to every earlier experiment.

    freeze_hidden=True creates the frozen reservoir variant: same AE
    architecture at random init, but none of the hidden-layer weights
    train — only the classification head learns.  This tests whether the
    AE *structure* (per-neuron decoder for NAN, shared decoder for ANN)
    gives a better random projection than a plain Linear (the existing
    'frozen' MLP variant).

    detach_head=True implements Bull's parameter partition: the hidden layer
    is shaped by its reconstruction objective alone and the head reads it
    without pushing any task gradient back. See the module docstring.
    Note this is orthogonal to freeze_hidden — freeze_hidden stops the hidden
    layer training at all, detach_head only stops the TASK loss training it.
    With freeze_hidden=True the detach flag is a no-op (recorded anyway so
    the config stays honest)."""
    has_ae = True

    def __init__(self, in_dim, hidden, num_classes, kind="nan",
                 activation="sigmoid", freeze_hidden=False,
                 detach_head=False):
        super().__init__()
        assert activation == "sigmoid", \
            "NAN/ANN layers are sigmoid by construction (models.py)"
        layer_cls = NeuronAELayer if kind == "nan" else LayerAELayer
        self.kind = kind
        self.frozen = freeze_hidden
        self.detach_head = detach_head
        self.hidden = layer_cls(in_dim, hidden)
        self.head = nn.Linear(hidden, num_classes)
        if freeze_hidden:
            for p in self.hidden.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        code = self.hidden(x)
        # Forward VALUES are identical either way; only the backward path
        # through this boundary changes, so accuracy is measured on exactly
        # the same network.
        return self.head(code.detach() if self.detach_head else code)

    def local_loss(self, x):
        # Unaffected by detach_head: the reconstruction objective never went
        # through the head in the first place.
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


VARIANTS = ("linear", "frozen", "mlp", "nan", "ann",
            "frozen_nan", "frozen_ann")

# Variants whose hidden layer carries an autoencoder, and which therefore
# (a) are affected by the schedule and (b) are the only ones detach_head
# should ever be applied to. group* and dend* are added by group_ae.py and
# dendritic_ae.py, which extend this set.
AE_VARIANTS = ("nan", "ann", "frozen_nan", "frozen_ann")


def is_ae_variant(variant):
    """True for any variant with a reconstruction objective, including the
    group_ae and dendritic_ae extensions. Use this to gate detach_head so a
    plain MLP baseline is never detached by accident."""
    return (variant in AE_VARIANTS
            or variant.startswith("group")
            or variant.startswith("dend"))


def apply_detach_head(model, detach_head=True, variant=None):
    """Cut the task gradient at the hidden/head boundary of a model that was
    NOT built by this file's build_model — i.e. GroupAEClassifier and
    DendriticAEClassifier, which define their own `forward`.

    Both expose `.hidden` and `.head`, so a forward pre-hook on the head can
    detach its input without touching either file's forward method. Call it
    once, straight after build_model:

        model = build_model(variant, ...)
        if is_ae_variant(variant):
            apply_detach_head(model, args.detach_head, variant)

    Returns the hook handle (or None if nothing was done) so it can be
    removed. Models built here already honour `detach_head` through their own
    forward, so this is a no-op for them and will not double-apply.
    """
    if not detach_head:
        return None
    if variant is not None and not is_ae_variant(variant):
        warnings.warn(
            f"apply_detach_head called on non-AE variant '{variant}'; "
            "refusing. The task-trained MLP is the baseline a detached NAN "
            "has to beat — detaching it turns it into 'frozen'.",
            RuntimeWarning)
        return None
    if getattr(model, "detach_head", False):
        return None                       # AEClassifier already handles it
    if not hasattr(model, "head") or not isinstance(model.head, nn.Module):
        warnings.warn(f"{type(model).__name__} has no `head` submodule; "
                      "detach_head not applied.", RuntimeWarning)
        return None

    def _pre_hook(_module, args):
        return (args[0].detach(),) + tuple(args[1:])

    model.detach_head = True              # so config records it and evaluate
                                          # /repeat calls stay idempotent
    return model.head.register_forward_pre_hook(_pre_hook)


def build_model(variant, in_dim, hidden, num_classes, activation="sigmoid",
                detach_head=False):
    """Build one variant.

    detach_head is applied to the autoencoder variants ONLY and silently
    ignored for linear / frozen / mlp — see is_ae_variant and the module
    docstring for why the MLP baseline must stay attached."""
    if variant == "linear":
        return LinearClassifier(in_dim, hidden, num_classes, activation)
    if variant == "frozen":
        return MLP(in_dim, hidden, num_classes, activation, freeze_hidden=True)
    if variant == "mlp":
        return MLP(in_dim, hidden, num_classes, activation)
    if variant in ("nan", "ann"):
        return AEClassifier(in_dim, hidden, num_classes, kind=variant,
                            activation=activation, detach_head=detach_head)
    if variant == "frozen_nan":
        return AEClassifier(in_dim, hidden, num_classes, kind="nan",
                            activation=activation, freeze_hidden=True,
                            detach_head=detach_head)
    if variant == "frozen_ann":
        return AEClassifier(in_dim, hidden, num_classes, kind="ann",
                            activation=activation, freeze_hidden=True,
                            detach_head=detach_head)
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
    detach = bool(getattr(args, "detach_head", False))
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
        "detach_head": detach,
        "learning_rule": (
            "BULL-FAITHFUL: hidden layer trained by reconstruction ONLY; "
            "the task gradient stops at the hidden/head boundary, so the "
            "encoder is never updated by cross-entropy. Applied to AE "
            "variants only — mlp/linear/frozen keep their usual training. "
            "NOTE: with disjoint parameter sets the interleaved and combined "
            "schedules coincide, and `lam` acts as an encoder-decoder "
            "learning-rate multiplier rather than a task/recon trade-off."
            if detach else
            "AUXILIARY-LOSS: cross-entropy is backpropagated through the "
            "head into the hidden layer, so the encoder is shaped by BOTH "
            "objectives. This is the Weeks 5-10 arrangement and is NOT the "
            "rule described in Bull (2025)."),
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
