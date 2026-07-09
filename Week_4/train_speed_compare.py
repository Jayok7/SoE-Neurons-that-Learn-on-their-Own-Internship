#!/usr/bin/env python3
"""
train_speed_compare.py  ANN vs NAN training-speed comparison (Fashion-MNIST)
============================================================================
A deliberately small training script whose only job is to answer one question:

    How much does the per-neuron autoencoder objective (NAN) slow training
    down, per epoch, compared with a plain ANN baseline of the same size?

For each architecture it trains a HiddenLayerClassifier and, every epoch,
records + prints the same block the FairFace reference script uses:

    Epoch NN/TT  loss=..  acc=..  top2=..  val_loss=..  val_acc=..  val_top2=..
                 recon_mse=..  (Xs)

The recon_mse column is the extra signal for the autoencoder objective. In
this codebase the ANN layer ('ann') has a single shared decoder so it still
reports a reconstruction MSE; the truly autoencoder-free baseline is 'nn',
whose recon is identically zero. Wall-clock seconds per epoch (the "speed" of
the title) are printed and stored in history.

At the end it saves, per architecture:
  - history JSON (all per-epoch lists incl. epoch_time)
  - a training-curves figure (loss | accuracy(+top2) | recon_mse) mirroring
    the reference plot_training_history layout
and a combined ann_vs_nan_speed.png / .json summarising epoch time + final F1.

Usage:
    python train_speed_compare.py                       # ann vs static, defaults
    python train_speed_compare.py --archs ann static spiking
    python train_speed_compare.py --hidden 20 --epochs 15 --res 28 --classes 10
"""

import os
import json
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless / cluster
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from models import get_loaders, HiddenLayerClassifier, DEVICE, FASHION_CLASSES

# f1_score is the one sklearn import we keep  it is the only metric that is
# awkward to write by hand correctly. Everything else (acc, top-2) is trivial.
from sklearn.metrics import f1_score


# ------------------------------------------------------------------------
# Metrics  kept tiny and dependency-free
# ------------------------------------------------------------------------
def topk_correct(logits, labels, k):
    """Number of samples whose true label is within the top-k predictions."""
    topk = logits.topk(k, dim=1).indices            # (B, k)
    return topk.eq(labels.unsqueeze(1)).any(dim=1).sum().item()


# ------------------------------------------------------------------------
# One epoch of training / evaluation
# ------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer):
    """Train for one epoch. Returns (loss, acc, top2, recon_mse)."""
    model.train()
    running_loss = running_recon = 0.0
    correct = top2 = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits, recon = model(x, want_local=True)   # recon is the AE MSE (0 for ann)
        loss = F.cross_entropy(logits, y)
        # NAN trains the classifier AND the per-neuron autoencoder together;
        # this joint objective is exactly the extra work we are timing.
        loss = loss + recon
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        running_loss += F.cross_entropy(logits, y).item() * bs
        running_recon += float(recon.detach()) * bs
        correct += logits.argmax(1).eq(y).sum().item()
        top2 += topk_correct(logits, y, 2)
        total += bs
    return (running_loss / total, correct / total,
            top2 / total, running_recon / total)


@torch.no_grad()
def evaluate(model, loader):
    """Evaluate. Returns (loss, acc, top2, recon_mse, preds, labels)."""
    model.eval()
    running_loss = running_recon = 0.0
    correct = top2 = total = 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, recon = model(x, want_local=True)
        bs = y.size(0)
        running_loss += F.cross_entropy(logits, y).item() * bs
        running_recon += float(recon) * bs
        preds = logits.argmax(1)
        correct += preds.eq(y).sum().item()
        top2 += topk_correct(logits, y, 2)
        total += bs
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    return (running_loss / total, correct / total, top2 / total,
            running_recon / total,
            np.concatenate(all_preds), np.concatenate(all_labels))


# ------------------------------------------------------------------------
# Sample prediction grid  actual garment images captioned with true label,
# top-2 predicted labels + confidences, and a green/red correct-or-not border.
# ------------------------------------------------------------------------
@torch.no_grad()
def plot_prediction_grid(model, loader, arch, res, output_path,
                         n=16, ncols=4):
    """Render a grid of test images with predicted vs true labels + top-2.

    Green title  = top-1 prediction correct.
    Red title    = top-1 wrong (the true class may still appear as the 2nd guess).
    """
    model.eval()
    x, y = next(iter(loader))                       # one test batch
    x, y = x[:n].to(DEVICE), y[:n]
    logits, _ = model(x, want_local=True)
    probs = F.softmax(logits, dim=1)
    top2_p, top2_i = probs.topk(2, dim=1)           # (n, 2) confidences + indices

    x = x.cpu().numpy(); top2_p = top2_p.cpu().numpy(); top2_i = top2_i.cpu().numpy()
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.4, nrows * 2.7))
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        ax = axes[i]; ax.axis("off")
        if i >= n:
            continue
        ax.imshow(x[i].reshape(res, res), cmap="gray")
        true_lab = FASHION_CLASSES[int(y[i])]
        p1, p2 = FASHION_CLASSES[top2_i[i, 0]], FASHION_CLASSES[top2_i[i, 1]]
        c1, c2 = top2_p[i, 0], top2_p[i, 1]
        correct = (top2_i[i, 0] == int(y[i]))
        colour = "green" if correct else "red"
        ax.set_title(
            f"true: {true_lab}\n"
            f"1. {p1} ({c1*100:.0f}%)\n"
            f"2. {p2} ({c2*100:.0f}%)",
            fontsize=8, color=colour,
        )

    fig.suptitle(f"Sample predictions  {arch}  (green=correct, red=wrong)",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.85)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  prediction grid saved -> {output_path}")


# ------------------------------------------------------------------------
# Plotting  mirrors the reference plot_training_history (loss | acc) and
# adds a third recon-mse panel, since that is the NAN-specific signal.
# ------------------------------------------------------------------------
def plot_training_history(history, arch, output_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss ({arch})"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Train Acc", linewidth=2)
    ax2.plot(history["val_acc"], label="Val Acc", linewidth=2)
    ax2.plot(history["train_top2"], label="Train Top-2", linewidth=1.5, linestyle="--")
    ax2.plot(history["val_top2"], label="Val Top-2", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy & Top-2 ({arch})"); ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3.plot(history["train_recon_mse"], label="Train Recon MSE", linewidth=2)
    ax3.plot(history["val_recon_mse"], label="Val Recon MSE", linewidth=2)
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Recon MSE")
    ax3.set_title(f"Autoencoder Recon Loss ({arch})"); ax3.legend(); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  training curves saved -> {output_path}")


def plot_speed_summary(all_hist, output_path):
    """Bar of mean epoch time per arch  the headline 'speed' answer."""
    archs = list(all_hist.keys())
    mean_t = [float(np.mean(all_hist[a]["epoch_time"])) for a in archs]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(archs, mean_t)
    for i, t in enumerate(mean_t):
        ax.text(i, t, f"{t:.2f}s", ha="center", va="bottom")
    ax.set_ylabel("Mean seconds / epoch")
    ax.set_title("Training speed: ANN vs NAN")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"speed summary saved -> {output_path}")


# ------------------------------------------------------------------------
# Train one architecture end-to-end
# ------------------------------------------------------------------------
def train_arch(arch, tr, te, in_dim, args):
    print(f"\n{'='*64}\nARCH: {arch}   (hidden={args.hidden}, epochs={args.epochs})\n{'='*64}")
    model = HiddenLayerClassifier(in_dim, hidden=args.hidden,
                                  num_classes=args.classes, arch=arch).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {
        "train_loss": [], "train_acc": [], "train_top2": [], "train_recon_mse": [],
        "val_loss": [], "val_acc": [], "val_top2": [], "val_recon_mse": [],
        "epoch_time": [],
    }
    best_val_acc = 0.0
    final_preds = final_labels = None

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc, tr_top2, tr_recon = train_one_epoch(model, tr, optimizer)
        va_loss, va_acc, va_top2, va_recon, final_preds, final_labels = evaluate(model, te)
        dt = time.time() - t0

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["train_top2"].append(tr_top2); history["train_recon_mse"].append(tr_recon)
        history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)
        history["val_top2"].append(va_top2); history["val_recon_mse"].append(va_recon)
        history["epoch_time"].append(dt)

        improved = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            improved = " ? best"

        # Same per-epoch display style as the reference script, extended with
        # top-2 and recon_mse.
        print(
            f"  Epoch {epoch+1:02d}/{args.epochs}  "
            f"loss={tr_loss:.4f}  acc={tr_acc:.4f}  top2={tr_top2:.4f}  "
            f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}  val_top2={va_top2:.4f}  "
            f"recon_mse={tr_recon:.4f}  ({dt:.1f}s){improved}"
        )

    macro_f1 = f1_score(final_labels, final_preds, average="macro")
    history["final_val_acc"] = history["val_acc"][-1]
    history["final_val_top2"] = history["val_top2"][-1]
    history["final_macro_f1"] = float(macro_f1)
    history["mean_epoch_time"] = float(np.mean(history["epoch_time"]))

    print(f"  --> final val_acc={history['final_val_acc']:.4f}  "
          f"val_top2={history['final_val_top2']:.4f}  "
          f"macro_F1={macro_f1:.4f}  "
          f"mean_epoch={history['mean_epoch_time']:.2f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"history_{arch}.json"), "w") as f:
        json.dump(history, f, indent=2)
    plot_training_history(history, arch,
                          os.path.join(args.out_dir, f"history_{arch}.png"))
    plot_prediction_grid(model, te, arch, args.res,
                         os.path.join(args.out_dir, f"predictions_{arch}.png"))
    return history


# ------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="ANN vs NAN training-speed comparison")
    p.add_argument("--archs", nargs="+", default=["ann", "static"],
                   choices=["ann", "static", "spiking", "nn"],
                   help="architectures to compare (ann = ANN baseline, "
                        "static/spiking = NAN)")
    p.add_argument("--hidden", type=int, default=128, help="hidden-layer width")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--res", type=int, default=28, help="image resolution (N-axis)")
    p.add_argument("--classes", type=int, default=10, help="garment classes to keep")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out-dir", default=os.environ.get("NAN_OUT_DIR", "speed_compare"))
    args = p.parse_args()

    print(f"DEVICE: {DEVICE}")
    tr, te, in_dim = get_loaders(resolution=args.res, num_classes=args.classes,
                                 batch_size=args.batch_size)
    print(f"in_dim={in_dim}  train_batches={len(tr)}  test_batches={len(te)}")

    all_hist = {arch: train_arch(arch, tr, te, in_dim, args) for arch in args.archs}

    plot_speed_summary(all_hist, os.path.join(args.out_dir, "ann_vs_nan_speed.png"))
    summary = {a: {"mean_epoch_time": all_hist[a]["mean_epoch_time"],
                   "final_val_acc": all_hist[a]["final_val_acc"],
                   "final_val_top2": all_hist[a]["final_val_top2"],
                   "final_macro_f1": all_hist[a]["final_macro_f1"]}
               for a in args.archs}
    with open(os.path.join(args.out_dir, "ann_vs_nan_speed.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*64}\nSUMMARY  (mean epoch time | final val_acc | top2 | macro-F1)\n{'='*64}")
    for a in args.archs:
        s = summary[a]
        print(f"  {a:8s}  {s['mean_epoch_time']:6.2f}s   "
              f"acc={s['final_val_acc']:.4f}  top2={s['final_val_top2']:.4f}  "
              f"F1={s['final_macro_f1']:.4f}")
    if "ann" in args.archs:
        base = summary["ann"]["mean_epoch_time"]
        for a in args.archs:
            if a != "ann":
                ratio = summary[a]["mean_epoch_time"] / base
                print(f"  {a} is {ratio:.2f}x the per-epoch cost of ann")


if __name__ == "__main__":
    main()