"""
train_curves.py -- how each group layout gets to where it ends up.

Every Week 10 figure is a single reading taken after the last epoch. That
answers "where does it land" and says nothing about "how does it get there".
Two layouts can finish at the same accuracy having taken completely different
routes, and the end-of-training snapshot cannot tell them apart.

This script trains one model per group layout and records THREE quantities at
the end of EVERY epoch:

    accuracy        on the held-out test set
    recon MSE       each layout's own reconstruction objective
    effective rank  how many independent signals the layer carries

Effective rank is the reason this needs its own training loop rather than a
re-plot of the existing JSON. sweep_hidden.py computes it once, after training;
getting it per epoch means measuring inside the loop. It is computed on a fixed
2,000-image slice with the model in eval mode, exactly as collapse_metrics does
at the end, so the last point of each curve is directly comparable with the
numbers in the Week 10 sweep.

WHAT IT PRODUCES
    results/<dataset>_H<width>_<schedule>/
        curves.png         ONE figure, three stacked panels, one line per
                           layout. This is the deliverable.
        curves.json        every number behind it, for re-plotting
        curves_table.txt   first/last epoch per layout, as text

    Three files. Deliberately. A directory of twenty near-identical PNGs is
    how a result gets lost.

USAGE
    python train_curves.py                                  # H=128, all layouts
    python train_curves.py --hidden 20                      # 20x1 ... 1x20
    python train_curves.py --hidden 128 --layouts nan group8 ann
    python train_curves.py --epochs 40 --seeds 3            # the real run
    python train_curves.py --epochs 2 --seeds 1 --rank_samples 500   # smoke test

    --rank_every N   measure effective rank every N epochs (default 1). The
                     measurement costs one forward pass over 2,000 images, so
                     at default settings it adds a few percent to runtime.

This expects Week_10/ to sit next to this directory, and imports the model and
metric code from there unchanged -- nothing is reimplemented, so these curves
and the Week 10 endpoints cannot drift apart.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# ---- find the Week 10 code ---------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_HERE, "..", "Week_10"),
              os.path.join(_HERE, "Week_10"),
              os.path.join(_HERE, "..")):
    if os.path.isfile(os.path.join(_cand, "group_ae.py")):
        sys.path.insert(0, os.path.abspath(_cand))
        break
else:
    sys.exit("Cannot find group_ae.py. Put this directory next to Week_10/, "
             "or run with PYTHONPATH=/path/to/Week_10")

from group_ae import build_model                    # noqa: E402
from datasets_ext import get_loaders                # noqa: E402
from simple_core import evaluate                    # noqa: E402
from collapse_metrics import (collect_activations,  # noqa: E402
                              _singular_values, effective_rank)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- layouts -----------------------------------------------------------------
def layouts_for(width):
    """Every equal partition of `width`, ANN first, NAN last.

    Returned as (variant, groups, label). `ann` IS group1 and `nan` IS
    group<H> -- exactly, per group_ae's self-test -- so the two ends use the
    same code as the Weeks 5-9 baselines rather than a re-implementation.
    """
    out = []
    for g in sorted(d for d in range(1, width + 1) if width % d == 0):
        variant = "ann" if g == 1 else "nan" if g == width else f"group{g}"
        out.append((variant, g, f"{g}x{width // g}"))
    return out


# ---- per-epoch effective rank ------------------------------------------------
def measure_eff_rank(model, loader, max_samples):
    """Effective rank of the hidden activations, the collapse_metrics way.

    Same function, same eval-mode forward, same fixed slice as the end-of-run
    report -- so an epoch-40 point here means the same thing as the single
    number in the Week 10 tables.
    """
    acts = collect_activations(model, loader, max_samples)
    if acts is None:
        return None
    centred = acts - acts.mean(axis=0, keepdims=True)
    return float(effective_rank(_singular_values(torch.from_numpy(centred))))


# ---- one run -----------------------------------------------------------------
def train_one(variant, args, tr, te, seed):
    """Train one layout for args.epochs, recording metrics after each epoch.

    The training loop is copied from sweep_hidden.train_one so the curves and
    the sweep describe the same procedure. Interleaved only: Week 10 fixed the
    schedule so every comparison sits on one footing, and mixing schedules into
    a curve plot would put two different training budgets on one axis.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(variant, args.res * args.res, args.hidden,
                        args.classes, args.activation).to(DEVICE)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    trains_ae = model.has_ae and not getattr(model, "frozen", False)

    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            if trains_ae and torch.rand(1).item() < args.ae_prob:
                model.local_loss(x).backward()
            else:
                F.cross_entropy(model(x), y).backward()
            opt.step()

        row = evaluate(model, te)
        row["epoch"] = epoch
        row["eff_rank"] = (measure_eff_rank(model, te, args.rank_samples)
                           if epoch % args.rank_every == 0
                           or epoch == args.epochs else None)
        hist.append(row)
    return hist


def aggregate(runs):
    """runs: list over seeds of lists over epochs of metric dicts.

    Returns {metric: {mean: [...], sem: [...]}}, skipping epochs where a metric
    was not measured (effective rank under --rank_every > 1) rather than
    interpolating across the gap.
    """
    out = {}
    for key in ("acc", "recon_mse", "eff_rank"):
        means, sems, epochs = [], [], []
        for i in range(len(runs[0])):
            vals = [r[i][key] for r in runs if r[i].get(key) is not None]
            if not vals:
                continue
            v = np.array(vals, dtype=float)
            epochs.append(runs[0][i]["epoch"])
            means.append(float(v.mean()))
            sems.append(float(v.std() / np.sqrt(len(v))) if len(v) > 1 else 0.0)
        out[key] = {"epochs": epochs, "mean": means, "sem": sems}
    return out


# ---- the one figure ----------------------------------------------------------
NAVY, MUTED, RULE = "#1E2761", "#5B6478", "#D8D8E0"


def plot_curves(results, order, args, path):
    """Three stacked panels sharing an epoch axis, one line per layout.

    Stacked rather than side by side because the shared x-axis is the whole
    point: it lets you read down a column and see what accuracy, reconstruction
    and rank were all doing at the same moment. Colour runs cool-to-warm along
    the ANN -> NAN dial, so the legend order matches the visual order.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12,
        "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": RULE, "grid.alpha": 0.8,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    panels = [
        ("acc",       "test accuracy (%)",  "Accuracy"),
        ("recon_mse", "reconstruction MSE", "Reconstruction error (each layout's own objective)"),
        ("eff_rank",  f"effective rank (of {args.hidden})", "Effective rank of the hidden layer"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 12.8), sharex=True)
    colours = cm.viridis(np.linspace(0.08, 0.92, len(order)))

    for ax, (key, ylabel, title) in zip(axes, panels):
        for (label, _), col in zip(order, colours):
            s = results[label][key]
            if not s["epochs"]:
                continue
            m, e = np.array(s["mean"]), np.array(s["sem"])
            ax.plot(s["epochs"], m, lw=2, color=col, label=label, marker="o",
                    ms=3.5)
            if e.any():
                ax.fill_between(s["epochs"], m - e, m + e, color=col,
                                alpha=0.18, lw=0)
        ax.set_ylabel(ylabel)
        ax.set_title(title, color=NAVY, fontweight="bold", loc="left", pad=8)

    # Rank panel: anchor at 0 so the axis cannot exaggerate a small change, but
    # do not force the ceiling into view when the data sits far below it -- an
    # under-trained run would otherwise be a flat line squashed against zero.
    top = max(v for label, _ in order
              for v in results[label]["eff_rank"]["mean"] or [0])
    if top > 0.4 * args.hidden:
        axes[2].set_ylim(0, args.hidden * 1.12)
        axes[2].axhline(args.hidden, color="#196B24", ls="--", lw=1.4)
        axes[2].text(0.99, 0.955, f"ceiling: H = {args.hidden}",
                     transform=axes[2].transAxes, color="#196B24",
                     fontsize=11, ha="right", va="top")
    else:
        axes[2].set_ylim(0, max(top * 1.3, 1e-6))
        axes[2].text(0.99, 0.955,
                     f"ceiling H = {args.hidden}, off scale above",
                     transform=axes[2].transAxes, color="#196B24",
                     fontsize=11, ha="right", va="top")

    axes[2].set_xlabel("epoch")
    axes[2].xaxis.set_major_locator(
        matplotlib.ticker.MaxNLocator(integer=True))

    # Legend below the panels: with eight layouts it covers curves wherever it
    # is put inside one.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="groups x neurons per group   "
               "(ANN on the left of the dial, NAN on the right)",
               loc="lower center", ncol=min(len(order), 8), frameon=False,
               fontsize=10.5, title_fontsize=10.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Training curves by group layout", color=NAVY, fontsize=16,
                 fontweight="bold", y=0.998)
    fig.text(0.5, 0.976, f"{args.dataset}   |   H = {args.hidden}   |   "
             f"{args.seeds} seed(s)   |   interleaved   |   shaded band = SEM",
             ha="center", color=MUTED, fontsize=11)
    fig.tight_layout(rect=[0, 0.045, 1, 0.968])
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  saved {path}")


def make_table(results, order):
    lines = [f"{'layout':>10} {'variant':10} "
             f"{'acc@1':>8} {'acc@end':>8} "
             f"{'recon@1':>9} {'recon@end':>10} "
             f"{'rank@1':>8} {'rank@end':>9}"]
    lines.append("-" * len(lines[0]))
    for label, variant in order:
        r = results[label]
        def fe(key, i, fmt):
            v = r[key]["mean"]
            return format(v[i], fmt) if v else "n/a"
        lines.append(f"{label:>10} {variant:10} "
                     f"{fe('acc', 0, '8.2f')} {fe('acc', -1, '8.2f')} "
                     f"{fe('recon_mse', 0, '9.4f')} {fe('recon_mse', -1, '10.4f')} "
                     f"{fe('eff_rank', 0, '8.2f')} {fe('eff_rank', -1, '9.2f')}")
    return "\n".join(lines)


# ---- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Per-epoch training curves for every group layout.")
    ap.add_argument("--dataset", default="emnist-balanced")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layouts", nargs="*", default=None,
                    help="variant names, e.g. nan group8 ann. Default: every "
                         "equal partition of --hidden.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--res", type=int, default=28)
    ap.add_argument("--classes", type=int, default=None)
    ap.add_argument("--activation", default="sigmoid")
    ap.add_argument("--ae_prob", type=float, default=0.5)
    ap.add_argument("--rank_samples", type=int, default=2000)
    ap.add_argument("--rank_every", type=int, default=1)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    tr, te, in_dim = get_loaders(dataset=args.dataset, resolution=args.res,
                                 num_classes=args.classes,
                                 batch_size=args.batch, verbose=True)
    args.res = int(round(in_dim ** 0.5))
    if args.classes is None:
        args.classes = int(max(int(y.max()) for _, y in te) + 1)

    all_layouts = layouts_for(args.hidden)
    if args.layouts:
        keep = set(args.layouts)
        all_layouts = [t for t in all_layouts if t[0] in keep]
        missing = keep - {t[0] for t in all_layouts}
        if missing:
            sys.exit(f"not equal partitions of H={args.hidden}: "
                     f"{sorted(missing)}\nAvailable: "
                     f"{[t[0] for t in layouts_for(args.hidden)]}")

    out = args.outdir or os.path.join(
        _HERE, "results", f"{args.dataset}_H{args.hidden}_interleaved")
    os.makedirs(out, exist_ok=True)

    print(f"\ndevice={DEVICE}  H={args.hidden}  epochs={args.epochs}  "
          f"seeds={args.seeds}  layouts={len(all_layouts)}")
    print(f"output -> {out}\n")

    t0 = time.time()
    results, order = {}, []
    for i, (variant, _g, label) in enumerate(all_layouts, 1):
        runs = []
        for seed in range(args.seeds):
            t1 = time.time()
            runs.append(train_one(variant, args, tr, te, seed))
            last = runs[-1][-1]
            print(f"  [{i}/{len(all_layouts)}] {label:>8} ({variant:8}) "
                  f"seed={seed}  acc={last['acc']:.2f}  "
                  f"recon={last['recon_mse']:.4f}  "
                  f"rank={last['eff_rank']:.2f}  [{time.time()-t1:.0f}s]")
        results[label] = aggregate(runs)
        order.append((label, variant))

    with open(os.path.join(out, "curves.json"), "w") as f:
        json.dump({"config": vars(args), "order": order, "results": results},
                  f, indent=1)
    table = make_table(results, order)
    print("\n" + table)
    with open(os.path.join(out, "curves_table.txt"), "w") as f:
        f.write(table + "\n")
    plot_curves(results, order, args, os.path.join(out, "curves.png"))
    print(f"\ndone in {time.time()-t0:.0f}s. Three files in {out}/")


if __name__ == "__main__":
    main()
