#!/usr/bin/env python3
"""
sweep_hidden.py — where does the hidden layer break, and where does it work best?

This is Week 5's simple_interleaved.py turned into a sweep. Everything about a
single run is unchanged (same models, same interleaved schedule, same metrics,
same evaluation protocol from simple_core.py); the only new thing is that the
whole thing is repeated across a list of hidden widths, and a collapse report is
taken at the end of every run.

    for width in --hidden_sizes:
        for variant in --variants:
            for seed in range(--seeds):
                train, evaluate every epoch, collapse-report at the end

The point of the sweep is to separate two curves that are usually conflated:

    accuracy vs width       — the performance curve. Where does it peak, where
                              does it saturate, where does it fall over?
    collapse index vs width — the health curve. At what width do the extra
                              neurons stop being independent?

The interesting result is where those two diverge. If accuracy saturates at
width W but the collapse index keeps rising past it, then the neurons added
beyond W are redundant rather than merely unnecessary — which is the specific
claim the per-neuron autoencoder architecture is supposed to be tested against.
The `mlp` variant is in the default set precisely so that "how different the
hidden neurons are" can be compared against a regular MLP at every width, which
was the open question in the meeting notes.

Dataset: Fashion-MNIST via models.get_loaders. The loader is parameterised by
resolution and class count already, so switching to MNIST or EMNIST later is a
change to get_loaders only, not to this script. --res is passed through, so a
future >28x28 dataset needs no change here either. Nothing below assumes 784
inputs or 10 classes; both come from the loader and from --classes.

USAGE
    python sweep_hidden.py                                   # default sweep
    python sweep_hidden.py --hidden_sizes 2 4 8 16 32 64 128 256 512
    python sweep_hidden.py --variants mlp nan ann --seeds 5 --epochs 15
    python sweep_hidden.py --hidden_sizes 8 64 512 --epochs 3 --seeds 1  # smoke test

OUTPUTS (subdir sweep_hidden/ under $NAN_OUT_DIR)
    results_sweep.json          full config, per-width final metrics, per-width
                                collapse reports (mean + SEM over seeds), and
                                the full per-epoch curves for every run
    sweep_acc.png               test accuracy vs hidden width (log2 x-axis)
    sweep_top2.png              top-2 accuracy vs hidden width
    sweep_ce.png                cross-entropy vs hidden width
    sweep_recon.png             reconstruction MSE vs hidden width (AE variants)
    sweep_collapse_index.png    collapse index vs hidden width
    sweep_eff_rank.png          effective rank vs width, with the y=x ceiling
    sweep_eff_rank_frac.png     effective rank as a fraction of width
    sweep_corr.png              mean |activation correlation| vs width
    sweep_dead.png              fraction of dead neurons vs width
    sweep_acc_vs_collapse.png   the two curves overlaid on twin axes
    summary_table.txt           the final numbers as plain text
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from simple_core import build_model, evaluate, config_record, save_json
from collapse_metrics import collapse_report, aggregate_reports, summarise

try:
    from models import DEVICE, get_loaders, outdir
except ImportError:
    from nan.models import DEVICE, get_loaders, outdir


# ==============================================================================
# Training — identical to Week 5's interleaved schedule
# ==============================================================================
def train_one(variant, hidden, tr, te, args, seed):
    """Train one model at one width. Returns (per-epoch metrics, model).

    The schedule is exactly Week 5's: for AE variants each batch is either a
    task update or a reconstruction update by an independent coin flip at
    --ae_prob; non-AE variants train on the task only. Kept byte-for-byte
    equivalent so sweep results are directly comparable to the Week 5 run at
    hidden=128.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(variant, args.res * args.res, hidden,
                        args.classes, args.activation).to(DEVICE)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    history = []
    for _epoch in range(args.epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            ae_cycle = (model.has_ae
                        and not getattr(model, "frozen", False)
                        and torch.rand(1).item() < args.ae_prob)
            if ae_cycle:
                model.local_loss(x).backward()
            else:
                F.cross_entropy(model(x), y).backward()
            optimizer.step()
        history.append(evaluate(model, te))
    return history, model


# ==============================================================================
# Plotting
# ==============================================================================
def _new_axes(xlabel="hidden layer width (neurons)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.3, which="both")
    return fig, ax


def plot_vs_width(results, widths, key, ylabel, title, path,
                  source="final", ceiling=False):
    """One line per variant: `key` at the final epoch (source='final') or from
    the collapse report (source='collapse'), plotted against hidden width."""
    import matplotlib.pyplot as plt
    fig, ax = _new_axes()
    plotted = False
    for variant in results:
        xs, ys, es = [], [], []
        for w in widths:
            entry = results[variant][str(w)]
            block = entry["final"] if source == "final" else entry["collapse"]
            if block is None:
                continue
            v = block.get(key)
            if v is None:
                continue
            xs.append(w); ys.append(v["mean"]); es.append(v["sem"])
        if not xs:
            continue
        plotted = True
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                    linewidth=2, label=variant)
    if not plotted:
        plt.close(fig)
        print(f"  [plot] skipped {key} — no variant records it")
        return
    if ceiling:
        ax.plot(widths, widths, "k--", alpha=0.4, linewidth=1,
                label="ceiling (rank = width)")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    ax.set_xticks(widths); ax.set_xticklabels([str(w) for w in widths])
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  saved {path}")


def plot_acc_vs_collapse(results, widths, path):
    """Accuracy and collapse index on twin axes — the plot that shows whether
    extra neurons stop helping before or after they stop being independent."""
    import matplotlib.pyplot as plt
    fig, ax = _new_axes()
    ax2 = ax.twinx()
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, variant in enumerate(results):
        c = colours[i % len(colours)]
        xs, acc, coll = [], [], []
        for w in widths:
            e = results[variant][str(w)]
            if e["collapse"] is None:
                continue
            xs.append(w)
            acc.append(e["final"]["acc"]["mean"])
            coll.append(e["collapse"]["collapse_index"]["mean"])
        if not xs:
            continue
        ax.plot(xs, acc, marker="o", color=c, linewidth=2,
                label=f"{variant} acc")
        ax2.plot(xs, coll, marker="s", color=c, linewidth=1.6,
                 linestyle="--", alpha=0.75, label=f"{variant} collapse")
    ax.set_ylabel("test accuracy (%)  [solid]")
    ax2.set_ylabel("collapse index  [dashed]  0 = healthy, 1 = total collapse")
    ax2.set_ylim(0, 1)
    ax.set_title("Performance vs collapse across hidden width")
    ax.set_xticks(widths); ax.set_xticklabels([str(w) for w in widths])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  saved {path}")


# ==============================================================================
# Summary table
# ==============================================================================
def summary_table(results, widths):
    lines = []
    header = (f"{'variant':11} {'width':>6} {'acc%':>8} {'top2%':>8} "
              f"{'CE':>8} {'recon':>9} {'effrank':>9} {'rank/H':>7} "
              f"{'|corr|':>7} {'dead%':>7} {'collapse':>9}")
    for variant in results:
        lines.append("")
        lines.append(header)
        lines.append("-" * len(header))
        for w in widths:
            e = results[variant][str(w)]
            f, c = e["final"], e["collapse"]
            rec = f.get("recon_mse")
            rec_s = f"{rec['mean']:.4f}" if rec else "n/a"
            if c is None:
                lines.append(f"{variant:11} {w:>6} {f['acc']['mean']:>8.2f} "
                             f"{f['top2_acc']['mean']:>8.2f} "
                             f"{f['ce_loss']['mean']:>8.4f} {rec_s:>9} "
                             f"{'n/a':>9} {'n/a':>7} {'n/a':>7} {'n/a':>7} "
                             f"{'n/a':>9}")
                continue
            lines.append(
                f"{variant:11} {w:>6} {f['acc']['mean']:>8.2f} "
                f"{f['top2_acc']['mean']:>8.2f} {f['ce_loss']['mean']:>8.4f} "
                f"{rec_s:>9} {c['eff_rank_act']['mean']:>9.2f} "
                f"{c['eff_rank_act_frac']['mean']:>7.3f} "
                f"{c['mean_abs_corr']['mean']:>7.3f} "
                f"{100 * c['frac_dead']['mean']:>7.1f} "
                f"{c['collapse_index']['mean']:>9.3f}")
    return "\n".join(lines)


# ==============================================================================
# Main
# ==============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--hidden_sizes", type=int, nargs="+",
                   default=[2, 4, 8, 16, 32, 64, 128, 256, 512],
                   help="hidden widths to sweep over")
    p.add_argument("--variants", nargs="+",
                   default=["linear", "frozen", "mlp", "nan", "ann"],
                   choices=["linear", "frozen", "mlp", "nan", "ann",
                            "frozen_nan", "frozen_ann"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--res", type=int, default=28,
                   help="28 = native Fashion-MNIST; raise for larger datasets")
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--activation", default="sigmoid", choices=["sigmoid", "relu"],
                   help="hidden activation for frozen/mlp "
                        "(nan/ann are sigmoid by construction)")
    p.add_argument("--ae_prob", type=float, default=0.5,
                   help="P(batch is a reconstruction update) for nan/ann")
    p.add_argument("--collapse_samples", type=int, default=2000,
                   help="test images used for the collapse metrics")
    p.add_argument("--subdir", default="sweep_hidden")
    args = p.parse_args()

    widths = sorted(args.hidden_sizes)
    t0 = time.time()
    tr, te, in_dim = get_loaders(resolution=args.res, num_classes=args.classes,
                                 batch_size=args.batch_size)
    n_train, n_test = len(tr.dataset), len(te.dataset)
    total = len(widths) * len(args.variants) * args.seeds
    print(f"device={DEVICE} widths={widths} variants={args.variants} "
          f"seeds={args.seeds} epochs={args.epochs} batch={args.batch_size} "
          f"lr={args.lr} res={args.res} act={args.activation} "
          f"ae_prob={args.ae_prob}")
    print(f"train={n_train} test={n_test} | {total} runs total")

    # results[variant][str(width)] = {"final": ..., "collapse": ..., "curves": ...}
    results = {v: {} for v in args.variants}
    run_i = 0
    for w in widths:
        print(f"\n{'=' * 66}\nhidden width = {w}\n{'=' * 66}")
        for variant in args.variants:
            per_seed_hist, per_seed_collapse = [], []
            for seed in range(args.seeds):
                run_i += 1
                hist, model = train_one(variant, w, tr, te, args, seed)
                rep = collapse_report(model, te, args.collapse_samples)
                per_seed_hist.append(hist)
                per_seed_collapse.append(rep)
                m = hist[-1]
                rm = (f"{m['recon_mse']:.4f}"
                      if m["recon_mse"] is not None else "n/a")
                print(f"  [{run_i}/{total}] {variant:11} w={w:<4} seed={seed} "
                      f"acc={m['acc']:.2f} top2={m['top2_acc']:.2f} "
                      f"ce={m['ce_loss']:.4f} recon={rm}")
                print(f"        collapse: {summarise(rep)}")

            # aggregate final-epoch metrics across seeds
            final = {}
            for k in per_seed_hist[0][-1]:
                vals = [h[-1][k] for h in per_seed_hist]
                if any(v is None for v in vals):
                    final[k] = None
                else:
                    a = np.array(vals, dtype=np.float64)
                    final[k] = {"mean": float(a.mean()),
                                "sem": float(a.std() / np.sqrt(len(a)))}

            # full per-epoch curves, kept so a width can be re-plotted later
            curves = {}
            for k in per_seed_hist[0][0]:
                if per_seed_hist[0][0][k] is None:
                    curves[k] = None
                else:
                    a = np.array([[ep[k] for ep in h] for h in per_seed_hist])
                    curves[k] = {"mean": a.mean(0).tolist(),
                                 "sem": (a.std(0) / np.sqrt(a.shape[0])).tolist()}

            results[variant][str(w)] = {
                "final": final,
                "collapse": aggregate_reports(per_seed_collapse),
                "curves": curves,
            }

    # ---- save ----
    args.hidden = widths                          # for the config record
    cfg = config_record(
        args,
        schedule=(f"interleaved (ae_prob={args.ae_prob}) for nan/ann; "
                  f"pure task for linear/frozen/mlp"),
        extra={"experiment": "hidden width sweep",
               "hidden_sizes": widths,
               "ae_prob": args.ae_prob,
               "variants": args.variants,
               "collapse_samples": args.collapse_samples,
               "train_size": n_train, "test_size": n_test,
               "collapse_metric_definitions":
                   "see collapse_metrics.py module docstring"})
    save_json({"config": cfg, "results": results},
              "results_sweep.json", subdir=args.subdir)

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    d = outdir(args.subdir)

    plot_vs_width(results, widths, "acc", "test accuracy (%)",
                  "Accuracy vs hidden width",
                  os.path.join(d, "sweep_acc.png"))
    plot_vs_width(results, widths, "top2_acc", "test top-2 accuracy (%)",
                  "Top-2 accuracy vs hidden width",
                  os.path.join(d, "sweep_top2.png"))
    plot_vs_width(results, widths, "ce_loss", "test cross-entropy",
                  "Cross-entropy vs hidden width",
                  os.path.join(d, "sweep_ce.png"))
    plot_vs_width(results, widths, "recon_mse", "reconstruction MSE",
                  "Reconstruction MSE vs hidden width (AE variants)",
                  os.path.join(d, "sweep_recon.png"))

    plot_vs_width(results, widths, "collapse_index",
                  "collapse index (1 - eff_rank/H)",
                  "Collapse vs hidden width  (0 = healthy, 1 = total collapse)",
                  os.path.join(d, "sweep_collapse_index.png"),
                  source="collapse")
    plot_vs_width(results, widths, "eff_rank_act", "effective rank",
                  "Effective rank of activations vs hidden width",
                  os.path.join(d, "sweep_eff_rank.png"),
                  source="collapse", ceiling=True)
    plot_vs_width(results, widths, "eff_rank_act_frac",
                  "effective rank / width",
                  "Fraction of the layer doing independent work",
                  os.path.join(d, "sweep_eff_rank_frac.png"),
                  source="collapse")
    plot_vs_width(results, widths, "mean_abs_corr",
                  "mean |activation correlation|",
                  "Pairwise neuron overlap vs hidden width",
                  os.path.join(d, "sweep_corr.png"), source="collapse")
    plot_vs_width(results, widths, "frac_dead", "fraction of dead neurons",
                  "Dead (constant-output) neurons vs hidden width",
                  os.path.join(d, "sweep_dead.png"), source="collapse")
    plot_vs_width(results, widths, "mean_abs_cos_dec",
                  "mean |cosine| of per-neuron decoders",
                  "Decoder overlap vs hidden width (NAN only)",
                  os.path.join(d, "sweep_cos_dec.png"), source="collapse")
    plot_acc_vs_collapse(results, widths,
                         os.path.join(d, "sweep_acc_vs_collapse.png"))

    # ---- table ----
    table = summary_table(results, widths)
    print(table)
    with open(os.path.join(d, "summary_table.txt"), "w") as f:
        f.write(table + "\n")
    print(f"\n  saved {os.path.join(d, 'summary_table.txt')}")
    print(f"\ndone in {time.time() - t0:.1f}s. Outputs in {d}/")


if __name__ == "__main__":
    main()
