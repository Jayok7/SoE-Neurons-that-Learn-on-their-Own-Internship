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

THE LEARNING RULE (--detach_head)
By default cross-entropy is backpropagated through the head INTO the hidden
layer, so an AE variant's encoder is shaped by both objectives at once and the
reconstruction acts as an auxiliary loss on top of ordinary backprop.
--detach_head cuts that path, giving Bull's rule instead: the hidden layer is
built by its reconstruction objective alone and the head reads whatever the
neurons independently encoded. It is applied to AE variants only (nan / ann /
group* / dend*); linear / frozen / mlp are untouched, because the task-trained
MLP is the baseline a detached NAN has to beat — detaching it too would
collapse it onto `frozen`. See simple_core's module docstring.

When detached, CE reaches only head parameters and reconstruction only hidden
parameters, so the two schedules coincide up to update frequency and Adam
state, lam becomes an encoder-decoder learning-rate multiplier rather than a
trade-off, and the half-task-budget confound noted above dissolves — there is
no shared budget left to split.

USAGE
    python sweep_hidden.py                                   # interleaved only
    python sweep_hidden.py --schedule interleaved combined    # both, side by side
    python sweep_hidden.py --schedule combined --lam 2.0      # combined only
    python sweep_hidden.py --hidden_sizes 2 4 8 16 32 64 128 256 512
    python sweep_hidden.py --variants mlp nan ann --seeds 5 --epochs 15
    python sweep_hidden.py --detach_head --variants linear frozen mlp nan ann
    python sweep_hidden.py --hidden_sizes 8 64 512 --epochs 3 --seeds 1  # smoke test

OUTPUTS (subdir sweep_hidden_<dataset>/, or sweep_hidden_<dataset>_det/ when
--detach_head is set, under $NAN_OUT_DIR)
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
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# ---- import bootstrap -------------------------------------------------------
# simple_core.py and models.py are expected to sit NEXT TO this script (copy or
# symlink them in). As a fallback, look in the sibling week directories so the
# script still runs if that step was missed. The script's own directory is
# always searched first, so a local copy always wins over a sibling.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_HERE,
                   os.path.join(_HERE, os.pardir, "Week_5"),
                   os.path.join(_HERE, os.pardir, "Week_4")):
    _candidate = os.path.abspath(_candidate)
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.append(_candidate)

from simple_core import (evaluate, config_record, save_json,
                         is_ae_variant, apply_detach_head, apply_task_grad_scale)
# build_model comes from group_ae, which handles the group* variants and
# defers every original variant to simple_core unchanged.
from group_ae import collect_group_stats
# build_model comes from dendritic_ae, which handles dend* variants, defers
# group* to group_ae, and everything else to simple_core unchanged.
from multi_decoder import build_model, collect_multi_stats
from collapse_metrics import collapse_report, aggregate_reports, summarise
from models import decoder_overlap_split
from inhibition import (register_penalties, wrap_inhibition,
                        collect_inhibition_stats, INHIBITIONS)

try:
    from models import DEVICE, outdir, PENALTIES
except ImportError:
    from nan.models import DEVICE, outdir, PENALTIES

# Adds "spectral" (differentiable 1 - effrank/H) to PENALTIES, so it flows
# through the existing --penalty / --penalty_sweep machinery unchanged.
register_penalties()

# get_loaders comes from datasets_ext, which adds MNIST/EMNIST and defers to
# models.get_loaders for fashion so those results stay bit-identical.
from datasets_ext import get_loaders, default_classes, DATASETS


# ==============================================================================
# Training — identical to Week 5's interleaved schedule
# ==============================================================================
def train_one(variant, hidden, tr, te, args, seed, schedule):
    """Train one model at one width under one schedule.

    Returns (per-epoch metrics, model). Both schedules are taken unchanged
    from Week 5 so results stay comparable with earlier runs:

      interleaved : each batch is EITHER a task update OR a reconstruction
                    update, by an independent coin flip at --ae_prob
                    (simple_interleaved.py).
      combined    : every batch optimises CE + lam * recon in one backward
                    pass (simple_combined.py).

    Variants without an autoencoder (linear / frozen / mlp) train on the task
    every batch under both schedules, so they are identical either way and are
    only ever run once.

    Note the schedules differ in task-update budget. Under `interleaved` an AE
    variant gets a classification update on roughly half its batches; under
    `combined` it gets one on every batch, the same as the MLP. That makes
    `combined` the schedule where an AE-vs-MLP accuracy comparison is fair on
    gradient budget, which is exactly the confound flagged in the Week 7 report.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(variant, args.res * args.res, hidden,
                        args.classes, args.activation).to(DEVICE)

    # Bull's rule, applied to AE variants ONLY. group_ae and dendritic_ae
    # define their own forward, so the cut is made with a forward pre-hook on
    # the head rather than by editing those files; simple_core's AEClassifier
    # handles it natively and the helper is a no-op there.
    if getattr(args, "detach_head", False) and is_ae_variant(variant):
        apply_detach_head(model, True, variant)
    elif getattr(args, "beta", 1.0) != 1.0 and is_ae_variant(variant):
        apply_task_grad_scale(model, args.beta, variant)

    # Mutual inhibition, applied AFTER detach: detach acts on the head, this
    # replaces model.hidden with a wrapped version. AE variants only. Must come
    # before the optimizer is built so any new buffers/params are picked up.
    wrap_inhibition(model, getattr(args, "inhibition", "none"),
                    strength=args.inhib_strength, k_frac=args.inhib_k_frac,
                    steps=args.inhib_steps, eta=args.inhib_eta,
                    theta_plus=args.inhib_theta_plus,
                    theta_decay=args.inhib_theta_decay,
                    k_min=args.inhib_k_min, boost=args.inhib_boost,
                    variant=variant)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    trains_ae = model.has_ae and not getattr(model, "frozen", False)

    # Diversity penalty. Applied to the RECONSTRUCTION side only -- it is a
    # statement about how neurons should divide the reconstruction, not about
    # the classification objective. Under `interleaved` that means it is added
    # on reconstruction batches; under `combined` it joins the single loss.
    #
    # `decoder` acts on the per-neuron decoder vectors, so it needs a layer that
    # HAS per-neuron decoders (NAN and the group variants). It is skipped with a
    # warning otherwise rather than silently doing nothing.
    #
    # (`hidden_layer`, not `hidden` — the latter is this function's width
    # argument and was previously shadowed here.)
    pen_fn = PENALTIES.get(args.penalty) if args.penalty != "none" else None
    hidden_layer = getattr(model, "hidden", None)
    if pen_fn is not None and args.penalty == "decoder":
        if hidden_layer is None or not hasattr(hidden_layer, "decoder_weight"):
            pen_fn = None      # e.g. ANN, plain MLP: no per-neuron decoders

    def penalty_term(x):
        if pen_fn is None or args.penalty_weight == 0 or hidden_layer is None:
            return 0.0
        return args.penalty_weight * pen_fn(hidden_layer, x)

    history = []
    for _epoch in range(args.epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            if schedule == "combined":
                loss = F.cross_entropy(model(x), y)
                if trains_ae and args.lam > 0:
                    loss = loss + args.lam * model.local_loss(x) + penalty_term(x)
                loss.backward()
            else:
                if trains_ae and torch.rand(1).item() < args.ae_prob:
                    (model.local_loss(x) + penalty_term(x)).backward()
                else:
                    F.cross_entropy(model(x), y).backward()
            optimizer.step()
        history.append(evaluate(model, te))
    return history, model


def build_run_specs(variants, schedules):
    """Expand variants x schedules into (label, variant, schedule) triples.

    AE variants (nan / ann and their frozen forms) are genuinely affected by
    the schedule and get one run per schedule, labelled `nan-comb` etc. so the
    two appear as separate curves on every plot. Variants with no autoencoder
    train identically either way, so they are run once and keep their plain
    name — this avoids drawing the same baseline twice.
    """
    # is_ae_variant comes from simple_core so this script, collapse_analysis.py
    # and the detach gate can never disagree about what counts as an AE variant.
    is_ae = is_ae_variant
    short = {"interleaved": "int", "combined": "comb"}
    specs, seen = [], set()
    for v in variants:
        if is_ae(v) and len(schedules) > 1:
            for s in schedules:
                specs.append((f"{v}-{short[s]}", v, s))
        elif is_ae(v):
            specs.append((v, v, schedules[0]))
        elif v not in seen:
            seen.add(v)
            specs.append((v, v, schedules[0]))
    return specs


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
    header = (f"{'run':13} {'width':>6} {'acc%':>8} {'top2%':>8} "
              f"{'CE':>8} {'recon':>9} {'effrank':>9} {'rank/H':>7} "
              f"{'|corr|':>7} {'dead%':>7} {'collapse':>9}")
    for label in results:
        lines.append("")
        lines.append(header)
        lines.append("-" * len(header))
        for w in widths:
            e = results[label][str(w)]
            f, c = e["final"], e["collapse"]
            rec = f.get("recon_mse")
            rec_s = f"{rec['mean']:.4f}" if rec else "n/a"
            if c is None:
                lines.append(f"{label:13} {w:>6} {f['acc']['mean']:>8.2f} "
                             f"{f['top2_acc']['mean']:>8.2f} "
                             f"{f['ce_loss']['mean']:>8.4f} {rec_s:>9} "
                             f"{'n/a':>9} {'n/a':>7} {'n/a':>7} {'n/a':>7} "
                             f"{'n/a':>9}")
                continue
            lines.append(
                f"{label:13} {w:>6} {f['acc']['mean']:>8.2f} "
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
                   metavar="V",
                   help="linear frozen mlp nan ann frozen_nan frozen_ann, "
                        "and group<G>[-learn|-hebb] e.g. group8, group8-learn")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--res", type=int, default=28,
                   help="28 = native Fashion-MNIST; raise for larger datasets")
    p.add_argument("--dataset", default="fashion", choices=sorted(DATASETS),
                   help="fashion (default), mnist, emnist-balanced, "
                        "emnist-letters, emnist-digits, emnist-bymerge, "
                        "emnist-byclass")
    p.add_argument("--classes", type=int, default=None,
                   help="defaults to the dataset's own class count")
    p.add_argument("--activation", default="sigmoid", choices=["sigmoid", "relu"],
                   help="hidden activation for frozen/mlp "
                        "(nan/ann are sigmoid by construction)")
    p.add_argument("--ae_prob", type=float, default=0.5,
                   help="interleaved only: P(batch is a reconstruction update)")
    p.add_argument("--schedule", nargs="+", default=["interleaved"],
                   choices=["interleaved", "combined"],
                   help="one or both. Passing both runs each AE variant twice "
                        "and plots them as separate curves")
    p.add_argument("--lam", type=float, default=2.0,
                   help="combined only: weight on the reconstruction term")
    p.add_argument("--beta", type=float, default=1.0,
                   help="fraction of the task gradient that reaches the hidden "
                        "layer. 1.0 attached, 0.0 equivalent to --detach_head, "
                        "in between a minimally attached regime. AE variants only")
    p.add_argument("--detach_head", action="store_true",
                   help="Bull's rule: train AE hidden layers on reconstruction "
                        "ONLY, cutting the task gradient at the hidden/head "
                        "boundary. Applied to nan/ann/group*/dend* only; "
                        "linear/frozen/mlp are untouched")
    p.add_argument("--penalty", default="none",
                   choices=["none", "decoder", "decoder_within",
                            "decoder_between", "activation", "spectral",
                            "ensemble"],
                   help="diversity penalty added to the reconstruction loss. "
                        "'decoder' pushes per-neuron decoder vectors apart "
                        "(NAN/group only); 'activation' decorrelates "
                        "activations; 'spectral' is 1 - effrank/H, i.e. the "
                        "reported collapse index used directly as a loss")
    p.add_argument("--inhibition", default="none", choices=list(INHIBITIONS),
                   help="mutual inhibition on the hidden layer. 'kwta' keeps "
                        "the top k activations per image; 'lateral' adds "
                        "anti-Hebbian recurrent inhibition with an adaptive "
                        "threshold. AE variants only")
    p.add_argument("--inhib_strength", type=float, default=1.0,
                   help="lateral only: scale on the recurrent inhibition term")
    p.add_argument("--inhib_k_frac", type=float, default=0.1,
                   help="kwta: fraction of neurons kept per image. "
                        "lateral: fraction counted as winners for theta")
    p.add_argument("--inhib_k_min", type=int, default=2,
                   help="kwta: floor on k, so a narrow layer does not round "
                        "down to a single winner")
    p.add_argument("--inhib_boost", type=float, default=1.0,
                   help="kwta: duty-cycle boosting strength. A neuron below "
                        "its fair share of wins has its SELECTION score "
                        "multiplied by exp(boost * relative deficit); its "
                        "output value is never boosted. 0 disables it. "
                        "Measured on structured data: dead fraction 82%% at 0, "
                        "10.9%% at 1.0, back to 50.8%% at 2.0")
    p.add_argument("--inhib_steps", type=int, default=3,
                   help="lateral only: settling iterations per forward pass")
    p.add_argument("--inhib_eta", type=float, default=0.01,
                   help="lateral only: anti-Hebbian update rate for M")
    p.add_argument("--inhib_theta_plus", type=float, default=0.05,
                   help="lateral only: threshold increment for winners. "
                        "RAISE THIS if effective rank collapses toward 1")
    p.add_argument("--inhib_theta_decay", type=float, default=0.9,
                   help="lateral only: multiplicative threshold decay")
    p.add_argument("--penalty_weight", type=float, default=0.0,
                   help="strength of the diversity penalty. 0 disables it")
    p.add_argument("--penalty_sweep", type=float, nargs="+", default=None,
                   help="sweep these penalty weights instead of a single value. "
                        "ALWAYS include 0.0 as the unpenalised control")
    p.add_argument("--collapse_samples", type=int, default=2000,
                   help="test images used for the collapse metrics")
    p.add_argument("--subdir", default=None,
                   help="default: sweep_hidden_<dataset>")
    args = p.parse_args()
    if args.classes is None:
        args.classes = default_classes(args.dataset)
    if args.subdir is None:
        # _det keeps a detached sweep from overwriting the attached results
        # it is meant to be compared against.
        args.subdir = (f"sweep_hidden_{args.dataset}"
                       + ("_det" if args.detach_head else
                          ("" if args.beta == 1.0 else f"_b{args.beta}"))
                       + ("" if args.inhibition == "none"
                          else f"_{args.inhibition}"))

    widths = sorted(args.hidden_sizes)
    schedules = list(dict.fromkeys(args.schedule))     # de-dup, keep order
    specs = build_run_specs(args.variants, schedules)

    # A penalty sweep multiplies the run list: each variant is run once per
    # weight, labelled `nan@p0.01`, so every weight appears as its own curve.
    weights = args.penalty_sweep
    if weights is not None:
        weights = list(dict.fromkeys(weights))
        if 0.0 not in weights:
            print("[warn] --penalty_sweep has no 0.0 entry; adding the "
                  "unpenalised control, without which nothing is interpretable")
            weights = [0.0] + weights
        if args.penalty == "none":
            raise SystemExit("--penalty_sweep needs --penalty decoder|activation")
        specs = [(f"{lbl}@p{w:g}", v, sc, w)
                 for (lbl, v, sc) in specs for w in weights]
    else:
        specs = [(lbl, v, sc, args.penalty_weight) for (lbl, v, sc) in specs]
    t0 = time.time()
    tr, te, in_dim = get_loaders(dataset=args.dataset, resolution=args.res,
                                 num_classes=args.classes,
                                 batch_size=args.batch_size)
    n_train, n_test = len(tr.dataset), len(te.dataset)
    total = len(widths) * len(specs) * args.seeds
    print(f"device={DEVICE} widths={widths} variants={args.variants} "
          f"schedules={schedules} seeds={args.seeds} epochs={args.epochs} "
          f"batch={args.batch_size} lr={args.lr} res={args.res} "
          f"act={args.activation} ae_prob={args.ae_prob} lam={args.lam} "
          f"detach_head={args.detach_head} inhibition={args.inhibition}")
    if args.inhibition != "none" or args.penalty == "spectral":
        print("  [inhibition] READ RANK AND ACCURACY TOGETHER. Week 7 found "
              "the frozen random layer had the highest effective rank in the "
              "sweep and the worst accuracy, so a mechanism that raises rank "
              "while lowering accuracy has pushed the layer toward the frozen "
              "regime and has failed, however good the rank looks.")
    if args.detach_head:
        print("  [detach_head] AE hidden layers trained by reconstruction "
              "ONLY; linear/frozen/mlp unchanged.")
        if len(schedules) > 1:
            print("  [detach_head] WARNING: both schedules requested, but with "
                  "disjoint parameter sets they should coincide. Expect the "
                  "two curves to overlap; a real gap means task gradient is "
                  "still reaching the encoder.")
    print(f"runs: {', '.join(lbl for lbl, _, _, _ in specs)}")
    print(f"train={n_train} test={n_test} | {total} runs total")

    # results[label][str(width)] = {"final": ..., "collapse": ..., "curves": ...}
    results = {lbl: {} for lbl, _, _, _ in specs}
    run_i = 0
    for w in widths:
        print(f"\n{'=' * 66}\nhidden width = {w}\n{'=' * 66}")
        for label, variant, schedule, pw in specs:
            args.penalty_weight = pw          # used inside train_one
            per_seed_hist, per_seed_collapse = [], []
            for seed in range(args.seeds):
                run_i += 1
                hist, model = train_one(variant, w, tr, te, args, seed,
                                        schedule)
                rep = collapse_report(model, te, args.collapse_samples)
                if seed == 0:
                    gstats = collect_group_stats(model)   # None for non-group
                    istats = collect_inhibition_stats(model)
                    dsplit = decoder_overlap_split(getattr(model, "hidden", None))
                    mstats = collect_multi_stats(model)   # None unless multi*
                per_seed_hist.append(hist)
                per_seed_collapse.append(rep)
                m = hist[-1]
                rm = (f"{m['recon_mse']:.4f}"
                      if m["recon_mse"] is not None else "n/a")
                print(f"  [{run_i}/{total}] {label:12} w={w:<4} seed={seed} "
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

            results[label][str(w)] = {
                "variant": variant,
                "schedule": schedule,
                "penalty": args.penalty,
                "penalty_weight": pw,
                "final": final,
                "collapse": aggregate_reports(per_seed_collapse),
                "group_stats": gstats,          # seed 0; None for non-group runs
                "inhibition_stats": istats,
                "decoder_split": dsplit,
                "ensemble_stats": mstats,       # seed 0; None unless multi*
                "curves": curves,
            }

    # ---- save ----
    args.hidden = widths                          # for the config record
    cfg = config_record(
        args,
        schedule=(f"schedules={schedules}; interleaved uses ae_prob="
                  f"{args.ae_prob}, combined uses CE + {args.lam}*recon on "
                  f"every batch; linear/frozen/mlp are pure task either way"
                  + ("; AE hidden layers DETACHED from the task gradient, so "
                     "the two schedules coincide and lam acts as an "
                     "encoder-decoder lr multiplier"
                     if args.detach_head else "")),
        extra={"experiment": "hidden width sweep",
               "dataset": args.dataset,
               "hidden_sizes": widths,
               "ae_prob": args.ae_prob,
               "lam": args.lam,
               "detach_head": args.detach_head,
               "task_beta": (0.0 if args.detach_head else args.beta),
               "inhibition": args.inhibition,
               "inhibition_params": {
                   "strength": args.inhib_strength,
                   "k_frac": args.inhib_k_frac,
                   "k_min": args.inhib_k_min,
                   "boost": args.inhib_boost,
                   "steps": args.inhib_steps,
                   "eta": args.inhib_eta,
                   "theta_plus": args.inhib_theta_plus,
                   "theta_decay": args.inhib_theta_decay},
               "schedules": schedules,
               "variants": args.variants,
               "penalty": args.penalty,
               "penalty_weights": weights if weights is not None
                                  else [args.penalty_weight],
               "run_labels": {lbl: {"variant": v, "schedule": s,
                                    "penalty_weight": w}
                              for lbl, v, s, w in specs},
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
