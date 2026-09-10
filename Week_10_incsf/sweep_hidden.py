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
    python sweep_hidden.py                                   # interleaved only
    python sweep_hidden.py --schedule interleaved combined    # both, side by side
    python sweep_hidden.py --schedule combined --lam 2.0      # combined only
    python sweep_hidden.py --hidden_sizes 2 4 8 16 32 64 128 256 512
    python sweep_hidden.py --variants mlp nan ann --seeds 5 --epochs 15
    python sweep_hidden.py --hidden_sizes 8 64 512 --epochs 3 --seeds 1  # smoke test

THE TWO WEEK 10 SWEEPS

  1. GROUP SWEEP — the partition of a fixed budget of neurons into shared
     decoders. At H = 20: 20x1, 10x2, 5x4, 4x5, 2x10, 1x20 (groups x units).
     `nan` IS group20 and `ann` IS group1, exactly (proven in group_ae's
     self-test), so the two endpoints come from the code already used in
     Weeks 5-9 rather than from a re-implementation. Parameter-neutral at
     every G, so this is not secretly a capacity sweep.

         python sweep_hidden.py --hidden_sizes 20 \
             --variants nan group10 group5 group4 group2 ann

  2. PENALTY SWEEP — the diversity penalty turned up from 0, reading off
     accuracy, effective rank and reconstruction MSE at each weight. p=0 is
     the control every other point is read against and is added automatically
     if omitted. Outputs land on a penalty x-axis (penalty_*.png), not the
     width x-axis, since every run sits at one width.

         python sweep_hidden.py --hidden_sizes 20 --variants nan group5 \
             --penalty decoder --penalty_sweep 0 0.01 0.1 1 10

OUTPUTS (subdir sweep_hidden_<dataset>/ under $NAN_OUT_DIR)
    results_sweep.json          full config, per-run final metrics, per-run
                                collapse reports (mean + SEM over seeds), and
                                the full per-epoch curves for every run
    summary_table.txt           the final numbers as plain text

    GROUP AXIS (x = partition of a fixed H, `G x units`; 1xH is ANN, Hx1 is NAN)
    groups_acc.png              test accuracy vs partition
    groups_recon.png            reconstruction MSE vs partition
    groups_eff_rank.png         effective rank of activations vs partition
    groups_eff_rank_frac.png    effective rank / width vs partition
    groups_collapse_index.png   collapse index vs partition
    groups_cos_dec.png          per-neuron decoder overlap vs partition
    groups_corr.png             mean |activation correlation| vs partition
    group_table.txt             the same numbers ordered by partition

    PENALTY AXIS (x = penalty weight; only with --penalty_sweep)
    penalty_acc.png             test accuracy vs penalty weight
    penalty_recon.png           reconstruction MSE vs penalty weight
    penalty_eff_rank.png        effective rank vs penalty weight
    penalty_eff_rank_frac.png   effective rank / width vs penalty weight
    penalty_cos_dec.png         per-neuron decoder overlap vs penalty weight
    penalty_table.txt           acc, recon and effective rank in one table

    WIDTH AXIS (sweep_*.png) — ONLY when --hidden_sizes has more than one entry.
    Week 10's sweeps hold the neuron budget fixed, so these are skipped: with a
    single width they would be one point per variant. They remain the right
    plots for a Week 7-9 style width sweep.
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

from simple_core import evaluate, config_record, save_json
# build_model comes from group_ae, which handles the group* variants and
# defers every original variant to simple_core unchanged.
from group_ae import build_model, collect_group_stats, parse_group_variant
from collapse_metrics import collapse_report, aggregate_reports, summarise

try:
    from models import DEVICE, outdir, PENALTIES
except ImportError:
    from nan.models import DEVICE, outdir, PENALTIES

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
    pen_fn = PENALTIES.get(args.penalty) if args.penalty != "none" else None
    hid_layer = getattr(model, "hidden", None)     # NOT `hidden`: that is the width
    if pen_fn is not None and args.penalty == "decoder":
        # hasattr("decoder_weight") is NOT the test. LayerAELayer (ANN) aliases
        # decoder_weight to its shared decoder's (in_dim, H) matrix so the
        # diagnostics read something, and penalising THAT decorrelates input
        # dimensions, not neurons -- a different penalty wearing the same name.
        # Use the same opt-in rule collapse_metrics._per_neuron_decoder uses,
        # so the penalty and the metric act on the same object.
        per_neuron = (getattr(hid_layer, "per_neuron_decoder", False)
                      or getattr(model, "kind", None) == "nan")
        if hid_layer is None or not per_neuron:
            pen_fn = None      # ANN, plain MLP, linear: no per-neuron decoders
            if seed == 0 and args.penalty_weight != 0:
                print(f"  [warn] {variant} has no per-neuron decoders; the "
                      f"decoder penalty is inactive for it. Its curve is the "
                      f"unpenalised model repeated at every weight.")

    def penalty_term(x):
        if pen_fn is None or args.penalty_weight == 0 or hid_layer is None:
            return 0.0
        return args.penalty_weight * pen_fn(hid_layer, x)

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
    ae_variants = {"nan", "ann", "frozen_nan", "frozen_ann"}
    is_ae = lambda v: v in ae_variants or v.startswith("group")
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


def groups_for(variant, width):
    """How many groups G this variant's hidden layer uses at this width.

    `nan` IS group<H> and `ann` IS group1 — exactly, not approximately; that is
    what group_ae's self-test proves. So both sit on the group axis at its two
    ends and there is no need to run group<H>/group1 separately. Variants with
    no group structure at all (mlp, linear, frozen) return None and are left
    off the axis rather than parked at an arbitrary G.
    """
    if variant == "nan":
        return width
    if variant == "ann":
        return 1
    spec = parse_group_variant(variant)
    return spec[0] if spec else None


def _group_series_name(label, variant, width, widths, pw_text):
    """Legend entry for a group-axis line: everything that is NOT G."""
    spec = parse_group_variant(variant)
    routing = spec[1] if spec else "fixed"
    parts = []
    if routing != "fixed":
        parts.append(routing)
    suffix = label.partition("@p")[0][len(variant):].lstrip("-")   # -int / -comb
    if suffix:
        parts.append(suffix)
    if len(widths) > 1:
        parts.append(f"H={width}")
    if pw_text:
        parts.append(f"p={pw_text}")
    return ", ".join(parts) or "fixed partition"


def plot_vs_groups(results, widths, key, ylabel, title, path, source="final"):
    """`key` against the NUMBER OF GROUPS, which is what this sweep varies.

    The total neuron count is held fixed, so width is a constant here and the
    width-axis plots degenerate to one point per variant. The x-axis is G, tick-
    labelled `G x k` (groups x units per group) at a single width: 1x20 on the
    left is ANN, 20x1 on the right is NAN, and everything between is the
    interpolation. Evenly spaced positions rather than a log axis, since the
    divisors of H are not evenly spaced and every partition deserves equal
    room.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.grid(alpha=0.3)
    series = {}
    for label in results:
        _, _, pw_text = label.partition("@p")
        for width in widths:
            entry = results[label].get(str(width))
            if entry is None:
                continue
            variant = entry["variant"]
            g = groups_for(variant, width)
            if g is None:
                continue
            block = entry["final"] if source == "final" else entry["collapse"]
            if block is None or block.get(key) is None:
                continue
            name = _group_series_name(label, variant, width, widths, pw_text)
            series.setdefault(name, []).append(
                (g, width, block[key]["mean"], block[key]["sem"]))
    if not series:
        plt.close(fig)
        print(f"  [plot] skipped {key} vs groups — no group-axis run records it")
        return

    all_g = sorted({g for pts in series.values() for g, _, _, _ in pts})
    if len(all_g) < 2:
        plt.close(fig)
        print(f"  [plot] skipped {key} vs groups — only one partition present")
        return
    pos = {g: i for i, g in enumerate(all_g)}
    for name, pts in sorted(series.items()):
        pts.sort()
        ax.errorbar([pos[g] for g, _, _, _ in pts],
                    [m for _, _, m, _ in pts],
                    yerr=[s for _, _, _, s in pts],
                    marker="o", capsize=3, linewidth=2, label=name)
    ax.set_xticks(range(len(all_g)))
    if len(widths) == 1:
        h = widths[0]
        ax.set_xticklabels([f"{g}x{h // g}" for g in all_g])
        ax.set_xlabel(f"partition of H = {h}:  groups x units per group   "
                      f"(left = ANN, right = NAN)")
    else:
        ax.set_xticklabels([str(g) for g in all_g])
        ax.set_xlabel("number of groups G   (G = 1 is ANN, G = H is NAN)")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  saved {path}")


def group_table(results, widths):
    """The group sweep as text, ordered by partition rather than by variant
    name, so the interpolation reads down the page in one direction."""
    rows = {}
    for label in results:
        _, _, pw_text = label.partition("@p")
        for width in widths:
            e = results[label].get(str(width))
            if e is None:
                continue
            g = groups_for(e["variant"], width)
            if g is None:
                continue
            name = _group_series_name(label, e["variant"], width, widths,
                                      pw_text)
            rows.setdefault(name, []).append((g, width, label, e))
    if not rows or all(len({g for g, _, _, _ in v}) < 2 for v in rows.values()):
        return ""
    lines = []
    header = (f"{'partition':>10} {'run':13} {'acc%':>8} {'recon':>9} "
              f"{'effrank':>9} {'rank/H':>7} {'|cos|dec':>9} {'collapse':>9}")
    for name, entries in sorted(rows.items()):
        lines.append("")
        lines.append(f"[{name}]")
        lines.append(header)
        lines.append("-" * len(header))
        for g, width, label, e in sorted(entries):
            f, c = e["final"], e["collapse"]
            rec = f.get("recon_mse")
            rec_s = f"{rec['mean']:.4f}" if rec else "n/a"
            part = f"{g}x{width // g}"
            if c is None:
                lines.append(f"{part:>10} {label:13} {f['acc']['mean']:>8.2f} "
                             f"{rec_s:>9} {'n/a':>9} {'n/a':>7} {'n/a':>9} "
                             f"{'n/a':>9}")
                continue
            cos = c.get("mean_abs_cos_dec")
            cos_s = f"{cos['mean']:.3f}" if cos else "n/a"
            lines.append(f"{part:>10} {label:13} {f['acc']['mean']:>8.2f} "
                         f"{rec_s:>9} {c['eff_rank_act']['mean']:>9.2f} "
                         f"{c['eff_rank_act_frac']['mean']:>7.3f} "
                         f"{cos_s:>9} {c['collapse_index']['mean']:>9.3f}")
    return "\n".join(lines)


def plot_vs_penalty(results, widths, weights, key, ylabel, title, path,
                    source="final"):
    """`key` against DIVERSITY-PENALTY WEIGHT, one line per variant.

    The width-axis plots are useless for a penalty sweep: every run sits at the
    same width, so each `nan@p0.1` becomes a one-point curve. This is the plot
    the penalty sweep is actually for — accuracy, effective rank and recon MSE
    as the penalty is turned up, with the p=0 control at the left-hand end.

    Weights are plotted at EVENLY SPACED positions with the weight as the tick
    label, not on a log axis: 0 has no place on a log axis and it is the one
    point every other point has to be read against.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.grid(alpha=0.3)
    pos = {w: i for i, w in enumerate(weights)}
    plotted = False
    series = {}
    for label in results:
        base, _, wtxt = label.partition("@p")
        if not wtxt:
            continue                       # not part of the sweep
        for width in widths:
            entry = results[label].get(str(width))
            if entry is None:
                continue
            block = entry["final"] if source == "final" else entry["collapse"]
            if block is None or block.get(key) is None:
                continue
            line = f"{base} (H={width})" if len(widths) > 1 else base
            xs, ys, es = series.setdefault(line, ([], [], []))
            xs.append(pos[entry["penalty_weight"]])
            ys.append(block[key]["mean"]); es.append(block[key]["sem"])
    for line, (xs, ys, es) in sorted(series.items()):
        order = np.argsort(xs)
        ax.errorbar(np.array(xs)[order], np.array(ys)[order],
                    yerr=np.array(es)[order], marker="o", capsize=3,
                    linewidth=2, label=line)
        plotted = True
    if not plotted:
        plt.close(fig)
        print(f"  [plot] skipped {key} vs penalty — no run records it")
        return
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels([f"{w:g}" for w in weights])
    ax.set_xlabel("diversity penalty weight  (0 = unpenalised control)")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  saved {path}")


def penalty_table(results, widths, weights):
    """The penalty sweep as text: accuracy, effective rank and recon MSE side
    by side, so the trade-off between them is readable without the plots."""
    lines = []
    header = (f"{'variant':12} {'width':>6} {'penalty':>8} {'acc%':>8} "
              f"{'recon':>9} {'effrank':>9} {'rank/H':>7} {'|cos|dec':>9} "
              f"{'collapse':>9}")
    rows = {}
    for label in results:
        base, _, wtxt = label.partition("@p")
        if not wtxt:
            continue
        for width in widths:
            e = results[label].get(str(width))
            if e is None:
                continue
            rows.setdefault((base, width), []).append((e["penalty_weight"], e))
    if not rows:
        return ""
    for (base, width), entries in sorted(rows.items()):
        lines.append("")
        lines.append(header)
        lines.append("-" * len(header))
        for pw, e in sorted(entries):
            f, c = e["final"], e["collapse"]
            rec = f.get("recon_mse")
            rec_s = f"{rec['mean']:.4f}" if rec else "n/a"
            if c is None:
                lines.append(f"{base:12} {width:>6} {pw:>8g} "
                             f"{f['acc']['mean']:>8.2f} {rec_s:>9} "
                             f"{'n/a':>9} {'n/a':>7} {'n/a':>9} {'n/a':>9}")
                continue
            cos = c.get("mean_abs_cos_dec")
            cos_s = f"{cos['mean']:.3f}" if cos else "n/a"
            lines.append(f"{base:12} {width:>6} {pw:>8g} "
                         f"{f['acc']['mean']:>8.2f} {rec_s:>9} "
                         f"{c['eff_rank_act']['mean']:>9.2f} "
                         f"{c['eff_rank_act_frac']['mean']:>7.3f} "
                         f"{cos_s:>9} "
                         f"{c['collapse_index']['mean']:>9.3f}")
    return "\n".join(lines)


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
    p.add_argument("--penalty", default="none",
                   choices=["none", "decoder", "activation"],
                   help="diversity penalty added to the reconstruction loss. "
                        "'decoder' pushes per-neuron decoder vectors apart "
                        "(NAN/group only); 'activation' decorrelates activations")
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
        args.subdir = f"sweep_hidden_{args.dataset}"

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
          f"act={args.activation} ae_prob={args.ae_prob} lam={args.lam}")
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
                "curves": curves,
            }

    # ---- save ----
    args.hidden = widths                          # for the config record
    cfg = config_record(
        args,
        schedule=(f"schedules={schedules}; interleaved uses ae_prob="
                  f"{args.ae_prob}, combined uses CE + {args.lam}*recon on "
                  f"every batch; linear/frozen/mlp are pure task either way"),
        extra={"experiment": "hidden width sweep",
               "dataset": args.dataset,
               "hidden_sizes": widths,
               "ae_prob": args.ae_prob,
               "lam": args.lam,
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

    # ---- width axis: only meaningful if more than one width was run --------
    # Week 10's sweeps hold the neuron budget fixed, so `widths` has one entry
    # and every one of these would be a single point per variant. They are the
    # right plots for a Week 7-9 style width sweep and are kept for that, but
    # drawing them here would just be clutter that invites misreading.
    if len(widths) > 1:
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
    else:
        print(f"  [plot] one width ({widths[0]}) — width-axis plots skipped; "
              f"the sweep axis is the group partition")

    # ---- group axis: the axis the group sweep actually varies --------------
    plot_vs_groups(results, widths, "acc", "test accuracy (%)",
                   "Accuracy vs group partition",
                   os.path.join(d, "groups_acc.png"))
    # NOTE on reading groups_recon.png: this is each model's OWN objective, and
    # the objectives differ across the axis. At G groups the loss averages G
    # reconstructions, each built from only H/G neurons; at G = H (NAN) it
    # averages H single-neuron reconstructions; at G = 1 (ANN) it is one
    # reconstruction from all H. Fewer, larger groups therefore have an easier
    # target and lower MSE almost by construction. The curve is worth having --
    # it says what each objective achieves at its own task -- but a drop from
    # 20x1 to 2x10 is NOT evidence that the 2x10 layer reconstructs better in
    # any shared sense. The activation-side metrics below are the comparable
    # ones, and accuracy is the common yardstick.
    plot_vs_groups(results, widths, "recon_mse", "reconstruction MSE",
                   "Reconstruction MSE vs group partition (own objective)",
                   os.path.join(d, "groups_recon.png"))
    plot_vs_groups(results, widths, "eff_rank_act",
                   "effective rank of activations",
                   "Effective rank vs group partition",
                   os.path.join(d, "groups_eff_rank.png"), source="collapse")
    plot_vs_groups(results, widths, "eff_rank_act_frac",
                   "effective rank / width",
                   "Fraction of the layer doing independent work",
                   os.path.join(d, "groups_eff_rank_frac.png"),
                   source="collapse")
    plot_vs_groups(results, widths, "collapse_index",
                   "collapse index (1 - eff_rank/H)",
                   "Collapse vs group partition  (0 = healthy, 1 = total)",
                   os.path.join(d, "groups_collapse_index.png"),
                   source="collapse")
    plot_vs_groups(results, widths, "mean_abs_cos_dec",
                   "mean |cosine| of per-neuron decoders",
                   "Decoder overlap vs group partition",
                   os.path.join(d, "groups_cos_dec.png"), source="collapse")
    plot_vs_groups(results, widths, "mean_abs_corr",
                   "mean |activation correlation|",
                   "Pairwise neuron overlap vs group partition",
                   os.path.join(d, "groups_corr.png"), source="collapse")
    gtable = group_table(results, widths)
    if gtable:
        print(gtable)
        with open(os.path.join(d, "group_table.txt"), "w") as f:
            f.write(gtable + "\n")
        print(f"  saved {os.path.join(d, 'group_table.txt')}")


    # ---- penalty-axis plots: the point of a --penalty_sweep run -------------
    if args.penalty_sweep is not None:
        pen = args.penalty
        plot_vs_penalty(results, widths, weights, "acc", "test accuracy (%)",
                        f"Accuracy vs {pen} penalty weight",
                        os.path.join(d, "penalty_acc.png"))
        plot_vs_penalty(results, widths, weights, "recon_mse",
                        "reconstruction MSE",
                        f"Reconstruction MSE vs {pen} penalty weight",
                        os.path.join(d, "penalty_recon.png"))
        plot_vs_penalty(results, widths, weights, "eff_rank_act",
                        "effective rank of activations",
                        f"Effective rank vs {pen} penalty weight",
                        os.path.join(d, "penalty_eff_rank.png"),
                        source="collapse")
        plot_vs_penalty(results, widths, weights, "eff_rank_act_frac",
                        "effective rank / width",
                        f"Effective rank fraction vs {pen} penalty weight",
                        os.path.join(d, "penalty_eff_rank_frac.png"),
                        source="collapse")
        plot_vs_penalty(results, widths, weights, "mean_abs_cos_dec",
                        "mean |cosine| of per-neuron decoders",
                        f"Decoder overlap vs {pen} penalty weight",
                        os.path.join(d, "penalty_cos_dec.png"),
                        source="collapse")
        ptable = penalty_table(results, widths, weights)
        if ptable:
            print(ptable)
            with open(os.path.join(d, "penalty_table.txt"), "w") as f:
                f.write(ptable + "\n")
            print(f"  saved {os.path.join(d, 'penalty_table.txt')}")

    # ---- table ----
    table = summary_table(results, widths)
    print(table)
    with open(os.path.join(d, "summary_table.txt"), "w") as f:
        f.write(table + "\n")
    print(f"\n  saved {os.path.join(d, 'summary_table.txt')}")
    print(f"\ndone in {time.time() - t0:.1f}s. Outputs in {d}/")


if __name__ == "__main__":
    main()