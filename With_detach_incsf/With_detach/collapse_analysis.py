#!/usr/bin/env python3
"""
collapse_analysis.py — quantify and SHOW hidden-layer collapse.

sweep_hidden.py answers "how much collapse, as a function of width". This
script answers "what does that collapse actually look like", by training a
small number of models and then producing, for each one, both the full numeric
report from collapse_metrics.py and a set of images that make the numbers
legible.

Each diagnostic below is paired with the number it visualises, so the figure
and the metric can be read against each other:

  1. per-neuron reconstruction gallery  (NAN only)   -> mean_abs_cos_dec
        The single most direct picture of collapse. Each neuron in a NAN layer
        has its own decoder and so reconstructs the whole image by itself. The
        gallery shows one test image and then that image as reconstructed by
        each of the first N neurons independently. A healthy layer gives N
        visibly different reconstructions, each emphasising different parts of
        the garment. A collapsed layer gives N copies of the same blurry blob —
        which is what "the neurons have all learned the same thing" means, made
        visible.

  2. decoder template gallery           (NAN only)   -> mean_abs_cos_dec
        Row j of decoder_weight reshaped to res x res: the image template that
        neuron j is contributing. Sorted by activation variance so the most
        active neurons come first. Collapse = all templates look alike.

  3. activation correlation heatmap     (all)        -> mean_abs_corr
        The H x H matrix of pairwise neuron correlations, neurons ordered by a
        hierarchical-style greedy clustering so that blocks of mutually
        redundant neurons appear as bright squares on the diagonal. A healthy
        layer is dark off the diagonal; a collapsed layer is bright everywhere.

  4. singular value spectrum            (all)        -> eff_rank_act, pr90_act
        Normalised singular values of the centred activation matrix, log y,
        with the effective rank and the 90%-variance cutoff marked. Collapse is
        a spectrum that falls off a cliff after the first few components.

  5. activation statistics panel        (all)        -> frac_dead, frac_saturated
        Per-neuron mean and std over the eval set, sorted. Dead neurons appear
        as a flat run at std ~ 0; saturated neurons as means pinned at 0 or 1.
        The dead-neuron threshold is drawn on the plot.

  6. neuron activation histogram        (all)        -> frac_saturated
        Distribution of all activation values. A healthy sigmoid layer spreads
        across (0,1); a saturated layer piles up at the two ends.

  7. cross-variant comparison bars      (all)        -> every headline metric
        The same metric side by side for every variant trained, so "NAN
        collapses more than a plain MLP at this width" is a bar chart rather
        than a paragraph.

Dataset: Fashion-MNIST via models.get_loaders, with --res and --classes passed
through. Nothing here hardcodes 28x28 or 10 classes, so MNIST / EMNIST /
larger-resolution datasets need only a change in get_loaders.

THE LEARNING RULE (--detach_head)
By default cross-entropy is backpropagated through the head INTO the hidden
layer, so an AE variant's encoder is shaped by both objectives at once.
--detach_head cuts that path, giving Bull's rule: the hidden layer is built by
its reconstruction objective alone. Applied to AE variants only (nan / ann /
group* / dend*); linear / frozen / mlp are untouched. See simple_core's module
docstring, and keep the flag matched to the sweep_hidden run this analysis is
meant to illustrate — a detached gallery next to an attached sweep is not the
same experiment.

The galleries are where a detached run should be most visibly different: with
no task gradient shaping the encoder, the per-neuron reconstructions and
decoder templates show what the local objective builds on its own.

USAGE
    python collapse_analysis.py                                  # interleaved
    python collapse_analysis.py --schedule interleaved combined   # both
    python collapse_analysis.py --hidden 64 --variants mlp nan ann
    python collapse_analysis.py --hidden 256 --n_neurons_shown 16
    python collapse_analysis.py --detach_head --hidden 128 --variants mlp nan ann
    python collapse_analysis.py --epochs 2 --seeds 1 --variants nan  # smoke test

OUTPUTS (subdir collapse_analysis_<dataset>_h<HIDDEN>/, with _det appended
when --detach_head is set, under $NAN_OUT_DIR)
    collapse_report.json              full numeric report, per variant,
                                      mean + SEM over seeds, plus per-seed raw
    recon_gallery_<variant>.png       per-neuron reconstructions (NAN)
    decoders_<variant>.png            decoder templates (NAN)
    corr_heatmap_<variant>.png        clustered activation correlation matrix
    spectrum_<variant>.png            singular value spectrum
    act_stats_<variant>.png           per-neuron mean/std, dead-line marked
    act_hist_<variant>.png            activation value histogram
    compare_metrics.png               headline metrics across variants
    collapse_summary.txt             the numbers as plain text
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
                         is_ae_variant, apply_detach_head)
# build_model comes from group_ae, which handles the group* variants and
# defers every original variant to simple_core unchanged.
from group_ae import build_model, collect_group_stats
from collapse_metrics import (collapse_report, aggregate_reports, summarise,
                              collect_activations, _singular_values,
                              effective_rank, components_for_variance)
from inhibition import (register_penalties, wrap_inhibition,
                        collect_inhibition_stats, INHIBITIONS)

# Keeps PENALTIES in step with sweep_hidden.py even though this script does not
# expose --penalty itself.
register_penalties()

try:
    from models import DEVICE, outdir
except ImportError:
    from nan.models import DEVICE, outdir

# get_loaders comes from datasets_ext, which adds MNIST/EMNIST and defers to
# models.get_loaders for fashion so those results stay bit-identical.
from datasets_ext import get_loaders, default_classes, DATASETS


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ==============================================================================
# Training — same schedule as Week 5 / sweep_hidden
# ==============================================================================
def train_one(variant, tr, te, args, seed, schedule):
    """Train one model under one schedule. See sweep_hidden.train_one for the
    definition of the two schedules; they are kept identical between the two
    scripts so a width appearing in both is the same computation."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(variant, args.res * args.res, args.hidden,
                        args.classes, args.activation).to(DEVICE)

    # Bull's rule, applied to AE variants ONLY — identical gate to
    # sweep_hidden.py so a width appearing in both is still the same
    # computation under either setting of the flag.
    if getattr(args, "detach_head", False) and is_ae_variant(variant):
        apply_detach_head(model, True, variant)

    # Mutual inhibition, same gate and same ordering as sweep_hidden.py, so a
    # width analysed here is the same computation as that width in the sweep.
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
    for _epoch in range(args.epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            if schedule == "combined":
                loss = F.cross_entropy(model(x), y)
                if trains_ae and args.lam > 0:
                    loss = loss + args.lam * model.local_loss(x)
                loss.backward()
            else:
                if trains_ae and torch.rand(1).item() < args.ae_prob:
                    model.local_loss(x).backward()
                else:
                    F.cross_entropy(model(x), y).backward()
            optimizer.step()
    return model, evaluate(model, te)


def build_run_specs(variants, schedules):
    """(label, variant, schedule) triples. AE variants run once per schedule
    when more than one is requested; others are schedule-independent."""
    # is_ae_variant comes from simple_core, so this script and sweep_hidden.py
    # cannot disagree. It also covers dend* variants, which the local lambda
    # this replaced did not — a dend run with two schedules previously
    # collapsed into a single spec and silently dropped one of them.
    is_ae = is_ae_variant
    short = {"interleaved": "int", "combined": "comb"}
    specs = []
    for v in variants:
        if is_ae(v) and len(schedules) > 1:
            for s in schedules:
                specs.append((f"{v}-{short[s]}", v, s))
        else:
            specs.append((v, v, schedules[0]))
    return specs


# ==============================================================================
# Per-neuron reconstruction error, and the static / dynamic split
# ==============================================================================
@torch.no_grad()
def _nan_parts(model, loader, max_samples=2000):
    """For a NAN layer, return (code, w, b, X) as numpy/torch on CPU.

    Neuron j's reconstruction of image x is

        recon_j(x)  =  b_j  +  h_j(x) * w_j

    so the ONLY thing that varies with the input is the scalar h_j. Everything
    else is fixed. That structure is what the two functions below quantify.
    """
    if getattr(model, "kind", None) != "nan":
        return None
    model.eval()
    hidden = model.hidden
    X, _ = loader.dataset.tensors
    x = X[:max_samples].to(DEVICE)
    code = hidden(x)                                     # (n, H)
    return (code.cpu(), hidden.decoder_weight.detach().cpu(),
            hidden.decoder_bias.detach().cpu(), x.cpu())


@torch.no_grad()
def per_neuron_mse(model, loader, max_samples=2000):
    """Mean squared error of each neuron's own reconstruction. (H,) or None.

    This is reconstruction ACCURACY per neuron, which is a different ordering
    from activation variance -- the gallery can be sorted by either.
    """
    parts = _nan_parts(model, loader, max_samples)
    if parts is None:
        return None
    code, w, b, x = parts
    out = np.empty(w.shape[0], dtype=np.float64)
    for j in range(w.shape[0]):                          # loop keeps memory flat
        rec = code[:, j:j + 1] * w[j].unsqueeze(0) + b[j].unsqueeze(0)
        out[j] = torch.mean((rec - x) ** 2).item()
    return out


@torch.no_grad()
def ghost_report(model, loader, max_samples=2000):
    """How much of a neuron's output is fixed, and how much responds to input?

    Splits recon_j(x) = b_j + h_j(x) * w_j into

        static   the part that is identical for every input,
                 b_j + mean(h_j) * w_j   -- the "ghost" garment
        dynamic  the part that moves,  (h_j - mean(h_j)) * w_j

    and reports the share of total output energy carried by each. It also
    correlates each neuron's static image with the dataset mean image, which
    tests directly whether the ghost is simply the average garment.

    A layer whose neurons are mostly static is one where almost nothing the
    neuron emits depends on what it was shown.
    """
    parts = _nan_parts(model, loader, max_samples)
    if parts is None:
        return None
    code, w, b, x = parts
    mean_img = x.mean(0)
    mu = code.mean(0)                                    # (H,)
    sd = code.std(0)                                     # (H,)
    static = b + mu.unsqueeze(1) * w                     # (H, in) fixed image
    static_energy = (static ** 2).mean(dim=1)            # (H,)
    dynamic_energy = (sd ** 2) * (w ** 2).mean(dim=1)    # (H,)
    frac_dynamic = dynamic_energy / (static_energy + dynamic_energy + 1e-12)

    sc = static - static.mean(dim=1, keepdim=True)
    mc = mean_img - mean_img.mean()
    corr = ((sc * mc).mean(dim=1)
            / (sc.std(dim=1) * mc.std() + 1e-12)).clamp(-1, 1)

    return {"frac_dynamic_mean": float(frac_dynamic.mean()),
            "frac_dynamic_median": float(frac_dynamic.median()),
            "frac_static_mean": float(1 - frac_dynamic.mean()),
            "corr_static_with_mean_image": float(corr.mean()),
            "per_neuron_frac_dynamic": frac_dynamic.tolist()}


# ==============================================================================
# 1. Per-neuron reconstruction gallery (NAN only)
# ==============================================================================
@torch.no_grad()
def recon_gallery(model, loader, res, path, title, n_neurons=12, n_images=3,
                  rank="variance"):
    """Row per test image: the original, then that image as reconstructed by
    each of the first `n_neurons` neurons on its own.

    Only NAN has per-neuron decoders. For ANN the reconstruction is a single
    shared output and there is nothing per-neuron to show, so this is skipped
    rather than faked.
    """
    if getattr(model, "kind", None) != "nan":
        print(f"  [recon_gallery] skipped — per-neuron decoders exist for NAN only")
        return
    plt = _plt()
    X, _ = loader.dataset.tensors
    x = X[:n_images].to(DEVICE)

    hidden = model.hidden
    code = hidden(x)                                        # (n_img, H)
    w, b = hidden.decoder_weight, hidden.decoder_bias       # (H, in_dim)

    # Which neurons to show. These are DIFFERENT orderings and they answer
    # different questions:
    #   "variance" (default) -- highest activation std over the eval set, i.e.
    #       the neurons whose output responds most to the input. Not a measure
    #       of reconstruction quality; a neuron can vary a lot and reconstruct
    #       badly. Chosen as the default so the gallery is not dominated by
    #       units that barely move.
    #   "accuracy" -- lowest per-neuron reconstruction MSE, i.e. the neurons
    #       that genuinely reconstruct best. Use this to check whether the
    #       sameness in the gallery is an artefact of the ordering.
    if rank == "accuracy":
        mse = per_neuron_mse(model, loader, 1000)
        order = np.argsort(mse)[:n_neurons]
    else:
        acts = collect_activations(model, loader, 1000)
        order = np.argsort(-acts.std(axis=0))[:n_neurons]

    fig, axes = plt.subplots(n_images, n_neurons + 1,
                             figsize=(1.05 * (n_neurons + 1), 1.25 * n_images))
    axes = np.atleast_2d(axes)
    for i in range(n_images):
        ax = axes[i, 0]
        ax.imshow(x[i].cpu().reshape(res, res), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if i == 0:
            ax.set_title("input", fontsize=8)
        for j, nidx in enumerate(order):
            # neuron nidx's own reconstruction of image i, in isolation
            r = (code[i, nidx] * w[nidx] + b[nidx]).cpu().reshape(res, res)
            ax = axes[i, j + 1]
            ax.imshow(r.clamp(0, 1), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(f"n{nidx}", fontsize=7)
    fig.suptitle(title + "\n(each column is ONE neuron reconstructing the "
                         "image by itself — identical columns = collapse)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ==============================================================================
# 2. Decoder template gallery (NAN only)
# ==============================================================================
@torch.no_grad()
def decoder_gallery(model, loader, res, path, title, n_neurons=32):
    if getattr(model, "kind", None) != "nan":
        print(f"  [decoder_gallery] skipped — NAN only")
        return
    plt = _plt()
    w = model.hidden.decoder_weight.detach().cpu().numpy()
    acts = collect_activations(model, loader, 1000)
    order = np.argsort(-acts.std(axis=0))[:min(n_neurons, w.shape[0])]

    cols = 8
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.25 * cols, 1.4 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for k, nidx in enumerate(order):
        ax = axes.flat[k]
        t = w[nidx].reshape(res, res)
        ax.imshow(t, cmap="RdBu_r", vmin=-np.abs(t).max(), vmax=np.abs(t).max())
        ax.set_title(f"n{nidx}", fontsize=7)
        ax.axis("off")
    fig.suptitle(title + "\n(per-neuron decoder templates, sorted by activation "
                         "variance — similar templates = collapse)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ==============================================================================
# 3. Clustered correlation heatmap
# ==============================================================================
def _greedy_cluster_order(corr):
    """Order neurons so that mutually-correlated ones sit next to each other.

    A simple greedy nearest-neighbour chain on |corr|: start from the neuron
    with the highest total similarity, then repeatedly append the unvisited
    neuron most similar to the one just placed. This needs no scipy dependency
    and is enough to make redundant blocks visible as diagonal squares.
    """
    n = corr.shape[0]
    sim = np.abs(corr).copy()
    np.fill_diagonal(sim, 0.0)
    order = [int(np.argmax(sim.sum(axis=1)))]
    unvisited = set(range(n)) - set(order)
    while unvisited:
        last = order[-1]
        nxt = max(unvisited, key=lambda j: sim[last, j])
        order.append(nxt)
        unvisited.remove(nxt)
    return np.array(order)


def corr_heatmap(acts, path, title, report):
    plt = _plt()
    centred = acts - acts.mean(axis=0, keepdims=True)
    std = centred.std(axis=0)
    std_safe = np.where(std > 1e-8, std, 1.0)
    z = centred / std_safe
    corr = np.clip((z.T @ z) / acts.shape[0], -1, 1)
    corr[std <= 1e-8, :] = 0.0                    # dead neurons: no correlation
    corr[:, std <= 1e-8] = 0.0

    order = _greedy_cluster_order(corr)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(np.abs(corr[np.ix_(order, order)]), cmap="magma",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("neuron (clustered order)")
    ax.set_ylabel("neuron (clustered order)")
    ax.set_title(f"{title}\nmean |corr| = {report['mean_abs_corr']:.3f}   "
                 f"pairs>0.9 = {100 * report['frac_pairs_corr_90']:.1f}%",
                 fontsize=10)
    fig.colorbar(im, ax=ax, label="|correlation|")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  saved {path}")


# ==============================================================================
# 4. Singular value spectrum
# ==============================================================================
def spectrum_plot(acts, path, title):
    plt = _plt()
    centred = acts - acts.mean(axis=0, keepdims=True)
    sv = _singular_values(torch.from_numpy(centred))
    h = acts.shape[1]
    er = effective_rank(sv)
    pr90 = components_for_variance(sv, 0.90)
    norm = sv / sv.max() if sv.max() > 0 else sv

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.semilogy(np.arange(1, len(norm) + 1), np.maximum(norm, 1e-12),
                marker="o", markersize=3, linewidth=1.6)
    ax.axvline(er, color="crimson", linestyle="--",
               label=f"effective rank = {er:.1f} / {h}")
    ax.axvline(pr90, color="seagreen", linestyle=":",
               label=f"90% variance at {pr90} components")
    ax.set_xlabel("component index")
    ax.set_ylabel("singular value (normalised, log scale)")
    ax.set_title(title + "\n(a cliff after a few components = collapse)",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  saved {path}")


# ==============================================================================
# 5 & 6. Activation statistics and histogram
# ==============================================================================
def act_stats_plot(acts, path, title, report, dead_tol=1e-3):
    plt = _plt()
    mean = acts.mean(axis=0)
    std = acts.std(axis=0)
    order = np.argsort(-std)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax1.plot(std[order], linewidth=1.8)
    ax1.axhline(dead_tol, color="crimson", linestyle="--",
                label=f"dead threshold ({dead_tol})")
    ax1.set_yscale("log")
    ax1.set_xlabel("neuron (sorted by std)")
    ax1.set_ylabel("activation std over eval set (log)")
    ax1.set_title(f"dead = {100 * report['frac_dead']:.1f}%   "
                  f"saturated = {100 * report['frac_saturated']:.1f}%",
                  fontsize=10)
    ax1.grid(alpha=0.3, which="both"); ax1.legend()

    ax2.plot(mean[order], linewidth=1.8)
    ax2.axhline(0.0, color="grey", linewidth=0.8)
    ax2.axhline(1.0, color="grey", linewidth=0.8)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("neuron (same order)")
    ax2.set_ylabel("mean activation")
    ax2.set_title("means pinned at 0 or 1 = saturation", fontsize=10)
    ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  saved {path}")


def act_hist_plot(acts, path, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.hist(acts.ravel(), bins=80, color="steelblue", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("activation value")
    ax.set_ylabel("count (log)")
    ax.set_title(title + "\n(mass piled at 0 and 1 = saturated layer)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  saved {path}")


# ==============================================================================
# 7. Cross-variant comparison
# ==============================================================================
def compare_metrics(agg, path, hidden):
    plt = _plt()
    metrics = [
        ("eff_rank_act_frac", "effective rank / H", (0, 1)),
        ("collapse_index", "collapse index", (0, 1)),
        ("mean_abs_corr", "mean |activation corr|", (0, 1)),
        ("frac_pairs_corr_90", "frac pairs |corr| > 0.9", (0, 1)),
        ("frac_dead", "fraction dead", (0, 1)),
        ("mean_abs_cos_dec", "mean |cos| decoders (NAN)", (0, 1)),
    ]
    variants = list(agg.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, (key, label, ylim) in zip(axes.flat, metrics):
        names, vals, errs = [], [], []
        for v in variants:
            block = agg[v]
            if block is None or block.get(key) is None:
                continue
            names.append(v)
            vals.append(block[key]["mean"])
            errs.append(block[key]["sem"])
        if not names:
            ax.text(0.5, 0.5, f"{label}\nnot applicable", ha="center",
                    va="center", fontsize=9)
            ax.axis("off")
            continue
        ax.bar(names, vals, yerr=errs, capsize=4, color="steelblue")
        ax.set_ylim(*ylim)
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Collapse metrics across variants (hidden width = {hidden})\n"
                 "higher = more collapsed for every panel except "
                 "'effective rank / H'", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    print(f"  saved {path}")


# ==============================================================================
# Text summary
# ==============================================================================
def text_summary(agg, perf, hidden):
    lines = [f"Collapse analysis — hidden width {hidden}", ""]
    header = (f"{'run':13} {'acc%':>7} {'effrank':>9} {'rank/H':>8} "
              f"{'PR90':>7} {'|corr|':>8} {'p>0.9':>8} {'dead%':>7} "
              f"{'sat%':>7} {'|cosdec|':>9} {'collapse':>9}")
    lines += [header, "-" * len(header)]
    for v, block in agg.items():
        acc = perf[v]["acc"]
        if block is None:
            lines.append(f"{v:13} {acc:>7.2f}   (no hidden layer)")
            continue
        cd = block.get("mean_abs_cos_dec")
        cd_s = f"{cd['mean']:.3f}" if cd else "n/a"
        lines.append(
            f"{v:13} {acc:>7.2f} {block['eff_rank_act']['mean']:>9.2f} "
            f"{block['eff_rank_act_frac']['mean']:>8.3f} "
            f"{block['pr90_act']['mean']:>7.1f} "
            f"{block['mean_abs_corr']['mean']:>8.3f} "
            f"{block['frac_pairs_corr_90']['mean']:>8.3f} "
            f"{100 * block['frac_dead']['mean']:>7.1f} "
            f"{100 * block['frac_saturated']['mean']:>7.1f} "
            f"{cd_s:>9} {block['collapse_index']['mean']:>9.3f}")
    lines += ["", "Reading the table:",
              "  rank/H near 1  -> neurons independent, layer healthy",
              "  rank/H near 0  -> layer behaves as far fewer neurons than H",
              "  |corr| near 1  -> neurons are duplicates of one another",
              "  dead%          -> neurons with constant output (no signal)",
              "  |cosdec| near 1-> NAN neurons reconstruct the same template",
              "  collapse       -> 1 - rank/H, the single headline number"]
    return "\n".join(lines)


# ==============================================================================
# Main
# ==============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--variants", nargs="+",
                   default=["mlp", "nan", "ann"],
                   metavar="V",
                   help="linear frozen mlp nan ann frozen_nan frozen_ann, "
                        "and group<G>[-learn|-hebb] e.g. group8, group8-learn")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--res", type=int, default=28)
    p.add_argument("--dataset", default="fashion", choices=sorted(DATASETS),
                   help="fashion (default), mnist, emnist-balanced, "
                        "emnist-letters, emnist-digits, emnist-bymerge, "
                        "emnist-byclass")
    p.add_argument("--classes", type=int, default=None,
                   help="defaults to the dataset's own class count")
    p.add_argument("--activation", default="sigmoid", choices=["sigmoid", "relu"])
    p.add_argument("--ae_prob", type=float, default=0.5,
                   help="interleaved only: P(batch is a reconstruction update)")
    p.add_argument("--schedule", nargs="+", default=["interleaved"],
                   choices=["interleaved", "combined"])
    p.add_argument("--lam", type=float, default=2.0,
                   help="combined only: weight on the reconstruction term")
    p.add_argument("--detach_head", action="store_true",
                   help="Bull's rule: train AE hidden layers on reconstruction "
                        "ONLY, cutting the task gradient at the hidden/head "
                        "boundary. Applied to nan/ann/group*/dend* only; "
                        "linear/frozen/mlp are untouched. Keep this matched to "
                        "the sweep_hidden run being illustrated")
    p.add_argument("--inhibition", default="none", choices=list(INHIBITIONS),
                   help="mutual inhibition on the hidden layer: 'kwta' or "
                        "'lateral'. Keep matched to the sweep_hidden run being "
                        "illustrated")
    p.add_argument("--inhib_strength", type=float, default=1.0)
    p.add_argument("--inhib_k_frac", type=float, default=0.1)
    p.add_argument("--inhib_k_min", type=int, default=2)
    p.add_argument("--inhib_boost", type=float, default=1.0,
                   help="kwta duty-cycle boosting; 0 disables")
    p.add_argument("--inhib_steps", type=int, default=3)
    p.add_argument("--inhib_eta", type=float, default=0.01)
    p.add_argument("--inhib_theta_plus", type=float, default=0.05,
                   help="RAISE THIS if effective rank collapses toward 1")
    p.add_argument("--inhib_theta_decay", type=float, default=0.9)
    p.add_argument("--collapse_samples", type=int, default=2000)
    p.add_argument("--gallery_rank", default="variance",
                   choices=["variance", "accuracy"],
                   help="order gallery neurons by activation variance (default) "
                        "or by per-neuron reconstruction accuracy")
    p.add_argument("--n_neurons_shown", type=int, default=12,
                   help="neurons per row in the reconstruction gallery")
    p.add_argument("--n_images_shown", type=int, default=3,
                   help="test images in the reconstruction gallery")
    p.add_argument("--n_decoders_shown", type=int, default=32,
                   help="templates in the decoder gallery")
    p.add_argument("--dead_tol", type=float, default=1e-3,
                   help="activation std below which a neuron counts as dead")
    p.add_argument("--subdir", default=None)
    args = p.parse_args()
    if args.classes is None:
        args.classes = default_classes(args.dataset)

    schedules = list(dict.fromkeys(args.schedule))
    specs = build_run_specs(args.variants, schedules)
    # _det keeps a detached analysis from overwriting the attached figures it
    # is meant to be read against.
    sub = args.subdir or (f"collapse_analysis_{args.dataset}_h{args.hidden}"
                          + ("_det" if args.detach_head else "")
                          + ("" if args.inhibition == "none"
                             else f"_{args.inhibition}"))
    t0 = time.time()
    tr, te, in_dim = get_loaders(dataset=args.dataset, resolution=args.res,
                                 num_classes=args.classes,
                                 batch_size=args.batch_size)
    print(f"device={DEVICE} variants={args.variants} hidden={args.hidden} "
          f"seeds={args.seeds} epochs={args.epochs} res={args.res} "
          f"act={args.activation} ae_prob={args.ae_prob} "
          f"detach_head={args.detach_head} inhibition={args.inhibition}")
    if args.inhibition != "none":
        print("  [inhibition] READ RANK AND ACCURACY TOGETHER — the frozen "
              "layer had the best rank and the worst accuracy in Week 7.")
    if args.detach_head:
        print("  [detach_head] AE hidden layers trained by reconstruction "
              "ONLY; linear/frozen/mlp unchanged.")
        if len(schedules) > 1:
            print("  [detach_head] WARNING: both schedules requested, but with "
                  "disjoint parameter sets they should coincide.")
    print(f"train={len(tr.dataset)} test={len(te.dataset)}")

    d = outdir(sub)
    agg, raw, perf, seed0_models, ghosts = {}, {}, {}, {}, {}
    inhib_stats = {}

    for label, variant, schedule in specs:
        print(f"\n[{label}]")
        reports, accs = [], []
        for seed in range(args.seeds):
            model, metrics = train_one(variant, tr, te, args, seed, schedule)
            rep = collapse_report(model, te, args.collapse_samples,
                                  args.dead_tol)
            reports.append(rep)
            accs.append(metrics["acc"])
            if seed == 0:
                seed0_models[label] = model       # the model the figures use
            print(f"  seed {seed}: acc={metrics['acc']:.2f} | {summarise(rep)}")
        agg[label] = aggregate_reports(reports)
        raw[label] = reports
        ghosts[label] = ghost_report(seed0_models[label], te, args.collapse_samples)
        inhib_stats[label] = collect_inhibition_stats(seed0_models[label])
        perf[label] = {"acc": float(np.mean(accs)),
                       "acc_sem": float(np.std(accs) / np.sqrt(len(accs))),
                       "variant": variant, "schedule": schedule}

    # ---- figures, all from the seed-0 model so they are mutually consistent --
    print("\nfigures:")
    for label, model in seed0_models.items():
        acts = collect_activations(model, te, args.collapse_samples)
        if acts is None:
            print(f"  [{label}] no hidden layer — figures skipped")
            continue
        rep = raw[label][0]
        variant = label            # filenames use the label so schedules differ
        tag = f"{label} (H={args.hidden}, seed 0)"

        recon_gallery(model, te, args.res,
                      os.path.join(d, f"recon_gallery_{variant}.png"),
                      f"Per-neuron reconstructions — {tag}",
                      args.n_neurons_shown, args.n_images_shown)
        decoder_gallery(model, te, args.res,
                        os.path.join(d, f"decoders_{variant}.png"),
                        f"Decoder templates — {tag}", args.n_decoders_shown)
        corr_heatmap(acts, os.path.join(d, f"corr_heatmap_{variant}.png"),
                     f"Activation correlation — {tag}", rep)
        spectrum_plot(acts, os.path.join(d, f"spectrum_{variant}.png"),
                      f"Activation spectrum — {tag}")
        act_stats_plot(acts, os.path.join(d, f"act_stats_{variant}.png"),
                       f"Per-neuron activation statistics — {tag}", rep,
                       args.dead_tol)
        act_hist_plot(acts, os.path.join(d, f"act_hist_{variant}.png"),
                      f"Activation distribution — {tag}")

    compare_metrics(agg, os.path.join(d, "compare_metrics.png"), args.hidden)

    # ---- save ----
    cfg = config_record(
        args,
        schedule=(f"schedules={schedules}; interleaved uses ae_prob="
                  f"{args.ae_prob}, combined uses CE + {args.lam}*recon on "
                  f"every batch"
                  + ("; AE hidden layers DETACHED from the task gradient"
                     if args.detach_head else "")),
        extra={"experiment": "collapse analysis",
               "dataset": args.dataset,
               "variants": args.variants,
               "schedules": schedules,
               "ae_prob": args.ae_prob,
               "lam": args.lam,
               "detach_head": args.detach_head,
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
               "run_labels": {lbl: {"variant": v, "schedule": sc}
                              for lbl, v, sc in specs},
               "collapse_samples": args.collapse_samples,
               "dead_tol": args.dead_tol,
               "collapse_metric_definitions":
                   "see collapse_metrics.py module docstring"})
    save_json({"config": cfg, "performance": perf,
               "collapse_aggregate": agg, "collapse_per_seed": raw,
               "ghost_report": ghosts,
               "inhibition_stats": inhib_stats},
              "collapse_report.json", subdir=sub)

    summary = text_summary(agg, perf, args.hidden)
    print("\n" + summary)
    with open(os.path.join(d, "collapse_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\n  saved {os.path.join(d, 'collapse_summary.txt')}")
    print(f"\ndone in {time.time() - t0:.1f}s. Outputs in {d}/")


if __name__ == "__main__":
    main()