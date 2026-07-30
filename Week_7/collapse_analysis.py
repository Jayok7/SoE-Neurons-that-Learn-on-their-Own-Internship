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

USAGE
    python collapse_analysis.py                                  # defaults
    python collapse_analysis.py --hidden 64 --variants mlp nan ann
    python collapse_analysis.py --hidden 256 --n_neurons_shown 16
    python collapse_analysis.py --epochs 2 --seeds 1 --variants nan  # smoke test

OUTPUTS (subdir collapse_analysis_h<HIDDEN>/ under $NAN_OUT_DIR)
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
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from simple_core import build_model, evaluate, config_record, save_json
from collapse_metrics import (collapse_report, aggregate_reports, summarise,
                              collect_activations, _singular_values,
                              effective_rank, components_for_variance)

try:
    from models import DEVICE, get_loaders, outdir
except ImportError:
    from nan.models import DEVICE, get_loaders, outdir


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ==============================================================================
# Training — same schedule as Week 5 / sweep_hidden
# ==============================================================================
def train_one(variant, tr, te, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(variant, args.res * args.res, args.hidden,
                        args.classes, args.activation).to(DEVICE)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
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
    return model, evaluate(model, te)


# ==============================================================================
# 1. Per-neuron reconstruction gallery (NAN only)
# ==============================================================================
@torch.no_grad()
def recon_gallery(model, loader, res, path, title, n_neurons=12, n_images=3):
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

    # order neurons by activation variance over the eval set: the neurons that
    # are actually doing something come first, so the gallery is not dominated
    # by dead units.
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
    header = (f"{'variant':11} {'acc%':>7} {'effrank':>9} {'rank/H':>8} "
              f"{'PR90':>7} {'|corr|':>8} {'p>0.9':>8} {'dead%':>7} "
              f"{'sat%':>7} {'|cosdec|':>9} {'collapse':>9}")
    lines += [header, "-" * len(header)]
    for v, block in agg.items():
        acc = perf[v]["acc"]
        if block is None:
            lines.append(f"{v:11} {acc:>7.2f}   (no hidden layer)")
            continue
        cd = block.get("mean_abs_cos_dec")
        cd_s = f"{cd['mean']:.3f}" if cd else "n/a"
        lines.append(
            f"{v:11} {acc:>7.2f} {block['eff_rank_act']['mean']:>9.2f} "
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
                   choices=["linear", "frozen", "mlp", "nan", "ann",
                            "frozen_nan", "frozen_ann"])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--res", type=int, default=28)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--activation", default="sigmoid", choices=["sigmoid", "relu"])
    p.add_argument("--ae_prob", type=float, default=0.5)
    p.add_argument("--collapse_samples", type=int, default=2000)
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

    sub = args.subdir or f"collapse_analysis_h{args.hidden}"
    t0 = time.time()
    tr, te, in_dim = get_loaders(resolution=args.res, num_classes=args.classes,
                                 batch_size=args.batch_size)
    print(f"device={DEVICE} variants={args.variants} hidden={args.hidden} "
          f"seeds={args.seeds} epochs={args.epochs} res={args.res} "
          f"act={args.activation} ae_prob={args.ae_prob}")
    print(f"train={len(tr.dataset)} test={len(te.dataset)}")

    d = outdir(sub)
    agg, raw, perf, seed0_models = {}, {}, {}, {}

    for variant in args.variants:
        print(f"\n[{variant}]")
        reports, accs = [], []
        for seed in range(args.seeds):
            model, metrics = train_one(variant, tr, te, args, seed)
            rep = collapse_report(model, te, args.collapse_samples,
                                  args.dead_tol)
            reports.append(rep)
            accs.append(metrics["acc"])
            if seed == 0:
                seed0_models[variant] = model     # the model the figures use
            print(f"  seed {seed}: acc={metrics['acc']:.2f} | {summarise(rep)}")
        agg[variant] = aggregate_reports(reports)
        raw[variant] = reports
        perf[variant] = {"acc": float(np.mean(accs)),
                         "acc_sem": float(np.std(accs) / np.sqrt(len(accs)))}

    # ---- figures, all from the seed-0 model so they are mutually consistent --
    print("\nfigures:")
    for variant, model in seed0_models.items():
        acts = collect_activations(model, te, args.collapse_samples)
        if acts is None:
            print(f"  [{variant}] no hidden layer — figures skipped")
            continue
        rep = raw[variant][0]
        tag = f"{variant} (H={args.hidden}, seed 0)"

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
        schedule=(f"interleaved (ae_prob={args.ae_prob}) for nan/ann; "
                  f"pure task for linear/frozen/mlp"),
        extra={"experiment": "collapse analysis",
               "variants": args.variants,
               "ae_prob": args.ae_prob,
               "collapse_samples": args.collapse_samples,
               "dead_tol": args.dead_tol,
               "collapse_metric_definitions":
                   "see collapse_metrics.py module docstring"})
    save_json({"config": cfg, "performance": perf,
               "collapse_aggregate": agg, "collapse_per_seed": raw},
              "collapse_report.json", subdir=sub)

    summary = text_summary(agg, perf, args.hidden)
    print("\n" + summary)
    with open(os.path.join(d, "collapse_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\n  saved {os.path.join(d, 'collapse_summary.txt')}")
    print(f"\ndone in {time.time() - t0:.1f}s. Outputs in {d}/")


if __name__ == "__main__":
    main()
