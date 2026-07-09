#!/usr/bin/env python3
"""
extra_sweeps.py — follow-up experiments from the sweep analysis (v2).

v2 changes:
  * recon_mse is now PLOTTED in every experiment, alongside recon_acc.
    recon_acc is a derived, clamped estimate (1 - mse/var); recon_mse is the
    raw quantity the autoencoders actually optimise, so both views are saved.
  * --optim {adam,sgd} and --lr flags (defaults adam/1e-3, the harness
    setting; kept for flexibility, not part of the current presentation).
  * combined-loss lambda grid centred on the optimum region found in the
    earlier W2-3 recon-scaler sweep (~13-20), spanning up to 100.
  * output filenames are tagged with the optimiser/lr when non-default, so
    adam and sgd runs never overwrite each other.

Experiments:
  penalty_recon : div_weight sweep, BOTH penalties overlaid, vs final
                  recon_acc / recon_mse / test acc / effective rank.
  ae_prob       : AE-cycle probability sweep (the real schedule-pressure knob).
  combined      : CE + lambda*recon on every batch (needs --confirm_combined).

Usage:
    python extra_sweeps.py --mode penalty_recon --arch static
    python extra_sweeps.py --mode ae_prob --arch static
    python extra_sweeps.py --mode combined --confirm_combined
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Same robust import pattern as run_experiments.py
try:
    from .models import DEVICE, HiddenLayerClassifier, get_loaders, PENALTIES, outdir
    from .experiments import multiseed, metrics, save_json, plot_curves
except ImportError:
    try:
        from nan.models import DEVICE, HiddenLayerClassifier, get_loaders, PENALTIES, outdir
        from nan.experiments import multiseed, metrics, save_json, plot_curves
    except ImportError:
        from models import DEVICE, HiddenLayerClassifier, get_loaders, PENALTIES, outdir
        from experiments import multiseed, metrics, save_json, plot_curves


DEFAULT_OPTIM, DEFAULT_LR = "adam", 1e-3


def make_optimizer(params, optim_name=DEFAULT_OPTIM, lr=DEFAULT_LR):
    """adam (harness default) or sgd."""
    if optim_name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    return torch.optim.Adam(params, lr=lr)


def run_tag(arch, optim_name, lr):
    """Filename tag: plain arch for the default optimiser, otherwise annotated
    (e.g. static_sgd0.1) so different optimiser runs coexist in one folder."""
    if optim_name == DEFAULT_OPTIM and abs(lr - DEFAULT_LR) < 1e-12:
        return arch
    return f"{arch}_{optim_name}{lr:g}"


# ==============================================================================
# 1. Training loops (local variants — the originals stay untouched)
# ==============================================================================
def run_interleaved_prob(model, train_loader, test_loader, probe_batch, epochs,
                         ae_prob=0.5, local_weight=1.0, div_weight=0.0,
                         penalty="decoder", optim_name=DEFAULT_OPTIM,
                         lr=DEFAULT_LR):
    """run_interleaved, but the AE-cycle probability, optimiser, and lr are
    parameters. ae_prob=0.5 + adam + lr=1e-3 reproduces the original exactly."""
    optimizer = make_optimizer(model.parameters(), optim_name, lr)
    penalty_fn = PENALTIES[penalty]
    history = []
    for _epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            run_task_cycle = (local_weight == 0 or ae_prob == 0
                              or torch.rand(1).item() >= ae_prob)
            if run_task_cycle:
                optimizer.zero_grad()
                F.cross_entropy(model(x), y).backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                _, recon_loss = model(x, want_local=True)
                loss = local_weight * recon_loss
                if div_weight > 0:
                    loss = loss + div_weight * penalty_fn(model.hidden, x)
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                else:                       # plain NN: no AE, fall back to task
                    F.cross_entropy(model(x), y).backward()
                    optimizer.step()
        history.append(metrics(model, test_loader, probe_batch))
    return history


def run_combined(model, train_loader, test_loader, probe_batch, epochs,
                 lam=1.0, div_weight=0.0, penalty="decoder",
                 optim_name=DEFAULT_OPTIM, lr=DEFAULT_LR):
    """Single combined loss on EVERY batch:  CE + lam * recon (+ dw * penalty).

    (Same form as the earlier W2-3 recon-scaler sweep: under SGD, lam*lr is
    the effective lr on the decoder path; under Adam, lam mainly sets the mix
    of task vs recon gradients on the shared encoder.)
    """
    optimizer = make_optimizer(model.parameters(), optim_name, lr)
    penalty_fn = PENALTIES[penalty]
    history = []
    for _epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits, recon_loss = model(x, want_local=True)
            loss = F.cross_entropy(logits, y)
            if lam > 0 and recon_loss.requires_grad:   # NN has no AE: CE only
                loss = loss + lam * recon_loss
                if div_weight > 0:
                    loss = loss + div_weight * penalty_fn(model.hidden, x)
            loss.backward()
            optimizer.step()
        history.append(metrics(model, test_loader, probe_batch))
    return history


# ==============================================================================
# 2. Plotting helpers
# ==============================================================================
def _plot_final_overlay(xs, series, xlabel, ylabel, fname, title,
                        subdir=None, logx=False):
    """series: {line_label: {"mean": [...], "sem": [...]}} over xs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, s in series.items():
        ax.errorbar(xs, s["mean"], yerr=s["sem"], marker="o", capsize=3,
                    linewidth=2, label=name)
    if logx:
        ax.set_xscale("symlog", linthresh=0.1)   # 0 stays plottable
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir(subdir), fname)
    plt.savefig(path, dpi=130); plt.close(fig)
    print(f"  saved {path}")


def _final(agg, key):
    """(mean, sem) of a metric at the final epoch of a multiseed aggregate."""
    return agg[key]["mean"][-1], agg[key]["sem"][-1]


def _collect(agg, store, keys):
    for k in keys:
        mean, sem = _final(agg, k)
        store[k]["mean"].append(mean)
        store[k]["sem"].append(sem)


TRACKED = ("acc", "eff_rank", "recon_acc", "recon_mse")


def _new_store():
    return {k: {"mean": [], "sem": []} for k in TRACKED}


# ==============================================================================
# 3. Experiment: penalty strength vs reconstruction (both penalties overlaid)
# ==============================================================================
def exp_penalty_recon_sweep(seeds, epochs, res=14, nc=10, hidden=128,
                            arch="static", optim_name=DEFAULT_OPTIM,
                            lr=DEFAULT_LR,
                            values=(0, 0.1, 0.5, 1, 2, 5, 10, 20)):
    values = list(values)
    tr, te, in_dim = get_loaders(resolution=res, num_classes=nc)
    xb = next(iter(te))[0].to(DEVICE)
    sub = "penalty_recon_sweep"
    tag = run_tag(arch, optim_name, lr)

    # results[metric][penalty] = {"mean": [...], "sem": [...]} over values
    results = {m: {} for m in TRACKED}
    all_epoch_curves = {}                 # full per-epoch aggregates, per penalty
    for pen in ("decoder", "activation"):
        store = _new_store()
        epoch_curves = {}
        for v in values:
            print(f"exp_penalty_recon_sweep [{tag}/{pen}]: div_weight={v} ...")
            agg = multiseed(lambda s: run_interleaved_prob(
                HiddenLayerClassifier(in_dim, hidden, nc, arch=arch).to(DEVICE),
                tr, te, xb, epochs, ae_prob=0.5, local_weight=1.0,
                div_weight=v, penalty=pen, optim_name=optim_name, lr=lr),
                seeds, in_dim, hidden, nc)
            epoch_curves[f"dw={v}"] = agg
            _collect(agg, store, TRACKED)
        for m in TRACKED:
            results[m][pen] = store[m]
        all_epoch_curves[pen] = epoch_curves
        # per-epoch LOSS curves (one line per div_weight), SEM-shaded
        plot_curves(epoch_curves, "recon_mse", "reconstruction loss (MSE)",
                    f"recon_mse_curves_{tag}_{pen}.png",
                    f"Reconstruction loss over training — {pen} penalty ({tag})",
                    subdir=sub)
        plot_curves(epoch_curves, "acc", "test accuracy (%)",
                    f"acc_curves_{tag}_{pen}.png",
                    f"Test accuracy over training — {pen} penalty ({tag})",
                    subdir=sub)
    save_json(all_epoch_curves, f"penalty_recon_curves_{tag}.json", subdir=sub)

    save_json({"values": values, "optim": optim_name, "lr": lr, **results},
              f"penalty_recon_sweep_{tag}.json", subdir=sub)

    figs = [("recon_acc", "final reconstruction accuracy (%)", "recon_vs_penalty"),
            ("recon_mse", "final reconstruction loss (MSE)", "recon_mse_vs_penalty"),
            ("acc", "final test accuracy (%)", "acc_vs_penalty"),
            ("eff_rank", "final effective rank", "rank_vs_penalty")]
    for key, ylabel, stem in figs:
        _plot_final_overlay(values, results[key],
                            "penalty strength (div_weight)", ylabel,
                            f"{stem}_{tag}.png",
                            f"{ylabel} vs penalty strength ({tag}, interleaved)",
                            subdir=sub, logx=True)

    print(f"\n  penalty vs reconstruction ({tag}, final epoch):")
    print(f"  {'dw':>6} {'penalty':11} {'recon_mse':>10} {'recon_acc%':>11} "
          f"{'acc%':>7} {'rank':>7}")
    for pen in ("decoder", "activation"):
        for i, v in enumerate(values):
            print(f"  {v:>6} {pen:11} {results['recon_mse'][pen]['mean'][i]:>10.4f} "
                  f"{results['recon_acc'][pen]['mean'][i]:>11.2f} "
                  f"{results['acc'][pen]['mean'][i]:>7.2f} "
                  f"{results['eff_rank'][pen]['mean'][i]:>7.1f}")
    return results


# ==============================================================================
# 4. Experiment: AE-cycle probability sweep (the real schedule-pressure knob)
# ==============================================================================
def exp_ae_prob_sweep(seeds, epochs, res=14, nc=10, hidden=128, arch="static",
                      optim_name=DEFAULT_OPTIM, lr=DEFAULT_LR,
                      probs=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9)):
    probs = list(probs)
    tr, te, in_dim = get_loaders(resolution=res, num_classes=nc)
    xb = next(iter(te))[0].to(DEVICE)
    sub = "ae_prob_sweep"
    tag = run_tag(arch, optim_name, lr)

    store = _new_store()
    epoch_curves = {}
    for p in probs:
        print(f"exp_ae_prob_sweep [{tag}]: ae_prob={p} ...")
        agg = multiseed(lambda s: run_interleaved_prob(
            HiddenLayerClassifier(in_dim, hidden, nc, arch=arch).to(DEVICE),
            tr, te, xb, epochs, ae_prob=p, local_weight=1.0,
            optim_name=optim_name, lr=lr),
            seeds, in_dim, hidden, nc)
        epoch_curves[f"p={p}"] = agg
        _collect(agg, store, TRACKED)
    plot_curves(epoch_curves, "recon_mse", "reconstruction loss (MSE)",
                f"ae_prob_recon_mse_curves_{tag}.png",
                f"Reconstruction loss over training per AE-cycle probability ({tag})",
                subdir=sub)
    plot_curves(epoch_curves, "acc", "test accuracy (%)",
                f"ae_prob_acc_curves_{tag}.png",
                f"Test accuracy over training per AE-cycle probability ({tag})",
                subdir=sub)
    save_json(epoch_curves, f"ae_prob_curves_{tag}.json", subdir=sub)

    save_json({"probs": probs, "optim": optim_name, "lr": lr, **store},
              f"ae_prob_sweep_{tag}.json", subdir=sub)

    _plot_final_overlay(probs,
                        {"test acc (%)": store["acc"],
                         "recon acc (%)": store["recon_acc"]},
                        "P(batch is an AE cycle)", "final accuracy (%)",
                        f"ae_prob_acc_{tag}.png",
                        f"Task vs reconstruction accuracy across schedule share ({tag})",
                        subdir=sub)
    _plot_final_overlay(probs, {"recon MSE": store["recon_mse"]},
                        "P(batch is an AE cycle)", "final reconstruction loss (MSE)",
                        f"ae_prob_recon_mse_{tag}.png",
                        f"Reconstruction loss across schedule share ({tag})",
                        subdir=sub)
    _plot_final_overlay(probs, {"effective rank": store["eff_rank"]},
                        "P(batch is an AE cycle)", "final effective rank",
                        f"ae_prob_rank_{tag}.png",
                        f"Neuron diversity across schedule share ({tag})",
                        subdir=sub)

    print(f"\n  AE-probability sweep ({tag}, final epoch):")
    print(f"  {'p':>5} {'acc%':>7} {'recon_mse':>10} {'recon_acc%':>11} {'rank':>7}")
    for i, p in enumerate(probs):
        print(f"  {p:>5} {store['acc']['mean'][i]:>7.2f} "
              f"{store['recon_mse']['mean'][i]:>10.4f} "
              f"{store['recon_acc']['mean'][i]:>11.2f} "
              f"{store['eff_rank']['mean'][i]:>7.1f}")
    return store


# ==============================================================================
# 5. Experiment: combined-loss lambda sweep (single loss CE + lambda*recon)
# ==============================================================================
def exp_combined_lambda_sweep(seeds, epochs, res=14, nc=10, hidden=128,
                              arch="static", optim_name=DEFAULT_OPTIM,
                              lr=DEFAULT_LR,
                              lams=(0, 0.5, 1, 2, 5, 13, 20, 50, 100)):
    """Lambda grid centred on the optimum region seen in the earlier W2-3
    recon-scaler sweep (~13-20), spanning 0..100."""
    lams = list(lams)
    tr, te, in_dim = get_loaders(resolution=res, num_classes=nc)
    xb = next(iter(te))[0].to(DEVICE)
    sub = "combined_sweep"
    tag = run_tag(arch, optim_name, lr)

    store = _new_store()
    epoch_curves = {}
    for lam in lams:
        print(f"exp_combined_lambda_sweep [{tag}]: lambda={lam} ...")
        agg = multiseed(lambda s: run_combined(
            HiddenLayerClassifier(in_dim, hidden, nc, arch=arch).to(DEVICE),
            tr, te, xb, epochs, lam=lam, optim_name=optim_name, lr=lr),
            seeds, in_dim, hidden, nc)
        epoch_curves[f"lam={lam}"] = agg
        _collect(agg, store, TRACKED)
    plot_curves(epoch_curves, "recon_mse", "reconstruction loss (MSE)",
                f"combined_recon_mse_curves_{tag}.png",
                f"Reconstruction loss over training per lambda ({tag})",
                subdir=sub)
    plot_curves(epoch_curves, "acc", "test accuracy (%)",
                f"combined_acc_curves_{tag}.png",
                f"Test accuracy over training per lambda ({tag})",
                subdir=sub)
    save_json(epoch_curves, f"combined_curves_{tag}.json", subdir=sub)

    save_json({"lams": lams, "optim": optim_name, "lr": lr, **store},
              f"combined_sweep_{tag}.json", subdir=sub)

    _plot_final_overlay(lams, {"recon MSE": store["recon_mse"]},
                        "reconstruction mix weight (lambda)",
                        "final reconstruction loss (MSE)",
                        f"combined_recon_mse_{tag}.png",
                        f"Combined loss: recon MSE vs lambda ({tag})",
                        subdir=sub, logx=True)
    _plot_final_overlay(lams, {"test acc (%)": store["acc"],
                               "recon acc (%)": store["recon_acc"]},
                        "reconstruction mix weight (lambda)", "final accuracy (%)",
                        f"combined_acc_{tag}.png",
                        f"Combined loss CE + lambda*recon: accuracy tradeoff ({tag})",
                        subdir=sub, logx=True)
    _plot_final_overlay(lams, {"effective rank": store["eff_rank"]},
                        "reconstruction mix weight (lambda)", "final effective rank",
                        f"combined_rank_{tag}.png",
                        f"Combined loss: neuron diversity vs lambda ({tag})",
                        subdir=sub, logx=True)

    print(f"\n  combined-loss lambda sweep ({tag}, final epoch):")
    print(f"  {'lam':>6} {'acc%':>7} {'recon_mse':>10} {'recon_acc%':>11} {'rank':>7}")
    for i, lam in enumerate(lams):
        print(f"  {lam:>6} {store['acc']['mean'][i]:>7.2f} "
              f"{store['recon_mse']['mean'][i]:>10.4f} "
              f"{store['recon_acc']['mean'][i]:>11.2f} "
              f"{store['eff_rank']['mean'][i]:>7.1f}")
    return store


# ==============================================================================
# 6. CLI
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Follow-up sweeps: penalty-vs-recon, AE-cycle probability, "
                    "and (opt-in) combined-loss lambda sweep.")
    parser.add_argument("--mode", default="penalty_recon",
                        choices=["penalty_recon", "ae_prob", "combined", "all"])
    parser.add_argument("--arch", default="static",
                        choices=["static", "spiking", "ann", "nn"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=9,
                        help="epochs per run (sweep runs; ~60%% of a full run)")
    parser.add_argument("--res", type=int, default=14)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--optim", default=DEFAULT_OPTIM, choices=["adam", "sgd"])
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--confirm_combined", action="store_true",
                        help="required to run the combined-loss sweep")
    args = parser.parse_args()

    seeds = range(args.seeds)
    t0 = time.time()
    print(f"device={DEVICE} mode={args.mode} arch={args.arch} optim={args.optim} "
          f"lr={args.lr} seeds={args.seeds} epochs={args.epochs}")

    common = dict(res=args.res, nc=args.classes, hidden=args.hidden,
                  arch=args.arch, optim_name=args.optim, lr=args.lr)

    if args.mode in ("penalty_recon", "all"):
        exp_penalty_recon_sweep(seeds, args.epochs, **common)
    if args.mode in ("ae_prob", "all"):
        exp_ae_prob_sweep(seeds, args.epochs, **common)
    if args.mode in ("combined", "all"):
        if args.confirm_combined:
            exp_combined_lambda_sweep(seeds, args.epochs, **common)
        else:
            print("combined-loss sweep skipped: pass --confirm_combined to run it "
                  "(see the docstring of run_combined for what it does).")

    print(f"\ndone in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
