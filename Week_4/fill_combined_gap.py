#!/usr/bin/env python3
"""
fill_combined_gap.py — fill in the missing small-lambda points in the
combined-loss sweep, merge with existing results, and re-plot.

Runs only the NEW lambda values (0.01, 0.05, 0.1, 0.2, 0.3) then loads
the existing combined_sweep_static.json, merges, sorts by lambda, and
overwrites the JSON + re-generates the recon_mse, acc, and rank plots.

Usage:
    python fill_combined_gap.py                          # defaults: static, h=128
    python fill_combined_gap.py --arch spiking           # spiking variant
    python fill_combined_gap.py --hidden 20              # h=20
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

try:
    from models import DEVICE, HiddenLayerClassifier, get_loaders, outdir
    from experiments import multiseed, metrics, save_json
except ImportError:
    from nan.models import DEVICE, HiddenLayerClassifier, get_loaders, outdir
    from nan.experiments import multiseed, metrics, save_json


TRACKED = ("acc", "eff_rank", "recon_acc", "recon_mse")


def run_combined(model, train_loader, test_loader, probe_batch, epochs,
                 lam=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for _epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits, recon_loss = model(x, want_local=True)
            loss = F.cross_entropy(logits, y)
            if lam > 0 and recon_loss.requires_grad:
                loss = loss + lam * recon_loss
            loss.backward()
            optimizer.step()
        history.append(metrics(model, test_loader, probe_batch))
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--arch", default="static",
                        choices=["static", "spiking", "ann", "nn"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=9)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--res", type=int, default=14)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--new_lams", nargs="+", type=float,
                        default=[0.01, 0.05, 0.1, 0.2, 0.3],
                        help="lambda values to fill in")
    args = parser.parse_args()

    sub = "combined_sweep"
    tag = args.arch
    t0 = time.time()

    # --- Load existing results ---
    existing_path = os.path.join(outdir(sub), f"combined_sweep_{tag}.json")
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            existing = json.load(f)
        print(f"loaded existing results: lams={existing['lams']}")
    else:
        existing = {"lams": [], "optim": "adam", "lr": 0.001}
        for k in TRACKED:
            existing[k] = {"mean": [], "sem": []}
        print("no existing results found, starting fresh")

    # --- Figure out which lambdas are truly new ---
    already = set(existing["lams"])
    new_lams = sorted([l for l in args.new_lams if l not in already])
    if not new_lams:
        print("all requested lambdas already exist, nothing to do")
        return
    print(f"new lambdas to run: {new_lams}")

    # --- Run the new points ---
    tr, te, in_dim = get_loaders(resolution=args.res, num_classes=args.classes)
    xb = next(iter(te))[0].to(DEVICE)
    seeds = range(args.seeds)

    new_results = {k: {"mean": [], "sem": []} for k in TRACKED}
    for lam in new_lams:
        print(f"\nlambda={lam} ...")
        agg = multiseed(lambda s: run_combined(
            HiddenLayerClassifier(in_dim, args.hidden, args.classes,
                                  arch=args.arch).to(DEVICE),
            tr, te, xb, args.epochs, lam=lam),
            seeds, in_dim, args.hidden, args.classes)
        for k in TRACKED:
            new_results[k]["mean"].append(agg[k]["mean"][-1])
            new_results[k]["sem"].append(agg[k]["sem"][-1])

    # --- Merge: combine old + new, sort by lambda ---
    all_lams = existing["lams"] + new_lams
    merged = {k: {"mean": existing[k]["mean"] + new_results[k]["mean"],
                   "sem":  existing[k]["sem"]  + new_results[k]["sem"]}
              for k in TRACKED}

    order = np.argsort(all_lams)
    all_lams_sorted = [all_lams[i] for i in order]
    for k in TRACKED:
        merged[k]["mean"] = [merged[k]["mean"][i] for i in order]
        merged[k]["sem"]  = [merged[k]["sem"][i]  for i in order]

    # --- Save merged JSON ---
    out = {"lams": all_lams_sorted, "optim": "adam", "lr": 0.001, **merged}
    save_json(out, f"combined_sweep_{tag}.json", subdir=sub)

    # --- Re-plot all three final-value figures ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _plot(xs, series, xlabel, ylabel, fname, title, logx=True):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for name, s in series.items():
            ax.errorbar(xs, s["mean"], yerr=s["sem"], marker="o", capsize=3,
                        linewidth=2, label=name)
        if logx:
            ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        path = os.path.join(outdir(sub), fname)
        plt.savefig(path, dpi=130); plt.close(fig)
        print(f"  saved {path}")

    xs = all_lams_sorted
    _plot(xs, {"recon MSE": merged["recon_mse"]},
          "reconstruction mix weight (lambda)",
          "final reconstruction loss (MSE)",
          f"combined_recon_mse_{tag}.png",
          f"Combined loss: recon MSE vs lambda ({tag})")

    _plot(xs, {"test acc (%)": merged["acc"],
               "recon acc (%)": merged["recon_acc"]},
          "reconstruction mix weight (lambda)", "final accuracy (%)",
          f"combined_acc_{tag}.png",
          f"Combined loss CE + lambda*recon: accuracy tradeoff ({tag})")

    _plot(xs, {"effective rank": merged["eff_rank"]},
          "reconstruction mix weight (lambda)", "final effective rank",
          f"combined_rank_{tag}.png",
          f"Combined loss: neuron diversity vs lambda ({tag})")

    print(f"\n  {'lam':>6} {'acc%':>7} {'recon_mse':>10} {'recon_acc%':>11} {'rank':>7}")
    for i, lam in enumerate(xs):
        print(f"  {lam:>6} {merged['acc']['mean'][i]:>7.2f} "
              f"{merged['recon_mse']['mean'][i]:>10.4f} "
              f"{merged['recon_acc']['mean'][i]:>11.2f} "
              f"{merged['eff_rank']['mean'][i]:>7.1f}")

    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
