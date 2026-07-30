#!/usr/bin/env python3
"""
simple_combined.py — the same foundation, combined-loss schedule.

Identical to simple_interleaved.py in every respect (models, metrics, data,
config recording, grids) EXCEPT the training rule for the AE variants:
instead of a coin flip between task and reconstruction updates, every batch
optimises one loss:

    loss = cross_entropy + lambda * reconstruction_MSE

linear / frozen / mlp ignore lambda (no autoencoder) and train identically
to the interleaved script, so those baselines are directly comparable
across the two scripts.

Outputs (subdir simple_combined/): same set of files as the interleaved
script — results_combined.json, four metric-curve PNGs, and the 3x3
classification / reconstruction grids per variant.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from simple_core import (build_model, evaluate, aggregate, config_record,
                         save_json, classification_grid, reconstruction_grid,
                         plot_metric, print_final_table)
try:
    from models import DEVICE, get_loaders, outdir
except ImportError:
    from nan.models import DEVICE, get_loaders, outdir


def train_one(variant, tr, te, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(variant, args.res * args.res, args.hidden,
                        args.classes, args.activation).to(DEVICE)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    history = []
    for _epoch in range(args.epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            if model.has_ae and args.lam > 0 and not getattr(model, 'frozen', False):
                loss = loss + args.lam * model.local_loss(x)
            loss.backward()
            optimizer.step()
        history.append(evaluate(model, te))
    return history, model


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--variants", nargs="+",
                        default=["mlp", "nan"],
                        choices=["linear", "frozen", "mlp", "nan", "ann",
                                 "frozen_nan", "frozen_ann"])
    parser.add_argument("--lam", type=float, default=2.0,
                        help="weight on the reconstruction term (AE variants)")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--res", type=int, default=28,
                        help="28 = native Fashion-MNIST resolution")
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--activation", default="sigmoid",
                        choices=["sigmoid", "relu"])
    args = parser.parse_args()

    sub = "simple_combined"
    t0 = time.time()
    tr, te, in_dim = get_loaders(resolution=args.res, num_classes=args.classes,
                                 batch_size=args.batch_size)
    n_train, n_test = len(tr.dataset), len(te.dataset)
    print(f"device={DEVICE} variants={args.variants} lam={args.lam} "
          f"seeds={args.seeds} epochs={args.epochs} batch={args.batch_size} "
          f"lr={args.lr} res={args.res} hidden={args.hidden} "
          f"act={args.activation} | train={n_train} test={n_test}")

    curves, final_models = {}, {}
    for variant in args.variants:
        print(f"\n[{variant}]")
        runs = []
        for seed in range(args.seeds):
            hist, model = train_one(variant, tr, te, args, seed)
            runs.append(hist)
            if seed == 0:
                final_models[variant] = model
            m = hist[-1]
            rm = f"{m['recon_mse']:.4f}" if m["recon_mse"] is not None else "n/a"
            print(f"  seed {seed}: acc={m['acc']:.2f} top2={m['top2_acc']:.2f} "
                  f"ce={m['ce_loss']:.4f} recon={rm}")
        curves[variant] = aggregate(runs)

    cfg = config_record(args, schedule="combined: CE + lambda*recon on every "
                        "batch for nan/ann; pure task for linear/frozen/mlp",
                        extra={"lambda": args.lam, "train_size": n_train,
                               "test_size": n_test,
                               "variants": args.variants})
    save_json({"config": cfg, "curves": curves},
              "results_combined.json", subdir=sub)

    d = outdir(sub)
    plot_metric(curves, "acc", "test accuracy (%)",
                os.path.join(d, "acc_curves.png"),
                f"Test accuracy over training (lambda={args.lam:g})")
    plot_metric(curves, "top2_acc", "test top-2 accuracy (%)",
                os.path.join(d, "top2_curves.png"),
                f"Top-2 accuracy over training (lambda={args.lam:g})")
    plot_metric(curves, "ce_loss", "test cross-entropy loss",
                os.path.join(d, "ce_curves.png"),
                f"Cross-entropy loss over training (lambda={args.lam:g})")
    plot_metric(curves, "recon_mse", "reconstruction MSE",
                os.path.join(d, "recon_curves.png"),
                f"Reconstruction MSE over training (lambda={args.lam:g})")

    for variant, model in final_models.items():
        classification_grid(
            model, te, args.res,
            os.path.join(d, f"cls_grid_{variant}.png"),
            f"Classifications — {variant}, combined lam={args.lam:g} "
            f"(true label; top-2 with confidence)")
        reconstruction_grid(
            model, te, args.res,
            os.path.join(d, f"recon_grid_{variant}.png"),
            f"Reconstructions — {variant}, combined lam={args.lam:g}")

    print_final_table(curves)
    print(f"\ndone in {time.time() - t0:.1f}s. Outputs in {d}/")


if __name__ == "__main__":
    main()
