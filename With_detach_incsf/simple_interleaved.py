#!/usr/bin/env python3
"""
simple_interleaved.py — the hidden-layer test, minimal and rigorous.

Question: is the hidden layer contributing, or acting as a reservoir?
    linear : hidden layer OFF (input -> head). Floor.
    frozen : hidden layer at random init, never trained (the reservoir).
    mlp    : hidden layer trained by the task (the contributor).
    nan/ann: autoencoder hidden layers, trained INTERLEAVED at ae_prob
             (default 0.5): each batch is a task update or a reconstruction
             update by an independent coin flip. No penalties, no extras.

Reading the comparison:
    mlp ≈ frozen        -> hidden training adds nothing (reservoir behaviour)
    frozen ≈ linear     -> even the random projection adds nothing
    mlp > frozen > linear -> the hidden layer genuinely contributes

Metrics recorded (test set, every epoch): accuracy, top-2 accuracy,
cross-entropy loss, reconstruction MSE (AE variants only).

--detach_head implements Bull's learning rule: the hidden layer of an AE
variant is shaped by its reconstruction objective ALONE, with the task
gradient stopping at the hidden/head boundary. Applied to nan/ann only;
linear/frozen/mlp keep their usual training, because the task-trained MLP is
the baseline a detached NAN has to beat. See simple_core's module docstring.

NOTE when detached: CE reaches only head parameters and reconstruction only
hidden parameters, so the two objectives have disjoint parameter sets. This
script and simple_combined.py should then agree to within seed noise — a free
correctness check that no task gradient is still leaking into the encoder. If
they disagree, fix that before running anything downstream.

Outputs (subdir simple_interleaved/, or simple_interleaved_det/ when
--detach_head is set, under $NAN_OUT_DIR):
    results_interleaved.json          config + per-epoch mean/SEM per variant
    acc / top2 / ce / recon PNGs      one overlay curve per metric
    cls_grid_<variant>.png            3x3 classification grid (top-2 + conf)
    recon_grid_<variant>.png          3x3 original|reconstruction grid (AE)
"""

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
import os


def train_one(variant, tr, te, args, seed):
    """Train a single model; returns (per-epoch metric dicts, model)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # detach_head is honoured natively by simple_core.build_model and is
    # silently ignored for linear / frozen / mlp.
    model = build_model(variant, args.res * args.res, args.hidden,
                        args.classes, args.activation,
                        detach_head=args.detach_head).to(DEVICE)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    history = []
    for _epoch in range(args.epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            ae_cycle = (model.has_ae
                        and not getattr(model, 'frozen', False)
                        and torch.rand(1).item() < args.ae_prob)
            if ae_cycle:
                model.local_loss(x).backward()
            else:
                F.cross_entropy(model(x), y).backward()
            optimizer.step()
        history.append(evaluate(model, te))
    return history, model


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--variants", nargs="+",
                        default=["linear", "frozen", "mlp", "nan"],
                        choices=["linear", "frozen", "mlp", "nan", "ann",
                                 "frozen_nan", "frozen_ann"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--res", type=int, default=28,
                        help="28 = native Fashion-MNIST resolution")
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--activation", default="sigmoid",
                        choices=["sigmoid", "relu"],
                        help="hidden activation for frozen/mlp "
                             "(nan/ann are sigmoid by construction)")
    parser.add_argument("--ae_prob", type=float, default=0.5,
                        help="P(batch is a reconstruction update) for nan/ann")
    parser.add_argument("--detach_head", action="store_true",
                        help="Bull's rule: train the AE hidden layer on "
                             "reconstruction ONLY, cutting the task gradient "
                             "at the hidden/head boundary. nan/ann only")
    args = parser.parse_args()

    sub = "simple_interleaved" + ("_det" if args.detach_head else "")
    t0 = time.time()
    tr, te, in_dim = get_loaders(resolution=args.res, num_classes=args.classes,
                                 batch_size=args.batch_size)
    n_train, n_test = len(tr.dataset), len(te.dataset)
    print(f"device={DEVICE} variants={args.variants} seeds={args.seeds} "
          f"epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
          f"res={args.res} hidden={args.hidden} act={args.activation} "
          f"ae_prob={args.ae_prob} detach_head={args.detach_head} "
          f"| train={n_train} test={n_test}")
    if args.detach_head:
        print("  [detach_head] AE hidden layers are trained by reconstruction "
              "ONLY; linear/frozen/mlp are unchanged.")

    curves, final_models = {}, {}
    for variant in args.variants:
        print(f"\n[{variant}]")
        runs = []
        for seed in range(args.seeds):
            hist, model = train_one(variant, tr, te, args, seed)
            runs.append(hist)
            if seed == 0:
                final_models[variant] = model     # seed-0 model for the grids
            m = hist[-1]
            rm = f"{m['recon_mse']:.4f}" if m["recon_mse"] is not None else "n/a"
            print(f"  seed {seed}: acc={m['acc']:.2f} top2={m['top2_acc']:.2f} "
                  f"ce={m['ce_loss']:.4f} recon={rm}")
        curves[variant] = aggregate(runs)

    # ---- save: one JSON with the full config + aggregated curves ----
    cfg = config_record(args, schedule=f"interleaved (ae_prob={args.ae_prob}) "
                        "for nan/ann; pure task for linear/frozen/mlp"
                        + ("; AE hidden layers DETACHED from the task gradient"
                           if args.detach_head else ""),
                        extra={"ae_prob": args.ae_prob,
                               "detach_head": args.detach_head,
                               "train_size": n_train, "test_size": n_test,
                               "variants": args.variants})
    save_json({"config": cfg, "curves": curves},
              "results_interleaved.json", subdir=sub)

    # ---- curve plots: one per metric ----
    d = outdir(sub)
    plot_metric(curves, "acc", "test accuracy (%)",
                os.path.join(d, "acc_curves.png"),
                "Test accuracy over training")
    plot_metric(curves, "top2_acc", "test top-2 accuracy (%)",
                os.path.join(d, "top2_curves.png"),
                "Top-2 accuracy over training")
    plot_metric(curves, "ce_loss", "test cross-entropy loss",
                os.path.join(d, "ce_curves.png"),
                "Cross-entropy loss over training")
    plot_metric(curves, "recon_mse", "reconstruction MSE",
                os.path.join(d, "recon_curves.png"),
                "Reconstruction MSE over training (AE variants)")

    # ---- 3x3 grids (seed-0 model, same 9 test images for every variant) ----
    for variant, model in final_models.items():
        classification_grid(
            model, te, args.res,
            os.path.join(d, f"cls_grid_{variant}.png"),
            f"Classifications — {variant} (true label; top-2 with confidence)")
        reconstruction_grid(
            model, te, args.res,
            os.path.join(d, f"recon_grid_{variant}.png"),
            f"Reconstructions — {variant}")

    print_final_table(curves)
    print(f"\ndone in {time.time() - t0:.1f}s. Outputs in {d}/")


if __name__ == "__main__":
    main()
