#!/usr/bin/env python3
"""
run_experiments.py — thin command-line entrypoint for the nan package.

All the logic lives in the nan/ package; this file only parses arguments and
dispatches to the right experiment. Run `python run_experiments.py --help`.

Examples:
    python run_experiments.py --mode curves --arch static --seeds 10 --epochs 15
    python run_experiments.py --mode div_sweep --arch spiking --penalty activation
    python run_experiments.py --mode schedule_compare --arch static --hidden 20
    python run_experiments.py --mode penalised_compare --arch static --div_weight 2.0
"""

import time
import argparse
import numpy as np
import torch

# Robust combined imports supporting both package usage and direct execution
try:
    from .models import DEVICE, OUT, HiddenLayerClassifier, get_loaders
    from .experiments import (
        run_interleaved, multiseed, save_json, run_and_visualize,
        exp_curves, exp_schedule_compare, exp_sweep, exp_seq_variants,
        exp_penalised_schedule_compare,
    )
except ImportError:
    try:
        from nan.models import DEVICE, OUT, HiddenLayerClassifier, get_loaders
        from nan.experiments import (
            run_interleaved, multiseed, save_json, run_and_visualize,
            exp_curves, exp_schedule_compare, exp_sweep, exp_seq_variants,
            exp_penalised_schedule_compare,
        )
    except ImportError:
        from models import DEVICE, OUT, HiddenLayerClassifier, get_loaders
        from experiments import (
            run_interleaved, multiseed, save_json, run_and_visualize,
            exp_curves, exp_schedule_compare, exp_sweep, exp_seq_variants,
            exp_penalised_schedule_compare,
        )



def main():
        import argparse
        parser = argparse.ArgumentParser(
            description="Neuron-autoencoder experiment harness. Pick a mode and "
                        "(optionally) an architecture; results land in ./experiments/.")
        parser.add_argument("--mode", default="demo",
                            choices=["demo", "curves", "div_sweep", "local_sweep",
                                     "seq_variants", "compare_arch", "compare_all4",
                                     "visualize", "visualize_all", "schedule_compare",
                                     "penalised_compare"],
                            help="which experiment to run")
        parser.add_argument("--arch", default="static",
                            choices=["static", "spiking", "ann", "nn"],
                            help="hidden-layer architecture for single-arch modes")
        parser.add_argument("--seeds", type=int, default=5,
                            help="number of random seeds to average over")
        parser.add_argument("--epochs", type=int, default=10,
                            help="training epochs (sweeps use ~60%% of this)")
        parser.add_argument("--res", type=int, default=14,
                            help="Fashion-MNIST resolution (the N-axis): 7, 14 or 28")
        parser.add_argument("--classes", type=int, default=10,
                            help="number of garment classes to keep (2..10)")
        parser.add_argument("--hidden", type=int, default=128,
                            help="hidden-layer width")
        parser.add_argument("--penalty", default="decoder",
                            choices=["decoder", "activation"],
                            help="diversity penalty formulation for div_sweep: "
                                 "'decoder' = weight decorrelation (original), "
                                 "'activation' = Barlow-Twins representation decorrelation")
        parser.add_argument("--div_weight", type=float, default=2.0,
                            help="diversity penalty strength for penalised_compare mode "
                                 "(default: 2.0)")
        args = parser.parse_args()
    
        seeds = range(args.seeds)
        sweep_epochs = max(1, int(args.epochs * 0.6))   # sweeps need fewer epochs
        t0 = time.time()
        print(f"device={DEVICE} mode={args.mode} arch={args.arch} "
              f"seeds={args.seeds} epochs={args.epochs} res={args.res} "
              f"classes={args.classes} hidden={args.hidden}")
    
        if args.mode == "demo":
            # tiny everything, run BOTH NAN archs so directive compliance is visible
            exp_curves(seeds=[0, 1], epochs=2, res=7, nc=10, hidden=64, arch="static")
            exp_curves(seeds=[0, 1], epochs=2, res=7, nc=10, hidden=64, arch="spiking")
    
        elif args.mode == "curves":
            exp_curves(seeds=seeds, epochs=args.epochs, res=args.res,
                       nc=args.classes, hidden=args.hidden, arch=args.arch)
    
        elif args.mode == "schedule_compare":
            exp_schedule_compare(seeds=seeds, epochs=args.epochs, res=args.res,
                                 nc=args.classes, hidden=args.hidden, arch=args.arch)
    
        elif args.mode == "div_sweep":
            exp_sweep("div", [0, 0.1, 0.5, 1, 2, 5, 10, 20], seeds=seeds,
                      epochs=sweep_epochs, res=args.res, nc=args.classes,
                      hidden=args.hidden, arch=args.arch, penalty=args.penalty)
    
        elif args.mode == "local_sweep":
            exp_sweep("local", [0, 0.1, 0.25, 0.5, 1, 2, 4], seeds=seeds,
                      epochs=sweep_epochs, res=args.res, nc=args.classes,
                      hidden=args.hidden, arch=args.arch)
    
        elif args.mode == "seq_variants":
            exp_seq_variants(seeds=seeds, epochs=args.epochs, res=args.res,
                             nc=args.classes, hidden=args.hidden, arch=args.arch)
    
        elif args.mode == "compare_arch":
            # same experiment for both NAN neuron models: static vs spiking
            for a in ("static", "spiking"):
                exp_seq_variants(seeds=seeds, epochs=args.epochs, res=args.res,
                                 nc=args.classes, hidden=args.hidden, arch=a)
                exp_sweep("div", [0, 0.5, 1, 2, 5, 10], seeds=seeds,
                          epochs=sweep_epochs, res=args.res, nc=args.classes,
                          hidden=args.hidden, arch=a)
    
        elif args.mode == "compare_all4":
            # the paper's full architecture comparison: NN vs ANN vs NAN(static) vs
            # NAN(spiking), interleaved training, final accuracy + diversity per arch.
            tr, te, in_dim = get_loaders(resolution=args.res, num_classes=args.classes)
            xb = next(iter(te))[0].to(DEVICE)
            print(f"\n  {'arch':10} {'acc':>8} {'+-sem':>7} {'eff_rank':>9} {'dec_sim':>8}")
            summary = {}
            for a in ("nn", "ann", "static", "spiking"):
                agg = multiseed(lambda s: run_interleaved(
                    HiddenLayerClassifier(in_dim, args.hidden, args.classes, arch=a).to(DEVICE),
                    tr, te, xb, epochs=args.epochs, local_weight=1.0),
                    seeds, in_dim, args.hidden, args.classes)
                summary[a] = {"acc": agg["acc"]["mean"][-1], "acc_sem": agg["acc"]["sem"][-1],
                              "eff_rank": agg["eff_rank"]["mean"][-1],
                              "dec_sim": agg["dec_sim"]["mean"][-1]}
                v = summary[a]
                ds = "n/a" if v["dec_sim"] < 0 else f"{v['dec_sim']:.3f}"
                print(f"  {a:10} {v['acc']:>8.2f} {v['acc_sem']:>7.2f} {v['eff_rank']:>9.1f} {ds:>8}")
            save_json(summary, "compare_all4.json", subdir="compare_all4")
    
        elif args.mode == "visualize":
            # Train a single model and generate neuron-level visualizations
            print(f">>> visualize: training {args.arch} and generating neuron plots")
            torch.manual_seed(42)
            np.random.seed(42)
            run_and_visualize(arch=args.arch, res=args.res, nc=args.classes,
                              hidden=args.hidden, epochs=args.epochs,
                              subdir="visualize")
    
        elif args.mode == "visualize_all":
            # Visualize all four architectures side-by-side for comparison
            print(">>> visualize_all: training all architectures and generating neuron plots")
            for a in ("nn", "ann", "static", "spiking"):
                print(f"\n  --- {a} ---")
                torch.manual_seed(42)
                np.random.seed(42)
                run_and_visualize(arch=a, res=args.res, nc=args.classes,
                                  hidden=args.hidden, epochs=args.epochs,
                                  subdir="visualize")

        elif args.mode == "penalised_compare":
            exp_penalised_schedule_compare(
                seeds=seeds, epochs=args.epochs, div_weight=args.div_weight,
                res=args.res, nc=args.classes, hidden=args.hidden, arch=args.arch)
    
        print(f"\ndone in {time.time()-t0:.1f}s. Outputs in ./{OUT}/")


if __name__ == "__main__":
    main()

