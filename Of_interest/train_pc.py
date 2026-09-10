"""
train_pc.py -- predictive coding versus backprop on a stack of AE layers.

Mirrors sweep_hidden.py's interface and output format so the results drop into
the same analysis. The difference is the learning rule and the fact that the
network is a STACK: predictive coding and backprop are near-identical with one
hidden layer, so depth is the whole point.

    --learning pc | backprop     which rule trains the stack
    --depth                      hidden layers in the stack
    --pc_steps, --pc_lr_v        relaxation settings (predictive coding only)
    --beta                       task signal reaching the stack; 0 is detached

READ THIS BEFORE INTERPRETING ANY COLLAPSE
An under-settled relaxation cannot carry the task error down the stack, so the
reconstruction objective wins and the layer goes constant. That looks exactly
like a genuine collapse and is not one. Measured on a NAN stack of depth 2:

    steps 16, lr_v 0.5   ->  chance,  code std 0.006
    steps 32, lr_v 0.5   ->  chance,  code std 0.007
    steps 64, lr_v 0.5   ->  100%,    code std 0.372
    steps 32, lr_v 1.0   ->  100%,    code std 0.370

Same network, same data, same seed. Always sweep pc_steps and pc_lr_v and
report the sweep; a single setting proves nothing about collapse.
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from models import DEVICE, outdir
except ImportError:
    from nan.models import DEVICE, outdir

from datasets_ext import get_loaders, default_classes
from predictive_coding import build_pc
from collapse_metrics import collapse_report, aggregate_reports


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ce = correct = top2 = n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        ce += F.cross_entropy(logits, y, reduction="sum").item()
        t2 = logits.topk(2, dim=1).indices
        correct += (t2[:, 0] == y).sum().item()
        top2 += (t2 == y.unsqueeze(1)).any(dim=1).sum().item()
        n += y.size(0)
    recon = None
    try:
        xb = next(iter(loader))[0].to(DEVICE)
        recon = float(model.local_loss(xb))
    except Exception:
        pass
    model.train()
    return {"acc": 100.0 * correct / n, "top2_acc": 100.0 * top2 / n,
            "ce_loss": ce / n, "recon_mse": recon}


def code_stats(model, loader, n_batches=8):
    """Per-layer activation spread. The direct read on whether a layer went
    constant, which is the failure mode both a real collapse and an
    under-settled relaxation produce."""
    model.eval()
    acc = None
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            a = model.activations(x.to(DEVICE))
            s = [float(h.std(dim=0).mean()) for h in a]
            acc = s if acc is None else [p + q for p, q in zip(acc, s)]
    model.train()
    return [v / min(n_batches, i + 1) for v in acc]


def run_one(variant, tr, te, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_pc(variant, args.res * args.res, args.hidden,
                     args.classes, args.depth).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist = []
    for ep in range(args.epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if args.learning == "pc":
                model.pc_update(x, y, opt, steps=args.pc_steps,
                                lr_v=args.pc_lr_v, beta=args.beta, lam=args.lam)
            else:
                model.bp_update(x, y, opt, beta=args.beta, lam=args.lam)
        hist.append(evaluate(model, te))
    return model, hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=["nan", "ann"])
    p.add_argument("--learning", default="pc", choices=["pc", "backprop"])
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--pc_steps", type=int, default=32)
    p.add_argument("--pc_lr_v", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=2.0)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--res", type=int, default=28)
    p.add_argument("--classes", type=int, default=None)
    p.add_argument("--dataset", default="emnist-balanced")
    p.add_argument("--collapse_samples", type=int, default=2000)
    p.add_argument("--subdir", default=None)
    args = p.parse_args()

    if args.classes is None:
        args.classes = default_classes(args.dataset)
    if args.subdir is None:
        args.subdir = (f"pc_{args.dataset}_d{args.depth}_{args.learning}"
                       + (f"_b{args.beta}" if args.beta != 1.0 else "")
                       + (f"_T{args.pc_steps}_lv{args.pc_lr_v}"
                          if args.learning == "pc" else ""))

    tr, te, _ = get_loaders(dataset=args.dataset, resolution=args.res,
                            num_classes=args.classes,
                            batch_size=args.batch_size)
    print(f"device={DEVICE} rule={args.learning} depth={args.depth} "
          f"H={args.hidden} beta={args.beta} seeds={args.seeds} "
          f"epochs={args.epochs}")
    if args.learning == "pc":
        print(f"  relaxation: {args.pc_steps} steps at lr_v={args.pc_lr_v}. "
              f"An under-settled relaxation looks exactly like collapse -- "
              f"sweep these before believing one.")

    results, t0 = {}, time.time()
    for variant in args.variants:
        per_seed, reps, codes = [], [], None
        for seed in range(args.seeds):
            model, hist = run_one(variant, tr, te, args, seed)
            per_seed.append(hist)
            reps.append(collapse_report(model, te, args.collapse_samples))
            if seed == 0:
                codes = code_stats(model, te)
            m = hist[-1]
            print(f"  {variant:8} seed={seed} acc={m['acc']:6.2f} "
                  f"recon={m['recon_mse']:.4f} "
                  f"code_std={['%.4f' % c for c in codes]}")
        keys = per_seed[0][0].keys()
        final = {}
        for k in keys:
            if per_seed[0][0][k] is None:
                final[k] = None
                continue
            v = np.array([h[-1][k] for h in per_seed])
            final[k] = {"mean": float(v.mean()),
                        "sem": float(v.std() / np.sqrt(len(v)))}
        results[variant] = {"final": final,
                            "collapse": aggregate_reports(reps),
                            "code_std": codes,
                            "curves": {"acc": [[h[e]["acc"]
                                                for e in range(args.epochs)]
                                               for h in per_seed]}}

    out = outdir(args.subdir)
    path = os.path.join(out, "results_pc.json")
    with open(path, "w") as f:
        json.dump({"config": vars(args) | {"device": DEVICE,
                                           "torch": torch.__version__},
                   "results": results}, f, indent=2)
    print(f"\nsaved {path}   [{time.time() - t0:.0f}s]")
    print(f"\n  {'variant':10}{'acc':>8}{'recon':>9}{'collapse':>10}"
          f"{'eff rank':>10}  code std per layer")
    def fmt(d, key, w, p_):
        # collapse_report returns None for metrics a variant does not expose
        v = (d or {}).get(key)
        return f"{v['mean']:{w}.{p_}f}" if isinstance(v, dict) else " " * (w - 3) + "n/a"

    for v, r in results.items():
        print(f"  {v:10}{r['final']['acc']['mean']:8.2f}"
              + fmt(r["final"], "recon_mse", 9, 4)
              + fmt(r["collapse"], "collapse_index", 10, 3)
              + fmt(r["collapse"], "eff_rank_act", 10, 2) + "  "
              + " ".join(f"{c:.4f}" for c in r["code_std"]))


if __name__ == "__main__":
    main()
