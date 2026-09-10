"""
train_ttfs.py -- train the time-to-first-spike NAN and watch the grouping form.

Trains one model per requested variant and records, after every epoch:

    accuracy        held-out test set
    recon MSE       each layout's own local objective
    effective rank  of the hidden CODES (first-spike times), same estimator
                    collapse_metrics uses, so the numbers sit on the same scale
                    as the rate-based project

and, for the dynamic routings, the diagnostics that say whether the grouping
found anything:

    used_groups          effective number of groups actually in use. If this
                         decays toward 1 the routing has collapsed and the
                         layer has quietly become a spiking ANN.
    within / between     mean spike-time gap between neurons in the same group
                         versus different groups. THIS IS THE RESULT. A
                         grouping built by coincidence detection should show a
                         smaller within-group gap; if within == between, the
                         rule has assigned neurons without finding structure.
    assign_entropy       0 = every neuron committed, 1 = spread evenly.

WHAT IT PRODUCES
    results/<dataset>_H<width>_T<steps>/
        curves.png       accuracy, recon, effective rank -- one line per variant
        routing.png      ONLY when a dynamic routing was run: used_groups and
                         the within/between spike-time gap over epochs
        curves.json      every number
        summary.txt      first/last epoch per variant, as text

USAGE
    python train_ttfs.py --variants ttfs_nan ttfs_group16 ttfs_ann
    python train_ttfs.py --variants ttfs_group16 ttfs_group16-coincidence
    python train_ttfs.py --epochs 30 --seeds 3 --hidden 128 --num_steps 30
    python train_ttfs.py --epochs 1 --seeds 1 --limit_batches 20   # smoke test

THE COMPARISON THAT MATTERS
    ttfs_group<G>                fixed block partition   -- the control
    ttfs_group<G>-coincidence    the same G, grouped by spike timing
    ttfs_group<G>-learned        the same G, grouped by gradient

    Same G, same parameter count, same everything else. Week 10 established
    that at fixed G every *fixed* partition performs identically, so any gap
    between these three is attributable to how the partition was chosen, which
    is the question the coincidence rule exists to answer.

RUN THIS DETACHED  (--detach_head)
    The rate-based re-run measured the group dial on both sides of the task
    gradient. Attached, going from one shared decoder to one per neuron -- a
    128-fold change in the layout -- moved accuracy by 0.22 points on EMNIST.
    Detached, the same dial spans 57 points and the collapse index rises
    monotonically with it.

    Choosing the partition differently at FIXED G is a smaller intervention
    than changing G. So an attached routing comparison is being run inside a
    region already measured to be flat, and a null result there would say
    nothing about the coincidence rule. --detach_head puts the run in the
    regime where the grouping is load-bearing.

    A direct check runs before training: one batch, cross-entropy only,
    assert the encoder gradient is exactly zero when detached and non-zero
    when not. That is a proof rather than a comparison of two runs, so it does
    not need seeds or care about schedule randomness.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_HERE, "..", "Week_10"),
              os.path.join(_HERE, "Week_10"), os.path.join(_HERE, "..")):
    if os.path.isfile(os.path.join(_cand, "collapse_metrics.py")):
        sys.path.insert(0, os.path.abspath(_cand))
        break
else:
    sys.exit("Cannot find Week_10/. Put this directory next to it, or set "
             "PYTHONPATH=/path/to/Week_10")
sys.path.insert(0, _HERE)

from datasets_ext import get_loaders                       # noqa: E402
from collapse_metrics import effective_rank, _singular_values  # noqa: E402
from ttfs_nan import build_ttfs                            # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- metrics -----------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, max_rank_samples, limit=None):
    """Accuracy, reconstruction MSE, and effective rank of the spike codes."""
    model.eval()
    correct = n = 0
    recon_sum = 0.0
    codes = []
    for i, (x, y) in enumerate(loader):
        if limit and i >= limit:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        recon_sum += model.local_loss(x).item() * x.size(0)
        if len(codes) * x.size(0) < max_rank_samples:
            codes.append(model.codes(x).cpu())
        n += y.size(0)
    c = torch.cat(codes)[:max_rank_samples].float()
    c = c - c.mean(0, keepdim=True)
    return {"acc": 100.0 * correct / n,
            "recon_mse": recon_sum / n,
            "eff_rank": float(effective_rank(_singular_values(c)))}


# ---- detach verification -----------------------------------------------------
def verify_detach(model, x, y, want_detached):
    """Assert the task gradient does (or does not) reach the encoder.

    One batch, cross-entropy only. This replaces comparing two training runs
    and asking whether they came out the same: the schedules differ in how
    many reconstruction updates they deliver, so they were never going to
    match, and a mismatch there said nothing about whether the detach worked.
    Reading the gradient answers the question directly.
    """
    model.zero_grad()
    F.cross_entropy(model(x), y).backward()
    g = model.hidden.encoder.weight.grad
    mag = 0.0 if g is None else float(g.abs().sum())
    head_mag = float(model.head.weight.grad.abs().sum())
    model.zero_grad()
    if want_detached and mag != 0.0:
        raise RuntimeError(
            f"--detach_head set but cross-entropy still reaches the encoder "
            f"(|grad| = {mag:.3e}). Do not trust this run.")
    if not want_detached and mag == 0.0:
        raise RuntimeError("task gradient is not reaching the encoder even "
                           "though --detach_head was not set.")
    if head_mag == 0.0:
        raise RuntimeError("the head is receiving no gradient at all.")
    return mag, head_mag


# ---- one run -----------------------------------------------------------------
def train_one(variant, args, tr, te, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_ttfs(variant, args.res * args.res, args.hidden, args.classes,
                       num_steps=args.num_steps, beta=args.beta,
                       slope=args.slope, head=args.head, latency=args.latency,
                       tau_route=args.tau_route, hebb_eta=args.hebb_eta,
                       sigma=args.sigma, init_logit=args.init_logit,
                       route_norm=args.route_norm,
                       detach_head=args.detach_head).to(DEVICE)

    # Calibrate the threshold on one real batch before anything trains. Skip
    # this and the layer is silent -- see ttfs_nan.calibrate.
    xb, yb = next(iter(tr))
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    thr = model.calibrate(xb, target_fire=args.target_fire)

    # Prove the gradient path is what the flags claim, before spending GPU
    # hours on it.
    enc_g, head_g = verify_detach(model, xb, yb, args.detach_head)
    if seed == 0:
        print(f"      detach check: |CE->encoder| = {enc_g:.3e}, "
              f"|CE->head| = {head_g:.3e}  OK")

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for i, (x, y) in enumerate(tr):
            if args.limit_batches and i >= args.limit_batches:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            if args.schedule == "combined":
                # Both objectives on every batch. Costs two spike-train passes
                # per batch instead of one, which is the dominant cost here --
                # budget roughly double the wall time.
                loss = (F.cross_entropy(model(x), y)
                        + args.lam * model.local_loss(x))
                loss.backward()
            elif torch.rand(1).item() < args.ae_prob:
                # Interleaved, matching Week 10 and Weeks 5-9: each batch is
                # either a task update or a reconstruction update, never both.
                model.local_loss(x).backward()
            else:
                F.cross_entropy(model(x), y).backward()
            opt.step()

        row = evaluate(model, te, args.rank_samples, args.limit_batches)
        row["epoch"] = epoch
        row.update({k: v for k, v in model.group_stats(xb).items()
                    if k in ("used_groups", "assign_entropy",
                             "within_group_gap", "between_group_gap",
                             "mean_first_spike", "frac_silent",
                             "mean_abs_cos_dec", "route_top1")})
        hist.append(row)
    return hist, thr


def aggregate(runs):
    keys = [k for k in runs[0][0] if k != "epoch"]
    out = {"epochs": [r["epoch"] for r in runs[0]]}
    for k in keys:
        rows = []
        for i in range(len(runs[0])):
            vals = [r[i][k] for r in runs if r[i].get(k) is not None]
            rows.append(vals)
        if not any(rows):
            out[k] = None
            continue
        out[k] = {
            "mean": [float(np.mean(v)) if v else None for v in rows],
            "sem": [float(np.std(v) / np.sqrt(len(v))) if len(v) > 1 else 0.0
                    for v in rows],
        }
    return out


# ---- figures -----------------------------------------------------------------
NAVY, MUTED, RULE, GREEN = "#1E2761", "#5B6478", "#D8D8E0", "#196B24"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12,
        "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": RULE, "grid.alpha": 0.8,
        "figure.facecolor": "white", "savefig.facecolor": "white"})
    return plt


def _lines(ax, results, order, key, colours):
    import numpy as np
    for label, col in zip(order, colours):
        s = results[label].get(key)
        if not s:
            continue
        ep = results[label]["epochs"]
        m = np.array([np.nan if v is None else v for v in s["mean"]],
                     dtype=float)
        e = np.array(s["sem"], dtype=float)
        ax.plot(ep, m, lw=2, marker="o", ms=3.5, color=col, label=label)
        if e.any():
            ax.fill_between(ep, m - e, m + e, color=col, alpha=0.18, lw=0)


def plot_curves(results, order, args, path):
    plt = _style()
    from matplotlib import cm, ticker
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 12.8), sharex=True)
    colours = cm.viridis(np.linspace(0.08, 0.92, len(order)))
    panels = [("acc", "test accuracy (%)", "Accuracy"),
              ("recon_mse", "reconstruction MSE",
               "Reconstruction error (each variant's own objective)"),
              ("eff_rank", f"effective rank (of {args.hidden})",
               "Effective rank of the spike-time codes")]
    for ax, (key, ylabel, title) in zip(axes, panels):
        _lines(ax, results, order, key, colours)
        ax.set_ylabel(ylabel)
        ax.set_title(title, color=NAVY, fontweight="bold", loc="left", pad=8)
    top = max(v for lab in order for v in results[lab]["eff_rank"]["mean"]
              if v is not None)
    if top > 0.4 * args.hidden:
        axes[2].set_ylim(0, args.hidden * 1.12)
        axes[2].axhline(args.hidden, color=GREEN, ls="--", lw=1.4)
        axes[2].text(0.99, 0.95, f"ceiling: H = {args.hidden}",
                     transform=axes[2].transAxes, color=GREEN, fontsize=11,
                     ha="right", va="top")
    else:
        axes[2].set_ylim(0, max(top * 1.3, 1e-6))
        axes[2].text(0.99, 0.95, f"ceiling H = {args.hidden}, off scale",
                     transform=axes[2].transAxes, color=GREEN, fontsize=11,
                     ha="right", va="top")
    axes[2].set_xlabel("epoch")
    axes[2].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(order), 4),
               frameon=False, fontsize=10.5)
    fig.suptitle("Time-to-first-spike NAN", color=NAVY, fontsize=16,
                 fontweight="bold", y=0.998)
    fig.text(0.5, 0.976, f"{args.dataset}   |   H = {args.hidden}   |   "
             f"T = {args.num_steps} steps   |   {args.seeds} seed(s)   |   "
             f"interleaved   |   band = SEM",
             ha="center", color=MUTED, fontsize=11)
    fig.tight_layout(rect=[0, 0.055, 1, 0.968])
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  saved {path}")


def plot_routing(results, order, args, path):
    """Only drawn when a dynamic routing was run -- for fixed routing both
    panels are flat lines by construction and the file would be noise."""
    plt = _style()
    from matplotlib import cm, ticker
    dyn = [l for l in order if "-" in l]
    if not dyn:
        return False
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 8.6), sharex=True)
    colours = cm.viridis(np.linspace(0.08, 0.92, len(dyn)))

    _lines(axes[0], results, dyn, "used_groups", colours)
    axes[0].axhline(args.groups_hint, color=GREEN, ls="--", lw=1.4)
    axes[0].text(0.99, 0.94, f"all {args.groups_hint} groups in use",
                 transform=axes[0].transAxes, color=GREEN, fontsize=11,
                 ha="right", va="top")
    axes[0].set_ylim(0, args.groups_hint * 1.15)
    axes[0].set_ylabel("effective groups in use")
    axes[0].set_title("Does the routing collapse?", color=NAVY,
                      fontweight="bold", loc="left", pad=8)

    for label, col in zip(dyn, colours):
        ep = results[label]["epochs"]
        for key, ls, tag in [("within_group_gap", "-", "within group"),
                             ("between_group_gap", "--", "between groups")]:
            s = results[label].get(key)
            if not s:
                continue
            m = np.array([np.nan if v is None else v for v in s["mean"]],
                         dtype=float)
            axes[1].plot(ep, m, ls, lw=2, color=col,
                         label=f"{label} - {tag}")
    axes[1].set_ylabel("mean spike-time gap (steps)")
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Did the grouping find anything? "
                      "(solid below dashed = yes)", color=NAVY,
                      fontweight="bold", loc="left", pad=8)
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # One legend only: the lower panel names every variant twice already
    # (within and between), so a figure-level legend would just repeat it.
    axes[1].legend(frameon=False, fontsize=10, loc="best")
    fig.suptitle("Routing diagnostics", color=NAVY, fontsize=16,
                 fontweight="bold", y=0.997)
    fig.text(0.5, 0.945, "dynamic routings only - a fixed partition is flat "
             "in both panels by construction", ha="center", color=MUTED,
             fontsize=11)
    fig.tight_layout(rect=[0, 0.01, 1, 0.935])
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  saved {path}")
    return True


def summary(results, order):
    head = (f"{'variant':28} {'acc@1':>7} {'acc@end':>8} {'recon@end':>10} "
            f"{'rank@end':>9} {'groups':>7} {'within':>7} {'between':>8} "
            f"{'silent':>7}")
    lines = [head, "-" * len(head)]
    for label in order:
        r = results[label]
        g = lambda k, i=-1: (None if not r.get(k)
                             else r[k]["mean"][i])
        fm = lambda v, f: "n/a" if v is None else format(v, f)
        lines.append(
            f"{label:28} {fm(g('acc', 0), '7.2f')} {fm(g('acc'), '8.2f')} "
            f"{fm(g('recon_mse'), '10.4f')} {fm(g('eff_rank'), '9.2f')} "
            f"{fm(g('used_groups'), '7.2f')} {fm(g('within_group_gap'), '7.2f')} "
            f"{fm(g('between_group_gap'), '8.2f')} "
            f"{fm(g('frac_silent'), '7.2f')}")
    return "\n".join(lines)


# ---- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train the TTFS spiking NAN.")
    ap.add_argument("--dataset", default="emnist-balanced")
    ap.add_argument("--variants", nargs="+",
                    default=["ttfs_group16", "ttfs_group16-coincidence"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--num_steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--res", type=int, default=28)
    ap.add_argument("--classes", type=int, default=None)
    ap.add_argument("--ae_prob", type=float, default=0.5,
                    help="interleaved only: P(batch is a reconstruction update)")
    ap.add_argument("--schedule", default="interleaved",
                    choices=["interleaved", "combined"],
                    help="combined runs both objectives on every batch and "
                         "costs roughly double, because the spike train is "
                         "unrolled once per objective")
    ap.add_argument("--lam", type=float, default=2.0,
                    help="combined only: weight on the reconstruction term")
    ap.add_argument("--detach_head", action="store_true",
                    help="Bull's rule: train the spiking layer on "
                         "reconstruction alone. This is the regime where the "
                         "grouping question is answerable -- see the module "
                         "docstring")
    ap.add_argument("--route_norm", default="zscore",
                    choices=["zscore", "center"],
                    help="zscore standardises the coincidence score per neuron "
                         "so tau_route is a real sharpness knob and the sigma "
                         "sweep is not confounded with routing strength. "
                         "center reproduces the original behaviour")
    ap.add_argument("--beta", type=float, default=0.9,
                    help="LIF membrane decay")
    ap.add_argument("--slope", type=float, default=25.0,
                    help="fast-sigmoid surrogate slope")
    ap.add_argument("--head", default="code", choices=["code", "ttfs"])
    ap.add_argument("--latency", default="linear", choices=["linear", "log"])
    ap.add_argument("--target_fire", type=float, default=0.9)
    ap.add_argument("--tau_route", type=float, default=0.25,
                    help="routing softmax temperature. Lower is sharper. With "
                         "route_norm=zscore the logits equilibrate near unit "
                         "scale, so 0.1-0.5 is the useful range; sweep it")
    ap.add_argument("--hebb_eta", type=float, default=0.02)
    ap.add_argument("--sigma", type=float, default=None,
                    help="coincidence width in timesteps (default T/8)")
    ap.add_argument("--init_logit", type=float, default=3.0,
                    help="how committed a dynamic routing starts. 3.0 is very "
                         "sticky: gradient-based `learned` routing will not "
                         "flip a single neuron out of its initial partition "
                         "in a short run. Use ~0.5 when comparing against "
                         "-learned, so both routings start equally free.")
    ap.add_argument("--rank_samples", type=int, default=2000)
    ap.add_argument("--limit_batches", type=int, default=0,
                    help="cap batches per epoch, for smoke tests")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    tr, te, in_dim = get_loaders(dataset=args.dataset, resolution=args.res,
                                 num_classes=args.classes,
                                 batch_size=args.batch, verbose=True)
    args.res = int(round(in_dim ** 0.5))
    if args.classes is None:
        args.classes = int(max(int(y.max()) for _, y in te) + 1)
    args.groups_hint = max(
        [int(v.split("group")[1].split("-")[0])
         for v in args.variants if "group" in v] or [args.hidden])

    out = args.outdir or os.path.join(
        _HERE, "results",
        f"{args.dataset}_H{args.hidden}_T{args.num_steps}"
        + ("_det" if args.detach_head else "_att")
        + ("" if args.schedule == "interleaved" else "_comb"))
    os.makedirs(out, exist_ok=True)

    print(f"\ndevice={DEVICE}  H={args.hidden}  T={args.num_steps}  "
          f"epochs={args.epochs}  seeds={args.seeds}")
    print(f"schedule={args.schedule}  detach_head={args.detach_head}  "
          f"route_norm={args.route_norm}  tau_route={args.tau_route}")
    if not args.detach_head:
        print("  [warning] running ATTACHED. The rate-based sweep measured the "
              "whole group dial at 0.22 points in this regime; a routing "
              "difference at fixed G is a smaller effect than that.")
    print(f"variants: {' '.join(args.variants)}")
    print(f"output -> {out}\n")

    t0 = time.time()
    results, order = {}, []
    for i, variant in enumerate(args.variants, 1):
        runs = []
        for seed in range(args.seeds):
            t1 = time.time()
            hist, thr = train_one(variant, args, tr, te, seed)
            runs.append(hist)
            last = hist[-1]
            print(f"  [{i}/{len(args.variants)}] {variant:26} seed={seed}  "
                  f"acc={last['acc']:.2f}  recon={last['recon_mse']:.4f}  "
                  f"rank={last['eff_rank']:.1f}  "
                  f"thr={thr:.3f}  silent={last['frac_silent']:.2f}  "
                  f"[{time.time()-t1:.0f}s]")
        results[variant] = aggregate(runs)
        order.append(variant)

    with open(os.path.join(out, "curves.json"), "w") as f:
        json.dump({"config": vars(args), "order": order, "results": results},
                  f, indent=1)
    text = summary(results, order)
    print("\n" + text)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    plot_curves(results, order, args, os.path.join(out, "curves.png"))
    plot_routing(results, order, args, os.path.join(out, "routing.png"))
    print(f"\ndone in {time.time()-t0:.0f}s. Outputs in {out}/")


if __name__ == "__main__":
    main()
