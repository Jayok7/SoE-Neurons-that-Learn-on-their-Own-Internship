"""
verify_detach.py -- prove the task gradient does or does not reach the encoder.

REPLACES THE EQUIVALENCE CHECK
The old check trained under both schedules and asked whether the numbers came
out the same. They did not, and it flagged a mismatch:

    ann   interleaved 80.79   combined 82.10   diff -1.31  <== INVESTIGATE

That was a false alarm, and the check was wrong rather than the run. Detaching
makes the two objectives touch DISJOINT PARAMETER SETS. It does not make them
deliver the same number of updates. Interleaved gives the encoder a
reconstruction step on ae_prob of batches; combined gives it one on every
batch, scaled by lam. The encoder therefore sees roughly twice the
reconstruction training under combined, and ANN -- which actually benefits from
more reconstruction, unlike collapsed NAN -- gains a point from it. Two runs
that differ in update count were never going to match, and the seed-to-seed
variation the comparison was measured against does not describe that
difference.

So the question is answered directly instead. One batch, cross-entropy only,
read the gradient on the encoder weight:

    detached      the gradient must be exactly zero
    attached      the gradient must be non-zero
    either way    the head must still be learning

No seeds, no schedule, no threshold, nothing to tune. It either holds or the
wiring is broken, and it costs one forward and one backward pass.

USAGE
    python verify_detach.py                       # all AE variants, detached
    python verify_detach.py --variants nan ann group8 dend4
    python verify_detach.py --no_detach           # check the attached case
    python verify_detach.py --hidden 128 --dataset emnist-balanced
"""
import argparse
import sys

import torch
import torch.nn.functional as F

try:
    from models import DEVICE
except ImportError:
    from nan.models import DEVICE

from simple_core import is_ae_variant, apply_detach_head
from dendritic_ae import build_model
from datasets_ext import get_loaders

try:
    from inhibition import wrap_inhibition
except ImportError:
    wrap_inhibition = None


def hidden_params(model):
    """Every trainable parameter of the hidden layer.

    Summing over all of them rather than picking out an encoder weight,
    because the question is whether cross-entropy reaches the hidden layer AT
    ALL, and because not every layer here has a `.encoder` -- DendriticAELayer
    does not, and singling out one tensor silently skipped it.
    """
    h = getattr(model, "hidden", None)
    if h is None:
        h = getattr(model, "fc", None)          # plain MLP
    if h is None:
        return []
    return [q for q in h.parameters() if q.requires_grad]


def check(variant, args, x, y):
    model = build_model(variant, x.shape[1], args.hidden, args.classes,
                        args.activation).to(DEVICE)
    detach = args.detach and is_ae_variant(variant)
    if detach:
        apply_detach_head(model, True, variant)
    if args.inhibition != "none" and wrap_inhibition is not None:
        wrap_inhibition(model, args.inhibition, k_frac=args.inhib_k_frac,
                        variant=variant)
        model = model.to(DEVICE)

    ps = hidden_params(model)
    if not ps:
        return variant, None, None, "no hidden layer", True

    model.zero_grad()
    F.cross_entropy(model(x), y).backward()
    enc = sum(0.0 if q.grad is None else float(q.grad.abs().sum()) for q in ps)
    head = float(model.head.weight.grad.abs().sum())

    if detach:
        ok = (enc == 0.0) and head > 0.0
        want = "zero"
    else:
        ok = enc > 0.0 and head > 0.0
        want = "non-zero"
    note = f"want {want}" + ("" if ok else "   <== BROKEN")
    return variant, enc, head, note, ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+",
                   default=["mlp", "nan", "ann", "group8", "dend4"])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--classes", type=int, default=None)
    p.add_argument("--res", type=int, default=28)
    p.add_argument("--dataset", default="fashion")
    p.add_argument("--activation", default="sigmoid")
    p.add_argument("--inhibition", default="none")
    p.add_argument("--inhib_k_frac", type=float, default=0.1)
    p.add_argument("--no_detach", dest="detach", action="store_false")
    args = p.parse_args()

    tr, te, in_dim = get_loaders(dataset=args.dataset, resolution=args.res,
                                 num_classes=args.classes, batch_size=64,
                                 verbose=False)
    x, y = next(iter(tr))
    x, y = x.to(DEVICE), y.to(DEVICE)
    if args.classes is None:
        args.classes = int(max(int(b.max()) for _, b in te) + 1)

    mode = "DETACHED" if args.detach else "ATTACHED"
    print(f"\ndetach verification -- {mode}   "
          f"({args.dataset}, H={args.hidden}, inhibition={args.inhibition})")
    print(f"  {'variant':12}{'|CE -> hidden|':>18}{'|CE -> head|':>15}   note")
    print("  " + "-" * 68)

    all_ok = True
    for v in args.variants:
        name, enc, head, note, ok = check(v, args, x, y)
        all_ok &= ok
        es = "   n/a" if enc is None else f"{enc:.6e}"
        hs = "   n/a" if head is None else f"{head:.4e}"
        flag = "" if is_ae_variant(v) or not args.detach else "  (not an AE variant)"
        print(f"  {name:12}{es:>18}{hs:>15}   {note}{flag}")

    print()
    if all_ok:
        print("  RESULT: gradient paths are exactly as configured.")
    else:
        print("  RESULT: BROKEN -- a gradient path is not what the flags say. "
              "Do not run the sweep.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
