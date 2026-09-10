"""
predictive_coding.py -- train a stack of autoencoder layers without backprop.

WHY DEPTH IS THE POINT
Predictive coding replaces the backward pass with a relaxation. Each layer holds
a value node, predicts the layer above, and the mismatch between prediction and
value is an error node. Inference settles the values; learning then updates each
weight using only the error at its own layer and the activity just below it. No
gradient is transported across layers.

With ONE hidden layer that distinction barely exists. Backprop crosses exactly
one boundary, head to hidden, and predictive coding's relaxation reduces to
passing the same error through the same matrix transpose. The two agree almost
by construction, so a single-hidden-layer comparison cannot say anything.

It only becomes a real question with a stack. At two or three layers backprop
transports an exact gradient several boundaries down, while predictive coding
propagates errors by local settling, and they genuinely differ. That is also the
regime the whole project has never tested: every experiment so far has been
[input, hidden, output].

WHAT THIS PREDICTS, GIVEN THE GRADIENT-SCALING RESULT
One part in a thousand of the true task gradient was enough to take a detached
NAN layer from 5.96 to 63.14. Predictive coding is a far better approximation of
the gradient than that. So the expectation is that it trains fine, and the
interesting question is not "does it work" but "does a NAN stack still collapse
under it" -- because predictive coding still carries task information down,
just locally. Whether that behaves like a small beta or like full detachment is
not predictable from anything measured so far.

THE ENERGY
For layers v_0 (clamped input) ... v_L (output), with mu_l = W_l f(v_{l-1}):

    E = sum_l  ||v_l - mu_l||^2 / 2        (+ the local reconstruction terms)

Inference settles v_1..v_{L-1} by descending E with the ends clamped. Learning
then takes one step on the same E with the values fixed, which for W_l depends
only on eps_l and f(v_{l-1}) -- local in the sense that matters.

    --pc_steps    relaxation iterations per batch (T). Default 32.
    --pc_lr_v     step size for the value nodes. Default 0.5.

    THESE TWO MATTER MORE THAN THEY LOOK. An under-settled relaxation cannot
    carry the task error down the stack, so the reconstruction objective wins
    and the layer collapses -- and the symptom is indistinguishable from a
    genuine collapse result. Measured at depth 2 with lam = 2: at 8 steps and
    lr_v = 0.2 accuracy sat at chance with code std 0.006, and at 64 steps with
    lr_v = 0.5 the same run reached 100% with code std 0.372. Always sweep
    them before reporting a collapse under predictive coding.
    --beta        weight on the task error entering the hidden layers, so
                  beta = 0 is a detached stack trained by reconstruction alone

USAGE
    python train_pc.py --variants nan ann --depth 2 --learning pc
    python train_pc.py --variants nan --depth 3 --learning backprop   # control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models import DEVICE
except ImportError:
    from nan.models import DEVICE

from multi_decoder import build_model as _build_single


def _make_layer(variant, in_dim, width):
    """Reuse the existing layer zoo by building a throwaway classifier and
    keeping its hidden layer. Guarantees the stack uses exactly the same
    NAN / ANN / group / ring / multi implementations as every other experiment."""
    return _build_single(variant, in_dim, width, 2, "sigmoid").hidden


class PCStack(nn.Module):
    """L hidden autoencoder layers plus a linear head.

    forward() is an ordinary feedforward pass, so evaluation is identical
    whichever learning rule was used. The learning rules differ only in how
    the weights got there.
    """

    has_ae = True

    def __init__(self, variant, in_dim, width, num_classes, depth=2):
        super().__init__()
        self.depth = depth
        dims = [in_dim] + [width] * depth
        self.layers = nn.ModuleList(
            [_make_layer(variant, dims[i], dims[i + 1]) for i in range(depth)])
        self.head = nn.Linear(width, num_classes)
        self.variant = variant

    # -- ordinary use --------------------------------------------------------
    def activations(self, x):
        out, h = [], x
        for lay in self.layers:
            h = lay(h)
            out.append(h)
        return out

    def forward(self, x):
        return self.head(self.activations(x)[-1])

    def local_loss(self, x):
        """Reconstruction summed over the stack. Layer l reconstructs its own
        input, which is layer l-1's activity, not the image."""
        total, h = 0.0, x
        for lay in self.layers:
            total = total + lay.local_loss(h)
            h = lay(h).detach()
        return total / self.depth

    # -- predictive coding ---------------------------------------------------
    def pc_update(self, x, y, opt, steps=32, lr_v=0.5, beta=1.0, lam=2.0):
        """One predictive-coding batch: settle the values, then update weights.

        Values start at the feedforward pass, which is the standard warm start
        and makes the first inference step exact. Only the hidden values move;
        the input is clamped and the target enters through the output error.
        """
        with torch.no_grad():
            v = [h.detach().clone() for h in self.activations(x)]

        for i in range(len(v)):
            v[i].requires_grad_(True)

        # ---- inference: settle the value nodes ----
        for _ in range(steps):
            energy = 0.0
            h = x
            for l, lay in enumerate(self.layers):
                mu = lay(h)
                energy = energy + 0.5 * ((v[l] - mu) ** 2).sum(dim=1).mean()
                h = v[l]
            if beta != 0.0:
                task = F.cross_entropy(self.head(v[-1]), y)
                energy = energy + beta * task
            g = torch.autograd.grad(energy, v, create_graph=False)
            with torch.no_grad():
                for i in range(len(v)):
                    v[i] -= lr_v * g[i]
            for i in range(len(v)):
                v[i].requires_grad_(True)

        v = [q.detach() for q in v]

        # ---- learning: one step with the values held fixed ----
        # Each layer's term uses only its own input and its own settled value,
        # so no gradient crosses a layer boundary.
        opt.zero_grad()
        pred, rec = 0.0, 0.0
        h = x
        for l, lay in enumerate(self.layers):
            pred = pred + 0.5 * ((v[l] - lay(h)) ** 2).sum(dim=1).mean()
            rec = rec + lay.local_loss(h)
            h = v[l]                      # settled value, not a live activation
        # rec is divided by depth exactly as bp_update does. Without this the
        # stack gets `depth` times the reconstruction weight under predictive
        # coding, which at depth 2 and lam 2 was enough to stop it learning
        # while backprop at the same lam was unaffected.
        loss = pred + lam * rec / self.depth \
            + F.cross_entropy(self.head(v[-1]), y)
        loss.backward()
        opt.step()
        return float(loss.detach())

    # -- backprop control ----------------------------------------------------
    def bp_update(self, x, y, opt, beta=1.0, lam=2.0):
        opt.zero_grad()
        acts = []
        h = x
        rec = 0.0
        for lay in self.layers:
            rec = rec + lay.local_loss(h)
            h = lay(h)
            acts.append(h)
        top = acts[-1]
        if beta == 0.0:
            top = top.detach()
        elif beta != 1.0:
            top = _GradScale.apply(top, beta)
        loss = F.cross_entropy(self.head(top), y) + lam * rec / self.depth
        loss.backward()
        opt.step()
        return float(loss.detach())


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.beta, None


def build_pc(variant, in_dim, width, num_classes, depth=2):
    return PCStack(variant, in_dim, width, num_classes, depth)


if __name__ == "__main__":
    torch.manual_seed(0)
    D, W, C, B = 120, 32, 5, 128
    # separable toy data, so "does it learn at all" is answerable in seconds
    y = torch.randint(0, C, (B * 8,))
    x = (torch.rand(B * 8, D) * 0.3)
    for c in range(C):
        x[y == c, c * 20:(c + 1) * 20] += 0.7
    x = x.clamp(0, 1)

    print("predictive_coding self-test")
    for depth in (1, 2):
        for rule in ("pc", "backprop"):
            torch.manual_seed(0)
            m = build_pc("nan", D, W, C, depth)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            for ep in range(12):
                for i in range(0, x.size(0), B):
                    xb, yb = x[i:i + B], y[i:i + B]
                    if rule == "pc":
                        m.pc_update(xb, yb, opt, steps=8, lr_v=0.2)
                    else:
                        m.bp_update(xb, yb, opt)
            with torch.no_grad():
                acc = (m(x).argmax(1) == y).float().mean() * 100
            print(f"  depth={depth} {rule:9} acc={acc:6.2f}%")

    # beta = 0 must cut the task path in both rules
    for rule in ("pc", "backprop"):
        torch.manual_seed(0)
        m = build_pc("nan", D, W, C, 2)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        before = m.layers[0].encoder.weight.detach().clone()
        for i in range(0, x.size(0), B):
            if rule == "pc":
                m.pc_update(x[i:i + B], y[i:i + B], opt, steps=8, lr_v=0.2, beta=0.0)
            else:
                m.bp_update(x[i:i + B], y[i:i + B], opt, beta=0.0)
        moved = (m.layers[0].encoder.weight - before).abs().sum()
        print(f"  beta=0 {rule:9} first-layer encoder moved by {moved:.4f} "
              f"(reconstruction only, should be > 0)")
    print("  all checks ran")
