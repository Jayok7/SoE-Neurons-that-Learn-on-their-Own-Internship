"""
dendritic_ae.py — DEPTH inside a NAN node.

The supervisor's two suggestions are orthogonal axes, and only one of them was
already implemented:

    WIDTH per node   how many units a node emits.  784 -> k -> 784.
                     ALREADY DONE: this is group_ae.py swept over G, with
                     k = H/G units per node. k=1 is NAN, k=H is ANN, both exact.

    DEPTH per node   how much computation happens inside a node before it emits
                     its single scalar.  784 -> m -> 1 -> m -> 784.
                     THIS FILE. The node still emits one number, exactly as in
                     NAN, but the mapping to and from that number is no longer
                     a single linear step.

Keeping them separate matters: the width axis changes what the classifier sees
(H stays fixed but nodes pool differently), whereas the depth axis leaves the
classifier's input completely unchanged at H values. So any accuracy change
under depth cannot be attributed to the head getting more features.

--------------------------------------------------------------------------
THE DENDRITIC NODE
--------------------------------------------------------------------------
Modelled on the dendrite literature, where a neuron is a small MLP: dendritic
subunits each integrate the input nonlinearly, and the soma combines them into
one output. Node j computes

    d_j = act( W1_j x + b1_j )            m dendritic subunits   (B, H, m)
    h_j = sigmoid( s_j . d_j + c_j )      the soma, one scalar   (B, H)
    e_j = act( h_j u_j + p_j )            m subunits on the way back
    x^_j = W2_j e_j + q_j                 this node's reconstruction (B, H, in)

and the loss is NAN's, unchanged: every node is scored on reconstructing the
whole input by itself, and the errors are averaged over nodes.

    m = 1 with `inner_act="linear"` is FUNCTIONALLY NAN. The encoder collapses
    to a single linear map 784 -> 1, and the decoder to h*(u*W2) + (p*W2 + q),
    which is NAN's  h*w + b  with a reparameterised w and b. Verified in the
    self-test by fitting both and comparing losses, not just asserted.

--------------------------------------------------------------------------
THE COST, STATED HONESTLY
--------------------------------------------------------------------------
Per node the parameter count is roughly 2*m*784, against NAN's 2*784. So depth
m multiplies hidden-layer parameters by about m:

    H = 128, m = 1      ~0.3 M      (= NAN)
    H = 128, m = 4      ~0.8 M      (2.7x NAN)
    H = 128, m = 8      ~1.6 M      (5.3x NAN)

The supervisor called this "resource heavy handed" and that is correct. It is
also a CONFOUND, not just a cost: a deeper node might win purely on parameters.
The width sweep does not have this problem -- group_ae is parameter-identical
to NAN at every G -- so any depth result needs a parameter-matched control.
`matched_width_for_depth()` below computes the NAN width with the same
parameter count, which is the control to run alongside.

Activation memory is unchanged from NAN: the (B, H, in_dim) reconstruction
tensor still dominates, and d/e are only (B, H, m).

--------------------------------------------------------------------------
USE
--------------------------------------------------------------------------
    --variants nan ann dend2 dend4 dend8
    --variants dend4-relu          (inner activation relu instead of tanh)

Import build_model from here instead of group_ae to get both families; it
defers to group_ae for group* variants and to simple_core for everything else.
"""

import os
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
for _c in (_HERE, os.path.join(_HERE, os.pardir, "Week_5"),
           os.path.join(_HERE, os.pardir, "Week_4")):
    _c = os.path.abspath(_c)
    if os.path.isdir(_c) and _c not in sys.path:
        sys.path.append(_c)

from group_ae import build_model as _build_group

_ACTS = {"tanh": torch.tanh, "relu": F.relu, "sigmoid": torch.sigmoid,
         "linear": lambda z: z}


class DendriticAELayer(nn.Module):
    """A NAN layer whose nodes each contain a small two-layer autoencoder."""

    per_neuron_decoder = False      # the decoder is nonlinear, so no single row

    def __init__(self, in_features, out_features, n_dendrites=4,
                 inner_act="tanh"):
        super().__init__()
        if inner_act not in _ACTS:
            raise ValueError(f"inner_act must be one of {sorted(_ACTS)}")
        H, m, D = out_features, n_dendrites, in_features
        self.in_features, self.out_features = D, H
        self.n_dendrites, self.inner_act = m, inner_act
        self.act = _ACTS[inner_act]

        # encoder: x -> m dendrites -> 1 soma
        self.enc_w = nn.Parameter(torch.empty(H, m, D))
        self.enc_b = nn.Parameter(torch.zeros(H, m))
        self.soma_w = nn.Parameter(torch.empty(H, m))
        self.soma_b = nn.Parameter(torch.zeros(H))
        # decoder: soma -> m subunits -> full image
        self.dec_in_w = nn.Parameter(torch.empty(H, m))
        self.dec_in_b = nn.Parameter(torch.zeros(H, m))
        self.dec_w = nn.Parameter(torch.empty(H, m, D))
        self.dec_b = nn.Parameter(torch.zeros(H, D))

        bound = (1.0 / D) ** 0.5
        nn.init.uniform_(self.enc_w, -bound, bound)
        nn.init.uniform_(self.dec_w, -0.01, 0.01)
        nn.init.uniform_(self.soma_w, -(1.0 / m) ** 0.5, (1.0 / m) ** 0.5)
        nn.init.uniform_(self.dec_in_w, -1.0, 1.0)

    def forward(self, x):
        """(B, in) -> (B, H). The node still emits exactly one value."""
        d = self.act(torch.einsum("bi,hmi->bhm", x, self.enc_w) + self.enc_b)
        return torch.sigmoid(torch.einsum("bhm,hm->bh", d, self.soma_w)
                             + self.soma_b)

    def reconstruct_per_neuron(self, code):
        """(B, H) -> (B, H, in): each node's own reconstruction."""
        e = self.act(code.unsqueeze(2) * self.dec_in_w + self.dec_in_b)
        return torch.einsum("bhm,hmi->bhi", e, self.dec_w) + self.dec_b

    def local_loss(self, x):
        recon = self.reconstruct_per_neuron(self.forward(x))
        return F.mse_loss(recon, x.unsqueeze(1).expand_as(recon))


class DendriticAEClassifier(nn.Module):
    has_ae = True

    def __init__(self, in_dim, hidden, num_classes, n_dendrites=4,
                 inner_act="tanh", activation="sigmoid", freeze_hidden=False):
        super().__init__()
        assert activation == "sigmoid", "soma is sigmoid by construction, as NAN is"
        self.kind = f"dend{n_dendrites}" + ("" if inner_act == "tanh"
                                            else f"-{inner_act}")
        self.frozen = freeze_hidden
        self.hidden = DendriticAELayer(in_dim, hidden, n_dendrites, inner_act)
        self.head = nn.Linear(hidden, num_classes)
        if freeze_hidden:
            for p in self.hidden.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        return self.head(self.hidden(x))

    def local_loss(self, x):
        return self.hidden.local_loss(x)

    @torch.no_grad()
    def reconstruct(self, x):
        return self.hidden.reconstruct_per_neuron(self.hidden(x)).mean(dim=1)


# ==============================================================================
# Parameter accounting — for the matched control
# ==============================================================================
def hidden_params(kind, in_dim, H, m=1):
    """Hidden-layer parameter count, excluding the classifier head."""
    if kind == "nan":
        return H * in_dim + H * in_dim + H * in_dim + H     # enc + dec_w + dec_b
    if kind == "dend":
        return (H * m * in_dim + H * m          # encoder
                + H * m + H                     # soma
                + H * m + H * m                 # decoder input stage
                + H * m * in_dim + H * in_dim)  # decoder output
    raise ValueError(kind)


def matched_width_for_depth(m, in_dim=784, H=128):
    """NAN width with the same hidden-parameter budget as dend-m at width H.

    Run this NAN alongside every depth condition. Without it, a win from depth
    cannot be separated from a win from simply having more parameters.
    """
    target = hidden_params("dend", in_dim, H, m)
    lo, hi = 1, 100000
    while lo < hi:
        mid = (lo + hi) // 2
        if hidden_params("nan", in_dim, mid) < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


_DEND_RE = re.compile(r"^dend(\d+)(?:-(tanh|relu|sigmoid|linear))?$")


def parse_dend_variant(v):
    mm = _DEND_RE.match(v)
    return (int(mm.group(1)), mm.group(2) or "tanh") if mm else None


def build_model(variant, in_dim, hidden, num_classes, activation="sigmoid",
                **kw):
    """Handles dend* here, group* via group_ae, everything else via simple_core."""
    spec = parse_dend_variant(variant)
    if spec is None:
        return _build_group(variant, in_dim, hidden, num_classes, activation, **kw)
    m, inner = spec
    return DendriticAEClassifier(in_dim, hidden, num_classes, n_dendrites=m,
                                 inner_act=inner, activation=activation)


# ==============================================================================
# Self-test
# ==============================================================================
def _self_test():
    from models import NeuronAELayer
    torch.manual_seed(0)
    B, D, H = 32, 40, 12
    x = torch.rand(B, D)

    # --- m=1 linear should be able to FIT NAN's loss, i.e. same function class
    dend = DendriticAELayer(D, H, n_dendrites=1, inner_act="linear")
    nan = NeuronAELayer(D, H)
    opt = torch.optim.Adam(dend.parameters(), lr=0.02)
    for _ in range(400):
        opt.zero_grad(); dend.local_loss(x).backward(); opt.step()
    opt2 = torch.optim.Adam(nan.parameters(), lr=0.02)
    for _ in range(400):
        opt2.zero_grad(); nan.local_loss(x).backward(); opt2.step()
    ld, ln = dend.local_loss(x).item(), nan.local_loss(x).item()
    print(f"  m=1 linear: dend loss {ld:.5f}  vs NAN {ln:.5f}  "
          f"(rel diff {abs(ld-ln)/max(ln,1e-9):.1%})")
    assert abs(ld - ln) / max(ln, 1e-9) < 0.10, "m=1 linear should match NAN"

    # --- shapes and gradient flow at every depth and activation
    for m in (1, 2, 4, 8):
        mod = DendriticAEClassifier(D, H, 5, n_dendrites=m)
        code = mod.hidden(x)
        rec = mod.hidden.reconstruct_per_neuron(code)
        assert code.shape == (B, H) and rec.shape == (B, H, D), (code.shape, rec.shape)
        loss = F.cross_entropy(mod(x), torch.randint(0, 5, (B,))) + mod.local_loss(x)
        loss.backward()
        g = sum(p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None)
        assert torch.isfinite(loss) and g > 0
    print("  shapes (B,H) and (B,H,in) correct, gradients finite, m = 1,2,4,8")

    for act in ("tanh", "relu", "sigmoid", "linear"):
        mod = DendriticAEClassifier(D, H, 5, n_dendrites=3, inner_act=act)
        mod.local_loss(x).backward()
        print(f"    inner_act={act:8} ok")

    # --- the parameter table the depth sweep needs
    print("\n  Parameter cost at in_dim=784, H=128 (hidden layer only):")
    base = hidden_params("nan", 784, 128)
    print(f"    {'config':16}{'params':>12}{'vs NAN':>9}{'matched NAN width':>20}")
    print(f"    {'nan (m=1)':16}{base:>12,}{1.0:>8.1f}x{'128':>20}")
    for m in (2, 4, 8):
        pm = hidden_params("dend", 784, 128, m)
        print(f"    {'dend'+str(m):16}{pm:>12,}{pm/base:>8.1f}x"
              f"{matched_width_for_depth(m):>20}")
    print("\n  all checks passed")


if __name__ == "__main__":
    print("dendritic_ae self-test")
    _self_test()