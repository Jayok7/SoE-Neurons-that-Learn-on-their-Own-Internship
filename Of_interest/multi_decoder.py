"""
multi_decoder.py -- several full decoders reading the whole hidden layer.

THE IDEA
NAN gives each neuron its own decoder reading only that neuron. ANN gives the
whole layer one decoder reading everything. This is the other corner: G
decoders, each reading EVERY neuron. Each one is a complete autoencoder in its
own right, so the layer carries an ensemble of ANNs rather than a partition.

    NAN      neuron i -> decoder i             (decoder sees 1 neuron)
    ANN      all neurons -> one decoder        (1 decoder sees all)
    ring     neuron i -> decoders i..i+o       (decoders share a weight bank)
    multi    all neurons -> each of G decoders (G independent weight banks)

WHY IT MIGHT WORK, AND WHY IT MIGHT NOT
Left alone it should do nothing. Every decoder faces the same least-squares
problem from the same code, so all G converge to the same solution and the
layer is ANN with redundant copies. That is the null result to expect, and it
is worth measuring precisely because it is the baseline the interesting case
has to beat.

The interesting case is with a diversity pressure between the decoders. If two
decoders are forced to be dissimilar and BOTH must reconstruct the input from
the same code, the code has to be rich enough that two different linear maps
can each recover the input. That is a stronger demand on the representation
than one map recovering it, and it is a demand that has no analogue in NAN --
where the pressure has to be added artificially because nothing in the
architecture supplies it.

So the experiment is really about the penalty, not the architecture. `multi<G>`
alone is the control; `multi<G>` with --penalty ensemble is the test.

COST
Each decoder is a full (hidden, in_dim) matrix, so parameters scale linearly in
G and this is the most expensive variant in the project. At H=128 and 784
inputs, G=4 costs about 402k hidden parameters against NAN's 301k, and G=8
costs 803k. Always run `multi1` alongside: it is one full decoder, which is
ANN with a per-pixel bias, and it is the honest control for everything else.

VARIANTS
    multi<G>            e.g. multi1, multi2, multi4, multi8
    penalty "ensemble"  mean squared cosine between the G decoders, so the
                        pressure acts between ensemble members rather than
                        between neurons

USAGE
    python sweep_hidden.py --variants ann multi1 multi2 multi4 multi8 \\
        --hidden_sizes 128 --detach_head
    python sweep_hidden.py --variants multi4 --penalty ensemble \\
        --penalty_sweep 0 0.01 0.1 1 10 --detach_head
"""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models import PENALTIES
except ImportError:
    from nan.models import PENALTIES

from dendritic_ae import build_model as _build_prev


_MULTI_RE = re.compile(r"^multi(\d+)$")


def parse_multi_variant(variant):
    m = _MULTI_RE.match(variant)
    return int(m.group(1)) if m else None


class MultiDecoderAELayer(nn.Module):
    """One encoder, G independent decoders, each reading the whole code.

    Exposes `decoder_weight` as the per-neuron view the diagnostics expect --
    the ensemble mean over decoders -- so collapse_metrics and the decoder
    overlap split keep working without a special case. `ensemble_weight` is the
    real (G, hidden, in) tensor and is what the ensemble penalty acts on.
    """

    def __init__(self, in_features, out_features, n_decoders=4):
        super().__init__()
        self.encoder = nn.Linear(in_features, out_features)
        self.n_decoders = int(n_decoders)
        self.ensemble_weight = nn.Parameter(
            torch.empty(self.n_decoders, out_features, in_features))
        self.ensemble_bias = nn.Parameter(
            torch.zeros(self.n_decoders, in_features))
        for g in range(self.n_decoders):
            nn.init.kaiming_uniform_(self.ensemble_weight[g], a=5 ** 0.5)

    def forward(self, x):
        return torch.sigmoid(self.encoder(x))

    # -- views the shared diagnostics expect ---------------------------------
    @property
    def decoder_weight(self):
        # Per-neuron outgoing vector, averaged over the ensemble. This is what
        # mean_abs_cos_dec and the within/between split read; averaging is the
        # honest summary since every decoder genuinely reads every neuron.
        return self.ensemble_weight.mean(dim=0)

    @property
    def decoder_bias(self):
        return self.ensemble_bias.mean(dim=0).unsqueeze(0).expand(
            self.ensemble_weight.shape[1], -1)

    def reconstructions(self, code):
        """(batch, G, in_dim) -- what each decoder makes of the code."""
        return torch.einsum("bh,ghd->bgd", code, self.ensemble_weight) \
            + self.ensemble_bias.unsqueeze(0)

    def local_loss(self, x):
        recon = self.reconstructions(torch.sigmoid(self.encoder(x)))
        return F.mse_loss(recon, x.unsqueeze(1).expand_as(recon))

    @torch.no_grad()
    def ensemble_stats(self):
        """How much the ensemble members actually differ."""
        g = self.ensemble_weight.shape[0]
        flat = F.normalize(self.ensemble_weight.reshape(g, -1), dim=1)
        cos = (flat @ flat.t()) ** 2
        off = ~torch.eye(g, dtype=torch.bool, device=cos.device)
        return {"n_decoders": g,
                "ensemble_cos2_mean": float(cos[off].mean()) if g > 1 else None,
                "ensemble_cos2_max": float(cos[off].max()) if g > 1 else None}


def penalty_ensemble(layer, x=None):
    """Mean squared cosine between whole decoders. Zero for a single decoder."""
    w = getattr(layer, "ensemble_weight", None)
    if w is None or w.shape[0] < 2:
        return torch.zeros((), device=layer.encoder.weight.device)
    g = w.shape[0]
    flat = F.normalize(w.reshape(g, -1), dim=1)
    cos = (flat @ flat.t()) ** 2
    off = ~torch.eye(g, dtype=torch.bool, device=cos.device)
    return cos[off].mean()


PENALTIES.setdefault("ensemble", penalty_ensemble)


class MultiDecoderClassifier(nn.Module):
    has_ae = True

    def __init__(self, in_dim, hidden, num_classes, n_decoders=4,
                 activation="sigmoid", freeze_hidden=False, detach_head=False):
        super().__init__()
        assert activation == "sigmoid", \
            "multi-decoder layers are sigmoid by construction, as NAN/ANN are"
        self.kind = f"multi{n_decoders}"
        self.frozen = freeze_hidden
        self.detach_head = detach_head
        self.hidden = MultiDecoderAELayer(in_dim, hidden, n_decoders)
        self.head = nn.Linear(hidden, num_classes)
        if freeze_hidden:
            for p in self.hidden.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        code = self.hidden(x)
        return self.head(code.detach() if self.detach_head else code)

    def local_loss(self, x):
        return self.hidden.local_loss(x)

    @torch.no_grad()
    def reconstruct(self, x):
        return self.hidden.reconstructions(self.hidden(x)).mean(dim=1)


def collect_multi_stats(model):
    h = getattr(model, "hidden", None)
    return h.ensemble_stats() if isinstance(h, MultiDecoderAELayer) else None


def build_model(variant, in_dim, hidden, num_classes, activation="sigmoid",
                **kw):
    g = parse_multi_variant(variant)
    if g is None:
        return _build_prev(variant, in_dim, hidden, num_classes, activation, **kw)
    return MultiDecoderClassifier(in_dim, hidden, num_classes, n_decoders=g,
                                  activation=activation)


def hidden_params(variant, in_dim=784, hidden=128):
    g = parse_multi_variant(variant)
    if g is None:
        return None
    return (in_dim * hidden + hidden          # encoder
            + g * hidden * in_dim             # decoders
            + g * in_dim)                     # decoder biases


if __name__ == "__main__":
    torch.manual_seed(0)
    D, H, C, B = 200, 32, 10, 64
    x = torch.rand(B, D)

    print("multi_decoder self-test")
    for v in ("multi1", "multi2", "multi4"):
        m = build_model(v, D, H, C)
        r = m.hidden.reconstructions(m.hidden(x))
        print(f"  {v:8} recon {tuple(r.shape)}  loss {m.local_loss(x):.5f}  "
              f"params {sum(p.numel() for p in m.hidden.parameters()):,}")

    m = build_model("multi4", D, H, C)
    print(f"  penalty on fresh init : {penalty_ensemble(m.hidden):.5f}")
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    for _ in range(120):
        opt.zero_grad()
        (m.local_loss(x) + 5.0 * penalty_ensemble(m.hidden)).backward()
        opt.step()
    print(f"  penalty after training: {penalty_ensemble(m.hidden):.5f}  "
          f"(should fall)")
    print(f"  stats: {collect_multi_stats(m)}")

    m2 = build_model("multi4", D, H, C)
    opt = torch.optim.Adam(m2.parameters(), lr=0.02)
    for _ in range(120):
        opt.zero_grad(); m2.local_loss(x).backward(); opt.step()
    print(f"  unpenalised after training: {penalty_ensemble(m2.hidden):.5f}  "
          f"(should be HIGH -- decoders converge to the same solution)")

    print(f"  fallback to nan works : "
          f"{type(build_model('nan', D, H, C)).__name__}")
    print(f"  'ensemble' registered : {'ensemble' in PENALTIES}")
