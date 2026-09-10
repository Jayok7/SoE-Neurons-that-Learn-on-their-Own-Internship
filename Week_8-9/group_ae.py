"""
group_ae.py — the NAN/ANN interpolation: neurons pooled into groups that share
a reconstruction target.

--------------------------------------------------------------------------
THE IDEA
--------------------------------------------------------------------------
NAN's reconstruction loss is a sum of H terms, each depending on exactly one
neuron's parameters:

    L_NAN = (1/H) sum_j || x - (h_j w_j + b_j) ||^2

Because the sum is separable, its global minimum has every neuron sitting at
the same optimal rank-1 autoencoder. Collapse is not a training failure in NAN;
it is the objective's optimum. ANN sits at the opposite extreme: one shared
decoder over the whole code, so neurons must divide labour, but no neuron owns
any part of the reconstruction.

This layer puts a knob between them. Neurons are assigned to G groups. Each
GROUP reconstructs the input from the pooled contributions of its members:

    recon_g = sum_j  R[j,g] * ( h_j * w_j )  +  sum_j R[j,g] * b_j
    L       = (1/G) sum_g || x - recon_g ||^2

R is the (H, G) responsibility matrix: R[j,g] is how much neuron j belongs to
group g. Within a group the loss is no longer separable, so members must
specialise relative to each other; across groups the terms stay independent, so
groups are free to differentiate.

--------------------------------------------------------------------------
WHY THIS IS THE RIGHT INTERPOLATION: BOTH ENDPOINTS ARE EXACT
--------------------------------------------------------------------------
  G = H, hard assignment. Group j contains only neuron j, so
      recon_j = h_j w_j + b_j
  which is NeuronAELayer exactly. This layer IS NAN at G = H.

  G = 1. Every neuron is in the one group, so
      recon = sum_j h_j w_j + sum_j b_j = h @ W + const
  which is LayerAELayer's Linear(H, in_dim) exactly. This layer IS ANN at G = 1.

Both are verified numerically in the self-test at the bottom of this file
(`python group_ae.py`), not merely claimed. So a sweep over G is a genuine
interpolation between the two architectures already studied, with G the only
thing changing — no new confounds.

Parameter cost is the same as NAN (H x in_dim for the decoders, H x in_dim for
the biases) plus the routing matrix (H x G), which is negligible. That matters:
a group sweep is not secretly a capacity sweep.

--------------------------------------------------------------------------
THREE WAYS TO GET R
--------------------------------------------------------------------------
routing="fixed"    R is a fixed one-hot block partition: neurons 0..k-1 in
                   group 0, k..2k-1 in group 1, and so on. Nothing is learned.
                   This is the control, and the one to run first: if a fixed
                   partition does not beat NAN, a learned one will not either,
                   and you will have found that out cheaply.

routing="learned"  R = softmax(A / tau) over groups, with A an (H, G) parameter
                   trained by ordinary backprop through the reconstruction
                   loss. Soft assignment, so a neuron can straddle groups. This
                   is mixture-of-experts routing on the reconstruction
                   objective.

routing="hebbian"  R = softmax(A / tau), but A is NOT trained by gradient. It
                   is updated under no_grad from co-activation statistics:

                       A[j,g] <- (1 - eta) * A[j,g] + eta * corr(h_j, p_g)

                   where p_g = sum_j R[j,g] h_j is the group's pooled activity.
                   Neurons that fire together with a group get pulled into it;
                   the softmax makes groups compete for them, which supplies
                   the anti-Hebbian pressure between groups that plain Hebbian
                   learning lacks. Both h and p are standardised over the batch
                   before correlating, so this is a correlation, not a raw
                   product, and it cannot run away.

                   Note what this does and does not do. It groups neurons that
                   ALREADY behave alike; the reconstruction loss then makes
                   that grouping useful, because within a group redundancy is
                   spent on a shared target rather than duplicated. It is a way
                   of exploiting the observed correlation structure, not of
                   suppressing it.

--------------------------------------------------------------------------
INTERFACE
--------------------------------------------------------------------------
GroupAELayer matches NeuronAELayer / LayerAELayer exactly: it exposes
`.encoder`, `.decoder_weight`, `.decoder_bias`, `forward(x) -> (batch, H)` and
`local_loss(x) -> scalar`. GroupAEClassifier matches simple_core.AEClassifier:
`.has_ae`, `.kind`, `.frozen`, `.hidden`, `.head`, `forward`, `local_loss`,
`reconstruct`. So every existing script, metric and plot works unchanged.

Because `decoder_weight` here really is one row per neuron (unlike ANN), the
per-neuron decoder metrics in collapse_metrics.py are meaningful for this
layer. The flag `per_neuron_decoder = True` tells collapse_metrics that.

--------------------------------------------------------------------------
USE
--------------------------------------------------------------------------
In sweep_hidden.py / collapse_analysis.py, replace

    from simple_core import build_model, evaluate, config_record, save_json
with
    from simple_core import evaluate, config_record, save_json
    from group_ae import build_model

then pass variants like:

    --variants nan ann group8 group32 group8-learn group8-hebb

`group<G>` is a fixed partition into G groups; add `-learn` or `-hebb` for the
other routings. `group1` and `group<H>` reproduce ANN and NAN respectively, so
including them is a useful internal consistency check on any sweep.

Suggested first run (H = 128, so G divides 128):

    python sweep_hidden.py --hidden_sizes 128 \
        --variants mlp nan ann group2 group4 group8 group16 group32 group64 \
        --schedule combined --seeds 3
"""

import os
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- import bootstrap (same as the other Week 7 scripts) --------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_HERE,
                   os.path.join(_HERE, os.pardir, "Week_5"),
                   os.path.join(_HERE, os.pardir, "Week_4")):
    _candidate = os.path.abspath(_candidate)
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.append(_candidate)

from simple_core import build_model as _build_base


# ==============================================================================
# The layer
# ==============================================================================
class GroupAELayer(nn.Module):
    """Hidden layer whose neurons are pooled into G groups sharing a decoder.

    Args:
        in_features:  input dimension (784 for 28x28).
        out_features: number of neurons H.
        n_groups:     G. Must divide H when routing="fixed" (the partition has
                      to be even); for the soft routings any G is allowed.
        routing:      "fixed" | "learned" | "hebbian".
        tau:          softmax temperature for the soft routings. Lower = harder
                      assignment. 1.0 is a reasonable default; 0.1 is close to
                      one-hot.
        hebb_eta:     EMA rate for the Hebbian routing update.
        hard:         straight-through top-1 assignment for the soft routings.
                      Leave on; see responsibilities() for why soft-only
                      routing collapses to uniform.
        init_logit:   how committed the soft routings start. The logits are
                      seeded with init_logit * (hard partition); 0 means start
                      from noise, which reliably collapses (see __init__).
    """

    per_neuron_decoder = True        # read by collapse_metrics._per_neuron_decoder

    def __init__(self, in_features, out_features, n_groups=8,
                 routing="fixed", tau=1.0, hebb_eta=0.01, init_logit=3.0,
                 hard=True):
        super().__init__()
        if routing not in ("fixed", "learned", "hebbian"):
            raise ValueError(f"unknown routing {routing!r}")
        if routing == "fixed" and out_features % n_groups != 0:
            divisors = [g for g in range(1, out_features + 1)
                        if out_features % g == 0]
            raise ValueError(
                f"fixed routing needs n_groups to divide hidden width: "
                f"{out_features} % {n_groups} != 0. Divisors: {divisors}")

        self.in_features = in_features
        self.out_features = out_features
        self.n_groups = n_groups
        self.routing = routing
        self.tau = tau
        self.hebb_eta = hebb_eta
        self.hard = hard

        self.encoder = nn.Linear(in_features, out_features)
        # One decoder row PER NEURON, exactly as in NeuronAELayer. Pooling
        # happens in the loss, not in the parameterisation, which is what keeps
        # the parameter count identical to NAN.
        self.decoder_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.decoder_bias = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.decoder_weight, a=5 ** 0.5)

        # An even block partition, used directly by "fixed" and as the starting
        # point for the soft routings.
        idx = torch.arange(out_features)
        block = torch.zeros(out_features, n_groups)
        block[idx, (idx * n_groups) // out_features] = 1.0

        if routing == "fixed":
            self.register_buffer("R_fixed", block)
        else:
            # IMPORTANT: start from the hard partition rather than from noise.
            # The reconstruction loss is symmetric under permuting the groups,
            # so a near-uniform R sits at a symmetric stationary point: every
            # group sees the same pooled code, every group learns the same
            # template, and gradient descent has no reason to break the tie.
            # In practice that is exactly what happens — uniform routing with
            # identical group templates, i.e. the layer quietly degenerates
            # into ANN with G redundant copies. Seeding the logits with a
            # committed partition breaks the symmetry on step one and leaves
            # the assignment free to move afterwards. `init_logit` sets how
            # committed the start is; 0 recovers the old noise-only behaviour.
            logits = (init_logit * block
                      + torch.randn(out_features, n_groups) * 0.01)
            if routing == "learned":
                self.route_logits = nn.Parameter(logits)
            else:                                    # hebbian: not a gradient parameter
                self.register_buffer("route_logits", logits)

    # ---- responsibilities -------------------------------------------------
    def responsibilities(self, hard=None):
        """(H, G) assignment matrix; rows sum to 1 for the soft routings.

        With `hard` (the default for the soft routings) a straight-through
        top-1 is applied: the forward pass uses a one-hot assignment, while the
        backward pass differentiates the softmax, so the logits still learn.

        This is not a detail. Every group reconstructs the SAME target x, so a
        neuron that spreads itself evenly over all G groups helps all of them
        at once, and soft routing therefore drifts back to uniform — at which
        point all groups share one pooled code, learn identical templates, and
        the layer has silently become ANN with G redundant copies. Measured on
        Fashion-MNIST: assignment entropy 0.95 of maximum and group-template
        overlap 0.92 after three epochs of purely soft routing.

        Forcing disjoint membership is what makes a group a group. Groups
        differentiate because their members are different, not because the loss
        rewards differentiation — it does not.
        """
        if self.routing == "fixed":
            return self.R_fixed
        hard = self.hard if hard is None else hard
        soft = F.softmax(self.route_logits / self.tau, dim=1)
        if not hard:
            return soft
        onehot = torch.zeros_like(soft).scatter_(
            1, soft.argmax(dim=1, keepdim=True), 1.0)
        return onehot + soft - soft.detach()      # straight-through

    def forward(self, x):
        return torch.sigmoid(self.encoder(x))

    # ---- the pooled reconstruction ----------------------------------------
    def group_reconstructions(self, code, R):
        """(batch, G, in_features): each group's pooled reconstruction.

            recon[b,g,:] = sum_j R[j,g] * code[b,j] * w[j,:]
                         + sum_j R[j,g] * bias[j,:]

        The weighted-code einsum is what couples group members: they share one
        target, so the only way to lower the loss is to cover different parts
        of it.
        """
        weighted = torch.einsum("bj,jg->bgj", code, R)          # (B, G, H)
        recon = torch.einsum("bgj,jd->bgd", weighted, self.decoder_weight)
        recon = recon + torch.einsum("jg,jd->gd", R, self.decoder_bias)
        return recon

    def local_loss(self, x):
        code = self.forward(x)
        R = self.responsibilities()
        recon = self.group_reconstructions(code, R)             # (B, G, in)
        loss = F.mse_loss(recon, x.unsqueeze(1).expand_as(recon))
        if self.routing == "hebbian":
            self._hebbian_update(code.detach(), R.detach())
        return loss

    # ---- Hebbian routing update -------------------------------------------
    @torch.no_grad()
    def _hebbian_update(self, code, R):
        """Pull each neuron toward the groups whose pooled activity it tracks.

        Both the neuron activations and the group activities are standardised
        over the batch first, so the update is a correlation in [-1, 1] rather
        than an unbounded product. The softmax in responsibilities() then makes
        groups compete for each neuron, which is the between-group repulsion
        that plain Hebbian learning does not provide on its own.
        """
        if not self.training:
            return
        pooled = code @ R                                       # (B, G)
        c = (code - code.mean(0, keepdim=True)) / (code.std(0, keepdim=True) + 1e-8)
        p = (pooled - pooled.mean(0, keepdim=True)) / (pooled.std(0, keepdim=True) + 1e-8)
        corr = (c.t() @ p) / code.size(0)                       # (H, G) in [-1,1]
        self.route_logits.mul_(1 - self.hebb_eta).add_(self.hebb_eta * corr)

    # ---- diagnostics -------------------------------------------------------
    @torch.no_grad()
    def group_stats(self):
        """How the groups are being used. Returns a JSON-friendly dict.

        occupancy       effective number of neurons assigned to each group
                        (column sums of R).
        assign_entropy  mean over neurons of the entropy of their assignment,
                        normalised to [0,1]. 0 = every neuron committed to one
                        group, 1 = every neuron spread evenly over all groups.
                        For fixed routing this is 0 by construction.
        used_groups     effective number of groups actually in use, from the
                        entropy of the occupancy distribution. If this falls
                        well below G, the routing has itself collapsed and the
                        layer has quietly reverted toward ANN.
        mean_abs_cos_group
                        mean |cosine| between the pooled group templates
                        (R^T @ decoder_weight). This is the group-level analogue
                        of the per-neuron decoder overlap: it asks whether the
                        GROUPS have differentiated, which is the whole point of
                        the architecture.
        """
        R = self.responsibilities(hard=False)   # report the soft assignment
        occ = R.sum(0)
        pr = R / R.sum(1, keepdim=True).clamp_min(1e-12)
        ent = -(pr * pr.clamp_min(1e-12).log()).sum(1).mean()
        norm = torch.log(torch.tensor(float(self.n_groups)))
        q = occ / occ.sum().clamp_min(1e-12)
        used = torch.exp(-(q * q.clamp_min(1e-12).log()).sum())

        R_eff = self.responsibilities()         # what the loss actually uses
        templates = R_eff.t() @ self.decoder_weight             # (G, in)
        u = F.normalize(templates, dim=1)
        cos = (u @ u.t()).clamp(-1, 1)
        if self.n_groups > 1:
            off = cos[~torch.eye(self.n_groups, dtype=torch.bool,
                                 device=cos.device)]
            mean_cos = off.abs().mean().item()
        else:
            mean_cos = 0.0

        return {"n_groups": int(self.n_groups),
                "routing": self.routing,
                "occupancy": occ.tolist(),
                "assign_entropy": float(ent / norm) if self.n_groups > 1 else 0.0,
                "used_groups": float(used),
                "mean_abs_cos_group": mean_cos}


# ==============================================================================
# Classifier wrapper — matches simple_core.AEClassifier
# ==============================================================================
class GroupAEClassifier(nn.Module):
    has_ae = True

    def __init__(self, in_dim, hidden, num_classes, n_groups=8,
                 routing="fixed", tau=1.0, hebb_eta=0.01, init_logit=3.0,
                 hard=True, activation="sigmoid", freeze_hidden=False):
        super().__init__()
        assert activation == "sigmoid", \
            "group AE layers are sigmoid by construction, as NAN/ANN are"
        self.kind = f"group{n_groups}" + ("" if routing == "fixed"
                                          else f"-{routing}")
        self.frozen = freeze_hidden
        self.hidden = GroupAELayer(in_dim, hidden, n_groups, routing,
                                   tau, hebb_eta, init_logit, hard)
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
        """Image-space reconstruction for display: the mean over groups."""
        code = self.hidden(x)
        R = self.hidden.responsibilities()
        return self.hidden.group_reconstructions(code, R).mean(dim=1)


# ==============================================================================
# build_model — drop-in replacement for simple_core.build_model
# ==============================================================================
_GROUP_RE = re.compile(r"^group(\d+)(?:-(learn|hebb|fixed))?$")

_ROUTING = {"learn": "learned", "hebb": "hebbian",
            "fixed": "fixed", None: "fixed"}


def parse_group_variant(variant):
    """'group8-learn' -> (8, 'learned'); returns None if not a group variant."""
    m = _GROUP_RE.match(variant)
    if not m:
        return None
    return int(m.group(1)), _ROUTING[m.group(2)]


def build_model(variant, in_dim, hidden, num_classes, activation="sigmoid",
                tau=1.0, hebb_eta=0.01, init_logit=3.0, hard=True,
                n_steps=20, residual=False):
    """Handles group and spiking variants; defers everything else to simple_core.

    Spiking variants (require snntorch):
        snn_nan       time-to-first-spike NAN, each neuron scored independently
        snn_nan-res   the same, but each neuron scored against the residual
                      left by the neurons that fired before it
    """
    if variant.startswith("snn_nan"):
        from spiking_nan import SpikingNANClassifier
        return SpikingNANClassifier(in_dim, hidden, num_classes,
                                    n_steps=n_steps,
                                    residual=variant.endswith("-res"))
    spec = parse_group_variant(variant)
    if spec is None:
        return _build_base(variant, in_dim, hidden, num_classes, activation)
    n_groups, routing = spec
    return GroupAEClassifier(in_dim, hidden, num_classes, n_groups=n_groups,
                             routing=routing, tau=tau, hebb_eta=hebb_eta,
                             init_logit=init_logit, hard=hard,
                             activation=activation)


def collect_group_stats(model):
    """group_stats() for a group model, else None. Safe to call on anything."""
    hidden = getattr(model, "hidden", None)
    return hidden.group_stats() if isinstance(hidden, GroupAELayer) else None


# ==============================================================================
# Self-test: prove the endpoints are exact, not approximate
# ==============================================================================
def _self_test():
    from models import NeuronAELayer, LayerAELayer
    torch.manual_seed(0)
    B, D, H = 7, 23, 12
    x = torch.rand(B, D)

    # --- G = H must equal NeuronAELayer (NAN) ---
    g = GroupAELayer(D, H, n_groups=H, routing="fixed")
    n = NeuronAELayer(D, H)
    n.encoder.load_state_dict(g.encoder.state_dict())
    with torch.no_grad():
        n.decoder_weight.copy_(g.decoder_weight)
        n.decoder_bias.copy_(g.decoder_bias)
    d_nan = (g.local_loss(x) - n.local_loss(x)).abs().item()

    # --- G = 1 must equal LayerAELayer (ANN) ---
    g1 = GroupAELayer(D, H, n_groups=1, routing="fixed")
    a = LayerAELayer(D, H)
    a.encoder.load_state_dict(g1.encoder.state_dict())
    with torch.no_grad():
        a.decoder.weight.copy_(g1.decoder_weight.t())      # (in, H) <- (H, in)
        a.decoder.bias.copy_(g1.decoder_bias.sum(0))       # pooled bias
    d_ann = (g1.local_loss(x) - a.local_loss(x)).abs().item()

    print(f"  |L_group(G=H) - L_NAN| = {d_nan:.3e}")
    print(f"  |L_group(G=1) - L_ANN| = {d_ann:.3e}")
    assert d_nan < 1e-6, "G=H should reproduce NAN exactly"
    assert d_ann < 1e-6, "G=1 should reproduce ANN exactly"

    # --- parameter count must match NAN (routing is negligible) ---
    p_group = sum(p.numel() for p in GroupAELayer(D, H, 4).parameters())
    p_nan = sum(p.numel() for p in NeuronAELayer(D, H).parameters())
    print(f"  params: group(G=4)={p_group}  nan={p_nan}  (equal: {p_group == p_nan})")

    # --- all three routings train and produce finite gradients ---
    for routing in ("fixed", "learned", "hebbian"):
        m = GroupAEClassifier(D, H, 5, n_groups=4, routing=routing)
        m.train()
        loss = F.cross_entropy(m(x), torch.randint(0, 5, (B,))) + m.local_loss(x)
        loss.backward()
        gsum = sum(p.grad.abs().sum().item() for p in m.parameters()
                   if p.grad is not None)
        st = m.hidden.group_stats()
        print(f"  routing={routing:8} loss={loss.item():.4f} grad={gsum:.3e} "
              f"used_groups={st['used_groups']:.2f} "
              f"assign_entropy={st['assign_entropy']:.3f}")
        assert torch.isfinite(loss) and gsum > 0

    print("  all checks passed")


if __name__ == "__main__":
    print("group_ae self-test")
    _self_test()
