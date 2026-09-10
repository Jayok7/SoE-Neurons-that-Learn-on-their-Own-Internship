"""
inhibition.py — mutual inhibition for the hidden layer.

WHY THIS FILE EXISTS
Weeks 5-10 measured collapse but never had a mechanism that acts against it,
because in those runs the classification gradient was reaching the encoder and
supplying the anti-collapse pressure implicitly. Week 10 confirmed this from
the other side: a decoder-similarity penalty swept over four orders of
magnitude changed effective rank by 0.1 units out of 128 while ~30 sat spare.

With --detach_head that implicit pressure is gone. The hidden layer is built by
reconstruction alone, and reconstruction is exactly the objective that pulls
every neuron toward the same solution — each neuron's best private
reconstruction of an anisotropic dataset is its leading principal direction,
which is the same direction for all of them. So collapse should return, and a
mechanism that acts against it becomes load-bearing rather than decorative.

THREE MECHANISMS, DELIBERATELY DIFFERENT IN KIND
They are separated because they fail differently, and knowing which one worked
is more informative than knowing that something did.

  spectral   A LOSS TERM. Differentiable spectral redundancy, 1 - effrank/H,
             computed from the activation covariance of the batch. This makes
             the metric Weeks 7-10 reported into the thing being optimised.
             Gradient-based, global (every neuron sees all the others), and the
             most direct possible attack on the number.

  kwta       AN ARCHITECTURAL CONSTRAINT. Only the k most active neurons per
             image pass their activation on; the rest emit zero. No loss term,
             no parameters. Competition is enforced by construction rather than
             encouraged by a penalty, so it cannot be traded away against
             reconstruction the way a penalty can.

             Tuned after the first detached sweep. That run used a single
             setting (k = 10% of the layer, no boosting) and gained +33 points
             on detached NAN at width 128, but left 36% of neurons dead --
             the first non-zero dead fraction anywhere in this project. Two
             changes follow from that:

             k_min      k = max(k_min, round(k_frac * H)), default 2. At width
                        8 the old rule gave k = 1, a single winner per image,
                        and that width showed the smallest gain (+14 against
                        +37 at width 512). One winner is the failure mode
                        `lateral` fell into; do not build it in by rounding.
             boost      a running duty cycle per neuron, with neurons that win
                        less than their fair share getting an additive bonus to
                        their score. This is what the dead fraction is asking
                        for: a neuron that never wins never trains, so it never
                        becomes competitive, so it never wins. Boosting breaks
                        that loop without touching the loss. Set
                        --inhib_boost 0 to recover the original behaviour.

  lateral    A LOCAL LEARNING RULE. Neurons inhibit each other through a
             recurrent matrix M, and M is grown ANTI-HEBBIANLY: pairs that fire
             together get a stronger mutual inhibition. No gradient touches M
             at all — it is updated from batch statistics under no_grad. This
             is the only one of the three that would survive the objection
             "that needs a global error signal", which matters given the whole
             point of the detached runs.

WHAT TO EXPECT, AND THE TRAP TO AVOID
Week 7 established that effective rank does not predict usefulness: the frozen
random layer had the highest rank in the entire sweep and the worst accuracy.
A mechanism that raises rank while lowering accuracy is pushing the
representation toward the frozen regime, and is a failure however good the
rank number looks. Every mechanism here must be judged on accuracy AND rank
together. `spectral` is the one most likely to fail this way, since it
optimises the rank number directly; `kwta` and `lateral` cannot chase it as
cleanly and are the more interesting results if they work.

A second trap: WTA collapses to a single winner. Your friend's spiking runs hit
this — one neuron fires, inhibits the others, receives most of the gradient,
and fires even more strongly next time. `lateral` guards against it with an
adaptive threshold (theta rises when a neuron wins, decays otherwise), and
`kwta` avoids it structurally because k winners are guaranteed. If you see
effective rank near 1 with `lateral`, raise --inhib_theta_plus first.

USE (nothing in group_ae.py or dendritic_ae.py needs editing)

    from inhibition import register_penalties, wrap_inhibition
    register_penalties()                       # adds "spectral" to PENALTIES
    model = build_model(...)
    wrap_inhibition(model, args.inhibition, strength=..., k_frac=...)

`spectral` arrives through models.PENALTIES, so it works with the existing
--penalty / --penalty_weight / --penalty_sweep machinery unchanged. `kwta` and
`lateral` are wrappers on the hidden layer and are selected with --inhibition.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Spectral redundancy as a differentiable loss
# ==============================================================================
def _activation_eigenvalues(activations, jitter=1e-6):
    """Eigenvalues of the (H, H) activation covariance, ascending, >= 0.

    eigvalsh on the H x H covariance rather than an SVD of the (B, H) matrix:
    same spectrum, but the cost is O(H^3) with H <= 512 instead of depending on
    batch size, and it is differentiable.

    The jitter is not cosmetic. The gradient of eigvalsh involves 1/(l_i - l_j)
    for every pair, so exactly-degenerate eigenvalues give infinities. A
    collapsed layer has many near-zero eigenvalues and is precisely the case
    where that bites, which is the case this penalty is meant to run in.
    """
    centred = activations - activations.mean(dim=0, keepdim=True)
    n = max(activations.size(0) - 1, 1)
    cov = (centred.t() @ centred) / n                       # (H, H)
    cov = cov + jitter * torch.eye(cov.size(0), device=cov.device,
                                   dtype=cov.dtype)
    return torch.linalg.eigvalsh(cov).clamp_min(0.0)


def spectral_redundancy(activations, eps=1e-12):
    """1 - effective_rank/H, in [0, 1). 0 = every neuron independent.

    Effective rank is exp(entropy of the normalised eigenvalue spectrum) —
    the same definition collapse_metrics.py uses for reporting, so a run
    optimising this is optimising the number the reports quote. That is the
    point, and also the reason to read accuracy alongside it.
    """
    eig = _activation_eigenvalues(activations)
    p = eig / (eig.sum() + eps)
    entropy = -(p * torch.log(p + eps)).sum()
    eff_rank = torch.exp(entropy)
    return 1.0 - eff_rank / activations.size(1)


def penalty_spectral(layer, x):
    """PENALTIES-compatible wrapper: fn(layer, x) -> scalar loss.

    Note this runs the encoder a second time. That is deliberate — it keeps the
    signature identical to penalty_decoder/penalty_activation so the existing
    sweep machinery needs no special case, and the cost is one extra forward
    through a single Linear.
    """
    return spectral_redundancy(layer(x))


def register_penalties():
    """Add 'spectral' to models.PENALTIES. Idempotent."""
    try:
        from models import PENALTIES
    except ImportError:
        from nan.models import PENALTIES
    PENALTIES.setdefault("spectral", penalty_spectral)
    return PENALTIES


# ==============================================================================
# 2. k-winner-take-all
# ==============================================================================
class KWinnerTakeAll(nn.Module):
    """Keep the k largest activations per image; zero the rest, with an
    optional duty-cycle boost so neurons that never win get a chance to.

    Wraps a hidden layer and preserves its interface (forward, local_loss, and
    the .encoder / .decoder_weight / .decoder_bias attributes the diagnostics
    read), so collapse_metrics and the galleries keep working untouched.

    The mask is applied to the layer's OUTPUT, which means local_loss sees the
    masked code too: a neuron that loses the competition on an image also stops
    being trained to reconstruct it. That is the intended coupling — it is the
    gating your friend had to add to his spiking runs, where decoders kept
    learning the dataset mean while their encoders were silent.

    Gradient: straight-through on the masked entries would let losers keep
    learning and defeat the point, so the mask is a hard multiply. Winners get
    the full gradient, losers get none.
    """

    def __init__(self, layer, k, hidden, hard_local_loss=True,
                 boost=1.0, duty_eta=0.01):
        super().__init__()
        self.layer = layer
        self.k = int(k)
        self.hard_local_loss = hard_local_loss
        self.boost = float(boost)
        self.duty_eta = float(duty_eta)
        self.target_duty = self.k / float(hidden)
        # Running fraction of images on which each neuron wins. Initialised at
        # the target so no neuron is boosted or penalised before any evidence.
        self.register_buffer("duty", torch.full((hidden,), self.target_duty))

    # -- keep the wrapped layer's attributes visible to the diagnostics ------
    @property
    def encoder(self):
        return self.layer.encoder

    @property
    def decoder_weight(self):
        return self.layer.decoder_weight

    @property
    def decoder_bias(self):
        return self.layer.decoder_bias

    @property
    def decoder(self):                      # ANN only
        return self.layer.decoder

    def _mask(self, code):
        if self.k >= code.size(1):
            return code
        if self.boost > 0:
            # Multiplicative boost on the SELECTION score only: a neuron that
            # has won none of its fair share gets its score multiplied by
            # exp(boost), one at or above its share gets 1.0. The value passed
            # on is always the neuron's real activation, so boosting can change
            # who wins but never inflate a code.
            #
            # Multiplicative rather than additive because the activations are
            # sigmoids: with k = 10% of a 128-wide layer the cut sits around
            # 0.6, and an additive bonus on the scale of the duty deficit
            # (about 0.1) is far too small to promote anything. Measured: the
            # additive version left the dead fraction unchanged at 0.89.
            rel = ((self.target_duty - self.duty) / self.target_duty).clamp(0.0, 1.0)
            score = code * torch.exp(self.boost * rel).unsqueeze(0)
        else:
            score = code
        kth = score.topk(self.k, dim=1).values[:, -1:]       # (B, 1)
        win = (score >= kth).to(code.dtype)
        # Updated on every masked call while training. Under the combined
        # schedule forward() and local_loss() each mask once per batch, so the
        # duty EMA simply moves at twice duty_eta -- harmless, and keeping it
        # unconditional means the statistic is never silently frozen by a
        # schedule that happens not to call forward().
        if self.training and self.boost > 0:
            with torch.no_grad():
                self.duty.mul_(1 - self.duty_eta).add_(
                    self.duty_eta * win.detach().mean(0))
        return code * win

    def forward(self, x):
        return self._mask(self.layer(x))

    @torch.no_grad()
    def inhibition_stats(self):
        """Duty-cycle health. frac_starved counts neurons winning on under a
        tenth of their fair share -- the population that ends up dead."""
        return {"duty_mean": float(self.duty.mean()),
                "duty_min": float(self.duty.min()),
                "duty_max": float(self.duty.max()),
                "target_duty": self.target_duty,
                "frac_starved": float((self.duty < 0.1 * self.target_duty)
                                      .float().mean()),
                "duty_gini": float(_gini(self.duty))}

    def local_loss(self, x):
        if not self.hard_local_loss:
            return self.layer.local_loss(x)
        # Recompute the per-neuron reconstruction from the MASKED code, so a
        # silenced neuron's decoder receives no gradient on this image.
        code = self._mask(self.layer(x))
        if hasattr(self.layer, "decoder"):                   # ANN
            return F.mse_loss(self.layer.decoder(code), x)
        recon = (code.unsqueeze(-1) * self.decoder_weight.unsqueeze(0)
                 + self.decoder_bias.unsqueeze(0))
        return F.mse_loss(recon, x.unsqueeze(1).expand_as(recon))


# ==============================================================================
# 3. Anti-Hebbian lateral inhibition
# ==============================================================================
class LateralInhibition(nn.Module):
    """Recurrent mutual inhibition with an anti-Hebbian, gradient-free M.

    Settling: starting from the feedforward drive u = encoder(x),

        h_0 = sigmoid(u)
        h_t = sigmoid(u - strength * (M @ h_{t-1}) - theta)

    for --inhib_steps iterations. M is off-diagonal and non-negative, so a
    neuron that is active suppresses the neurons it is coupled to.

    M is grown ANTI-HEBBIANLY under no_grad, from the batch correlation of the
    settled activations:

        M <- (1 - eta) * M + eta * relu(corr(h, h))_offdiag

    Pairs that fire together come to inhibit each other, which is exactly the
    reading your friend proposed: simultaneous firing is evidence the two
    neurons encode the same thing. Because M never receives a gradient, this
    mechanism does not depend on a global error signal at all — the property
    that makes it worth having under --detach_head.

    theta is a per-neuron adaptive threshold: it rises additively each time a
    neuron is among the most active and decays multiplicatively every step.
    Without it the whole thing degenerates to a single permanent winner, which
    is the failure mode your friend documented in the WTA spiking runs.

    Both M and theta are buffers, so they are saved with the model, move with
    .to(device), and are frozen correctly by model.eval().
    """

    def __init__(self, layer, hidden, strength=1.0, steps=3, eta=0.01,
                 theta_plus=0.05, theta_decay=0.9, active_frac=0.1):
        super().__init__()
        self.layer = layer
        self.strength = float(strength)
        self.steps = int(steps)
        self.eta = float(eta)
        self.theta_plus = float(theta_plus)
        self.theta_decay = float(theta_decay)
        self.n_active = max(1, int(round(active_frac * hidden)))
        self.register_buffer("M", torch.zeros(hidden, hidden))
        self.register_buffer("theta", torch.zeros(hidden))
        self.register_buffer("_offdiag",
                             1.0 - torch.eye(hidden).unsqueeze(0).squeeze(0))

    @property
    def encoder(self):
        return self.layer.encoder

    @property
    def decoder_weight(self):
        return self.layer.decoder_weight

    @property
    def decoder_bias(self):
        return self.layer.decoder_bias

    @property
    def decoder(self):
        return self.layer.decoder

    def _settle(self, x):
        # M and theta are CLONED snapshots. Neither is learned by gradient —
        # M is anti-Hebbian and theta is bookkeeping — so detaching them is
        # semantically correct, and taking a copy is also NECESSARY: _update
        # mutates both buffers in place after the forward pass, and autograd
        # refuses to backprop through a tensor whose version has changed since
        # it was used. .detach() alone is not enough here, because a detached
        # view SHARES the version counter of the original; only .clone() gives
        # a snapshot the in-place update cannot invalidate.
        M = self.M.detach().clone()
        theta = self.theta.detach().clone()
        u = self.layer.encoder(x)                            # (B, H)
        h = torch.sigmoid(u)
        for _ in range(self.steps):
            inhibition = h @ M.t()                           # (B, H)
            h = torch.sigmoid(u - self.strength * inhibition - theta)
        return h

    def forward(self, x):
        h = self._settle(x)
        if self.training:
            self._update(h.detach())
        return h

    @torch.no_grad()
    def _update(self, h):
        """Anti-Hebbian M update and adaptive-threshold bookkeeping."""
        # --- M: correlated pairs inhibit each other ---
        c = (h - h.mean(0, keepdim=True)) / (h.std(0, keepdim=True) + 1e-8)
        corr = (c.t() @ c) / h.size(0)                        # (H, H) in [-1,1]
        target = torch.relu(corr) * self._offdiag             # only positive
        self.M.mul_(1 - self.eta).add_(self.eta * target)

        # --- theta: winners get harder to activate, everyone else relaxes ---
        self.theta.mul_(self.theta_decay)
        k = min(self.n_active, h.size(1))
        winners = h.topk(k, dim=1).indices                    # (B, k)
        counts = torch.zeros_like(self.theta)
        counts.scatter_add_(0, winners.reshape(-1),
                            torch.ones(winners.numel(), device=h.device,
                                       dtype=self.theta.dtype))
        self.theta.add_(self.theta_plus * counts / h.size(0))

    def local_loss(self, x):
        """Reconstruction from the SETTLED code, so the decoders are trained
        against the activations the classifier actually sees."""
        code = self._settle(x)
        if hasattr(self.layer, "decoder"):                    # ANN
            return F.mse_loss(self.layer.decoder(code), x)
        recon = (code.unsqueeze(-1) * self.decoder_weight.unsqueeze(0)
                 + self.decoder_bias.unsqueeze(0))
        return F.mse_loss(recon, x.unsqueeze(1).expand_as(recon))

    @torch.no_grad()
    def inhibition_stats(self):
        """Diagnostics to save alongside the collapse report."""
        off = self.M[self._offdiag.bool()]
        return {"M_mean": float(off.mean()),
                "M_max": float(off.max()) if off.numel() else 0.0,
                "M_frac_active": float((off > 1e-3).float().mean()),
                "theta_mean": float(self.theta.mean()),
                "theta_max": float(self.theta.max()),
                "theta_gini": float(_gini(self.theta))}


def _gini(v):
    """Inequality of theta across neurons. Near 0 = competition is shared out;
    near 1 = one neuron is winning everything, i.e. the WTA failure mode."""
    v = v.flatten().sort().values
    n = v.numel()
    if n == 0 or v.sum() <= 0:
        return 0.0
    idx = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


# ==============================================================================
# 4. The wrapper entry point
# ==============================================================================
INHIBITIONS = ("none", "kwta", "lateral")


def wrap_inhibition(model, mode, strength=1.0, k_frac=0.1, steps=3,
                    eta=0.01, theta_plus=0.05, theta_decay=0.9,
                    k_min=2, boost=1.0, duty_eta=0.01, variant=None):
    """Replace model.hidden with an inhibited version of itself, in place.

    Applied to AE variants only, for the same reason detach_head is: the plain
    MLP is the baseline these runs are measured against and must stay
    untouched. Returns a dict describing what was done (for the config record),
    or None if nothing was.

    Call it AFTER build_model and AFTER apply_detach_head — detach acts on the
    head, this acts on the hidden layer, and the two do not interact.
    """
    if mode in (None, "none"):
        return None
    if mode not in INHIBITIONS:
        raise ValueError(f"unknown inhibition '{mode}'; "
                         f"choose from {INHIBITIONS}")
    hidden_layer = getattr(model, "hidden", None)
    if hidden_layer is None or not getattr(model, "has_ae", False):
        return None                       # linear / frozen / mlp: leave alone
    if getattr(model, "frozen", False):
        return None                       # nothing trains; inhibition is moot

    # Width lookup has to survive every layer type in the project.
    # DendriticAELayer has no `.encoder`, so asking for one crashed the
    # dend* variants outright.
    if hasattr(hidden_layer, "encoder"):
        width = hidden_layer.encoder.out_features
    elif hasattr(hidden_layer, "out_features"):
        width = int(hidden_layer.out_features)
    elif hasattr(hidden_layer, "decoder_weight"):
        width = int(hidden_layer.decoder_weight.shape[0])
    else:
        warnings.warn(
            f"cannot determine hidden width for "
            f"{type(hidden_layer).__name__}; inhibition not applied.",
            RuntimeWarning)
        return None
    if mode == "kwta":
        k = max(int(k_min), int(round(k_frac * width)))
        k = min(k, width)
        model.hidden = KWinnerTakeAll(hidden_layer, k, width,
                                      boost=boost, duty_eta=duty_eta)
        info = {"inhibition": "kwta", "k": k, "k_frac": k_frac,
                "k_min": int(k_min), "boost": boost, "duty_eta": duty_eta,
                "width": width}
    else:
        model.hidden = LateralInhibition(
            hidden_layer, width, strength=strength, steps=steps, eta=eta,
            theta_plus=theta_plus, theta_decay=theta_decay, active_frac=k_frac)
        info = {"inhibition": "lateral", "strength": strength, "steps": steps,
                "eta": eta, "theta_plus": theta_plus,
                "theta_decay": theta_decay, "active_frac": k_frac,
                "width": width}
    info["variant"] = variant
    return info


def collect_inhibition_stats(model):
    """None unless the model carries an inhibited hidden layer."""
    h = getattr(model, "hidden", None)
    return (h.inhibition_stats()
            if isinstance(h, (LateralInhibition, KWinnerTakeAll)) else None)


# ==============================================================================
# Self-test:  python inhibition.py
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, H = 64, 40, 16
    x = torch.rand(B, D)

    print("spectral_redundancy sanity")
    ident = torch.randn(B, H)                       # independent -> low
    clone = torch.randn(B, 1).repeat(1, H)          # all identical -> ~1
    print(f"  independent activations : {spectral_redundancy(ident):.4f}"
          "   (want near 0)")
    print(f"  cloned activations      : {spectral_redundancy(clone):.4f}"
          "   (want near 1 - 1/H = "
          f"{1 - 1 / H:.4f})")
    g = torch.randn(B, H, requires_grad=True)
    spectral_redundancy(g).backward()
    print(f"  gradient finite         : {bool(torch.isfinite(g.grad).all())}")

    try:
        from models import NeuronAELayer
    except ImportError:
        from nan.models import NeuronAELayer

    class Toy(nn.Module):
        has_ae = True
        def __init__(self):
            super().__init__()
            self.hidden = NeuronAELayer(D, H)
            self.head = nn.Linear(H, 5)
        def forward(self, xx):
            return self.head(self.hidden(xx))
        def local_loss(self, xx):
            return self.hidden.local_loss(xx)

    for mode in ("kwta", "lateral"):
        m = Toy()
        info = wrap_inhibition(m, mode, strength=1.0, k_frac=0.25)
        m.train()
        out = m(x)
        loss = m.local_loss(x) + F.cross_entropy(out, torch.zeros(B, dtype=torch.long))
        loss.backward()
        code = m.hidden(x)
        grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in m.parameters() if p.requires_grad)
        print(f"\n{mode}: {info}")
        print(f"  out shape {tuple(out.shape)}  code shape {tuple(code.shape)}")
        print(f"  frac of code exactly zero: {(code == 0).float().mean():.3f}")
        print(f"  all grads finite: {grads_ok}")
        print(f"  stats: {collect_inhibition_stats(m)}")

    print("\nregister_penalties ->", sorted(register_penalties()))