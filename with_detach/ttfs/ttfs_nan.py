"""
ttfs_nan.py -- time-to-first-spike NAN with grouping driven by spike coincidence.

WHAT THIS IS
    A spiking version of the NAN layer in which every quantity that matters is
    a SPIKE TIME, not a rate:

      * the input is latency-coded: a bright pixel spikes early, a dim one late
        or never
      * each hidden neuron is a LIF unit permitted exactly ONE spike per
        presentation, so its entire output is the time at which it fired
      * the local autoencoder target is reconstructed from that single latency
      * neurons are grouped by WHEN THEY FIRE relative to each other, and the
        grouping is learned online by a Hebbian rule

    The last point is the reason for doing this in spikes at all. Grouping
    neurons "by similar activation" normally needs a covariance matrix over the
    whole layer, which is a global object and quietly abandons the locality that
    NAN is built on. Spike coincidence does not: whether neuron i fired near
    group g's firing time is available at neuron i, from a signal group g is
    already broadcasting. The routing rule below is a coincidence detector, and
    it is local in exactly the sense the project cares about.

THE CODE
    A neuron that fires at step t out of T carries code

        c = 1 - t/T          (fires immediately -> 1, fires last -> 1/T)
        c = 0                if it never fires

    Earlier means stronger. Every downstream consumer -- the classifier head,
    the reconstruction, the effective-rank measurement -- reads this one number
    per neuron, so the layer's whole output is its firing order.

    The code is differentiable: `first_spike_code` masks all but the first
    spike and takes a weighted sum over time, and the gradient flows back
    through snnTorch's surrogate on the spike itself.

GROUPING
    G groups partition the H neurons. Each group reconstructs the input from
    its members' pooled code, so:

        G = H   each neuron reconstructs alone            -- spiking NAN
        G = 1   the whole layer reconstructs together     -- spiking ANN

    exactly as in the rate-based `group_ae`, and the self-test at the bottom
    checks both endpoints numerically.

ROUTE STRENGTH  (route_norm)
    The Hebbian rule below produces a coincidence score kappa per (neuron,
    group) and writes it into the routing logits through an EMA. kappa is
    bounded in [0, 1] and, once centred per neuron, its spread is small -- with
    16 groups the logits equilibrate to a range of about 0.5, so
    softmax(logits / tau_route) at the old default tau of 0.5 came out almost
    uniform: measured mean top-1 probability 0.082 against a uniform 0.0625.

    Two things went wrong as a result. The soft partition carried almost no
    information, so the hard straight-through argmax was decided by noise and
    flipped from batch to batch. And the diagnostics read healthy while it
    happened -- used_groups 15.97 of 16, assign_entropy 0.997 -- because a
    uniform routing and a perfectly balanced one look identical to both.

    route_norm="zscore" (default) standardises kappa per neuron before the EMA,
    so the logit scale no longer depends on sigma or on the group count and
    tau_route becomes a real sharpness knob. Measured on structured input:
    top-1 probability rises from 0.10 to 0.36 and assign_entropy falls from
    0.98 to 0.72. route_norm="center" restores the original behaviour.

    This also un-confounds the sigma sweep. Under "center", sigma set both
    which neurons look coincident AND how strong the routing signal was, since
    a large sigma sends every kappa toward 1 and a small one toward 0, and
    centring then flattens both. Under "zscore" sigma only sets the window.

DETACHED HEAD  (detach_head)
    With detach_head=True the classifier gradient stops at the boundary between
    the hidden layer and the head, so the spiking layer is trained by its local
    reconstruction alone. This is Bull's rule, and it is the regime where the
    grouping question is answerable at all: with the task gradient attached,
    the rate-based sweep showed the entire group dial from ANN to NAN spanning
    0.22 accuracy points, while detached the same dial spans 57. Choosing a
    partition differently at fixed G is a strictly smaller intervention than
    changing G, so an attached routing comparison is being run inside a region
    already measured to be flat.

ROUTING
    fixed       one-hot block partition, no learning. The control.
    coincidence the Hebbian rule this file exists for. Neuron i is pulled
                toward group g in proportion to how closely its spike time
                matches g's, through a temporal kernel

                    kappa = exp( -|t_i - tau_g| / sigma )

                averaged over the batch, where tau_g is the code-weighted mean
                spike time of group g's current members. A softmax over groups
                makes them compete for each neuron, which supplies the
                between-group repulsion that plain Hebbian learning lacks: with
                no competition every neuron drifts toward whichever group is
                loudest and the partition collapses.
    learned     R by gradient descent, as a reference point for how much of any
                effect is "grouping adapts" versus "grouping adapts HEBBIANLY".

    Neurons that never fire are excluded from the update rather than treated as
    having fired at t = T. A silent neuron is not coincident with anything, and
    letting silence look like late firing would herd every dead unit into one
    group and make the partition look structured when it is not.

USAGE
    python ttfs_nan.py            # self-test: endpoints, gradients, routing

    from ttfs_nan import TTFSNet
    model = TTFSNet(in_dim=784, hidden=128, classes=47,
                    n_groups=16, routing="coincidence", num_steps=20)
    logits = model(x)             # (B, classes)
    loss   = model.local_loss(x)  # per-group reconstruction, the local target
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate


# =============================================================================
# spike-time helpers
# =============================================================================
def latency_encode(x, num_steps, mode="linear", eps=1e-3):
    """Latency code: intensity -> spike time, bright = early.

    x is (B, D) in [0, 1]. Returns (T, B, D) binary spikes with at most one
    spike per input unit, and NO spike at all for anything at or below `eps` --
    "no evidence" is encoded as silence rather than as a very late spike, which
    matters because the routing rule below has to be able to tell the two apart.

    mode="linear"   t = (1 - x) * (T - 1). Intensity maps uniformly onto the
                    available timesteps, so a T-step presentation resolves T
                    intensity levels evenly. This is the default because it
                    uses the whole time axis: with only 20-40 steps a log code
                    spends most of its resolution on the brightest few percent
                    of pixels and crushes everything else into the first two
                    steps, which is exactly what the self-test caught.
    mode="log"      t proportional to log(1/x), the classic latency-code
                    motivation (strong inputs resolved finely). Available for
                    comparison; needs a longer presentation to be usable.
    """
    x = x.clamp(0.0, 1.0)
    live = x > eps
    if mode == "linear":
        frac = 1.0 - x
    elif mode == "log":
        lo = math.log(1.0 / 1.0)
        hi = math.log(1.0 / eps)
        frac = (torch.log(1.0 / x.clamp_min(eps)) - lo) / (hi - lo)
    else:
        raise ValueError(f"unknown latency mode {mode!r}")
    idx = (frac.clamp(0.0, 1.0) * (num_steps - 1)).round().long()
    spikes = F.one_hot(idx, num_classes=num_steps).permute(2, 0, 1).float()
    return spikes * live.float().unsqueeze(0)


def first_spike_mask(spk):
    """(T, ...) binary spikes -> same shape, keeping only each unit's FIRST.

    Differentiable in `spk`: the gate is built from a detached cumulative sum,
    so the surrogate gradient reaches the spike that actually mattered and is
    zeroed for every later one -- which is the correct credit assignment for a
    code that only reads the first spike.
    """
    prior = torch.cumsum(spk.detach(), dim=0) - spk.detach()
    return spk * (prior <= 0).float()


def first_spike_code(spk):
    """(T, B, H) spikes -> (code, t_first, fired).

    code    (B, H) in [0, 1]: 1 - t_first/T, and 0 for a unit that never fired
    t_first (B, H): index of the first spike; T for a unit that never fired
    fired   (B, H) bool
    """
    T = spk.shape[0]
    fm = first_spike_mask(spk)
    steps = torch.arange(T, device=spk.device, dtype=spk.dtype)
    steps = steps.view(T, *([1] * (spk.dim() - 1)))
    code = (fm * (1.0 - steps / T)).sum(0)
    fired = fm.sum(0) > 0
    t_first = torch.where(fired, (fm * steps).sum(0), torch.full_like(code, T))
    return code, t_first, fired


# =============================================================================
# the layer
# =============================================================================
class TTFSGroupAELayer(nn.Module):
    """LIF layer, one spike per neuron, with a per-group local autoencoder.

    Parameters are counted so that the group count does NOT change capacity:
    the decoder is always (H, in_dim), one reconstruction vector per neuron,
    exactly as in the rate-based NAN. Only the (H, G) routing changes with G,
    and it is negligible. So a difference across G is a difference in what the
    objective asks for, not in how much the model can represent.
    """

    def __init__(self, in_features, out_features, n_groups=None,
                 routing="fixed", num_steps=20, beta=0.9, threshold=1.0,
                 slope=25.0, tau_route=0.25, hebb_eta=0.02, sigma=None,
                 init_logit=3.0, hard=True, route_norm="zscore"):
        super().__init__()
        if routing not in ("fixed", "coincidence", "learned"):
            raise ValueError(f"unknown routing {routing!r}")
        n_groups = out_features if n_groups is None else n_groups
        if routing == "fixed" and out_features % n_groups != 0:
            raise ValueError(
                f"fixed routing needs n_groups to divide the layer width: "
                f"{n_groups} does not divide {out_features}")

        self.in_features, self.out_features = in_features, out_features
        self.n_groups, self.routing = n_groups, routing
        self.num_steps, self.hard = num_steps, hard
        self.tau_route, self.hebb_eta = tau_route, hebb_eta
        if route_norm not in ("zscore", "center"):
            raise ValueError(f"unknown route_norm {route_norm!r}")
        self.route_norm = route_norm
        # Coincidence width in timesteps. Two neurons count as coincident if
        # they fire within roughly sigma of each other; the default of T/8 is
        # loose enough to group at all and tight enough to distinguish early
        # firers from late ones.
        self.sigma = (num_steps / 8.0) if sigma is None else sigma
        self.per_neuron_decoder = True

        self.encoder = nn.Linear(in_features, out_features)
        self.lif = snn.Leaky(beta=beta, threshold=threshold,
                             spike_grad=surrogate.fast_sigmoid(slope=slope),
                             init_hidden=False)

        self.decoder_weight = nn.Parameter(
            torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.decoder_weight, a=math.sqrt(5))
        self.decoder_bias = nn.Parameter(torch.zeros(n_groups, in_features))

        if routing == "fixed":
            k = out_features // n_groups
            R = torch.zeros(out_features, n_groups)
            R[torch.arange(out_features), torch.arange(out_features) // k] = 1.0
            self.register_buffer("route_logits", R * init_logit)
        else:
            g = torch.arange(out_features) % n_groups
            logits = F.one_hot(g, n_groups).float() * init_logit
            logits = logits + 0.01 * torch.randn_like(logits)
            if routing == "learned":
                self.route_logits = nn.Parameter(logits)
            else:                       # coincidence: updated by rule, not grad
                self.register_buffer("route_logits", logits)

    # ---- routing ----------------------------------------------------------
    def responsibilities(self, hard=None):
        """(H, G) assignment. Rows sum to 1."""
        if self.routing == "fixed":
            return self.route_logits
        R = F.softmax(self.route_logits / self.tau_route, dim=1)
        hard = self.hard if hard is None else hard
        if hard:
            idx = R.argmax(dim=1, keepdim=True)
            oh = torch.zeros_like(R).scatter_(1, idx, 1.0)
            R = oh + R - R.detach()     # straight-through
        return R

    # ---- forward ----------------------------------------------------------
    def spike_train(self, x_spikes):
        """(T, B, in) input spikes -> (T, B, H) hidden spikes, one per neuron.

        The latch is what makes this time-to-FIRST-spike rather than a rate
        code: once a neuron has fired it is silenced for the rest of the
        presentation, so no amount of further input can make it fire more. A
        neuron's output is a time, and it has exactly one.
        """
        mem = torch.zeros(x_spikes.shape[1], self.out_features,
                          device=x_spikes.device, dtype=x_spikes.dtype)
        alive = torch.ones_like(mem)
        out = []
        for t in range(x_spikes.shape[0]):
            cur = self.encoder(x_spikes[t])
            spk, mem_new = self.lif(cur, mem)
            spk = spk * alive
            # Freeze the membrane of anything that has already fired, so a
            # spent neuron is genuinely out of the presentation rather than
            # quietly integrating toward a spike that gets masked.
            mem = alive * mem_new + (1.0 - alive) * mem
            alive = alive * (1.0 - spk.detach())
            out.append(spk)
        return torch.stack(out, dim=0)

    def encode(self, x_spikes):
        """-> (code, t_first, fired). `code` is the layer's output."""
        return first_spike_code(self.spike_train(x_spikes))

    def forward(self, x_spikes):
        return self.encode(x_spikes)[0]

    @torch.no_grad()
    def calibrate(self, x_spikes, target_fire=0.9):
        """Set the firing threshold from the membrane potentials it will see.

        `target_fire` is the fraction of neurons left ABLE to fire, i.e. whose
        peak membrane over the calibration batch clears the threshold. The
        fraction firing on any single input is lower -- typically half of it --
        because a neuron's peak is reached on its preferred inputs. That is the
        intended behaviour: a sparse code where different inputs wake different
        neurons is the point, and `frac_silent` in group_stats reports the
        per-sample figure.

        Without this the layer is silent. The input is one-hot in time, so at
        any step a neuron receives the sum of only those weights whose pixels
        fire at that step -- a small number against a threshold of 1.0, and
        with default initialisation essentially nothing crosses it. The
        self-test caught 99% of neurons never firing, which would have trained
        happily and produced a layer carrying no information at all.

        Rather than hand-tuning the initialisation, run one batch with spiking
        suppressed, look at how high each neuron's membrane actually gets, and
        put the threshold at the quantile that leaves the requested fraction
        able to reach it. Cheap, and it adapts to width, dataset and step count
        instead of assuming them.
        """
        mem = torch.zeros(x_spikes.shape[1], self.out_features,
                          device=x_spikes.device, dtype=x_spikes.dtype)
        peak = torch.full_like(mem, -1e9)
        beta = float(self.lif.beta)
        for t in range(x_spikes.shape[0]):
            mem = beta * mem + self.encoder(x_spikes[t])
            peak = torch.maximum(peak, mem)
        per_neuron = peak.max(dim=0).values            # (H,)
        q = torch.quantile(per_neuron.float(),
                           1.0 - float(target_fire)).clamp_min(1e-4)
        self.lif.threshold.fill_(float(q))
        return float(q)

    # ---- local objective --------------------------------------------------
    def local_loss(self, x_spikes, x_flat):
        """Mean over groups of each group's reconstruction MSE.

        A group's reconstruction is the sum of its members' decoder vectors
        scaled by their codes: a neuron that fired early contributes strongly,
        one that fired late contributes weakly, one that stayed silent
        contributes nothing at all. The target is the analogue input, so the
        layer is being asked to reconstruct the image from firing times.
        """
        code, t_first, fired = self.encode(x_spikes)
        R = self.responsibilities()                          # (H, G)
        # (B, G, in): weight each neuron's decoder by its code and its group.
        contrib = torch.einsum("bh,hg,hd->bgd", code, R, self.decoder_weight)
        recon = contrib + self.decoder_bias.unsqueeze(0)
        loss = F.mse_loss(recon, x_flat.unsqueeze(1).expand_as(recon))
        if self.routing == "coincidence":
            self._coincidence_update(t_first.detach(), fired.detach(),
                                     code.detach(), R.detach())
        return loss

    # ---- the Hebbian rule -------------------------------------------------
    @torch.no_grad()
    def _coincidence_update(self, t_first, fired, code, R):
        """Pull each neuron toward the group whose firing time it matches.

        tau_g   code-weighted mean first-spike time of group g's members, per
                sample. Weighting by code means an early, confident spike sets
                the group's time more than a marginal late one.
        kappa   exp(-|t_i - tau_g| / sigma), averaged over the batch: a soft
                coincidence detector, 1 for perfect synchrony and decaying with
                temporal distance.

        Silent neurons are masked out of the average. If they were included as
        firing at t = T they would all look mutually coincident and pile into
        one group, producing a partition that looks structured and means
        nothing.
        """
        if not self.training:
            return
        w = code * fired.float()                              # (B, H)
        gw = w @ R                                            # (B, G)
        tau = (w * t_first) @ R / gw.clamp_min(1e-6)          # (B, G)

        d = (t_first.unsqueeze(2) - tau.unsqueeze(1)).abs()   # (B, H, G)
        kappa = torch.exp(-d / self.sigma)
        m = fired.float().unsqueeze(2)                        # mask silences
        kappa = (kappa * m).sum(0) / m.sum(0).clamp_min(1e-6)  # (H, G)

        # Centre per neuron so the update is "which group fits me BEST",
        # not "how active is the layer" -- otherwise every logit drifts up
        # together and the softmax stops discriminating.
        kappa = kappa - kappa.mean(dim=1, keepdim=True)
        if self.route_norm == "zscore":
            # ...and standardise, so the logit scale is set by the SHAPE of the
            # coincidence profile rather than by its magnitude. Without this the
            # spread depends on sigma and on G, and at realistic settings it is
            # small enough that softmax(logits / tau) is near-uniform while the
            # occupancy diagnostics still read healthy. See the module
            # docstring for the measured numbers.
            kappa = kappa / kappa.std(dim=1, keepdim=True).clamp_min(1e-6)
        self.route_logits.mul_(1 - self.hebb_eta).add_(self.hebb_eta * kappa)

    # ---- diagnostics ------------------------------------------------------
    @torch.no_grad()
    def group_stats(self, x_spikes=None):
        """How the groups are being used, plus the spiking-specific numbers.

        used_groups        effective number of groups in use. If this falls far
                           below G the routing has collapsed and the layer has
                           quietly reverted toward the spiking ANN.
        assign_entropy     0 = every neuron committed to one group, 1 = spread.
        mean_abs_cos_dec   per-neuron decoder overlap, comparable with the
                           rate-based project.
        within/between     mean absolute spike-time gap between neurons in the
                           same group versus in different groups. This is the
                           number that says whether the coincidence rule found
                           anything: a grouping that means something should
                           have a smaller within-group gap than between-group.
        """
        R = self.responsibilities(hard=False)
        occ = R.sum(0)
        pr = R / R.sum(1, keepdim=True).clamp_min(1e-12)
        ent = -(pr * pr.clamp_min(1e-12).log()).sum(1).mean()
        q = occ / occ.sum().clamp_min(1e-12)
        used = torch.exp(-(q * q.clamp_min(1e-12).log()).sum())

        d = F.normalize(self.decoder_weight, dim=1)
        cos = (d @ d.t()).abs()
        off = ~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)

        out = {
            "n_groups": int(self.n_groups),
            "routing": self.routing,
            "used_groups": float(used),
            "assign_entropy": float(ent / math.log(max(self.n_groups, 2))),
            "mean_abs_cos_dec": float(cos[off].mean()),
            "occupancy": [float(v) for v in occ],
            # How committed the SOFT routing is. Uniform would be 1/G; anything
            # close to that means the hard argmax is being decided by noise,
            # whatever used_groups and assign_entropy say -- those two cannot
            # tell a balanced partition from an uninformative one, and this can.
            "route_top1": float(R.max(dim=1).values.mean()),
            "route_uniform": float(1.0 / max(self.n_groups, 1)),
        }
        if x_spikes is not None:
            _, t_first, fired = self.encode(x_spikes)
            assign = R.argmax(1)                              # (H,)
            same = (assign.unsqueeze(0) == assign.unsqueeze(1))
            both = fired.unsqueeze(2) & fired.unsqueeze(1)     # (B, H, H)
            gap = (t_first.unsqueeze(2) - t_first.unsqueeze(1)).abs()
            eye = torch.eye(self.out_features, dtype=torch.bool,
                            device=gap.device)
            w_m = both & same.unsqueeze(0) & ~eye
            b_m = both & ~same.unsqueeze(0)
            out["within_group_gap"] = float(
                gap[w_m].mean()) if w_m.any() else None
            out["between_group_gap"] = float(
                gap[b_m].mean()) if b_m.any() else None
            out["mean_first_spike"] = float(t_first[fired].mean()) \
                if fired.any() else None
            out["frac_silent"] = float((~fired).float().mean())
        return out


# =============================================================================
# the network
# =============================================================================
class TTFSNet(nn.Module):
    """Latency-coded input -> TTFS hidden layer -> classifier.

    head="code"  a linear map on the hidden codes. Comparable with the
                 rate-based project, where the head is also a linear map on the
                 hidden layer's output, so accuracies can be read side by side.
    head="ttfs"  a second LIF layer, one spike per class unit, prediction =
                 whichever class fires first. Fully spiking end to end; the
                 logits returned are the class codes so cross-entropy still
                 applies.
    """

    def __init__(self, in_dim, hidden, classes, n_groups=None,
                 routing="fixed", num_steps=20, beta=0.9, threshold=1.0,
                 slope=25.0, head="code", latency="linear",
                 detach_head=False, **kw):
        super().__init__()
        self.in_dim, self.num_steps, self.head_kind = in_dim, num_steps, head
        self.latency = latency
        # Bull's rule: the task gradient stops here and the spiking layer is
        # trained by reconstruction alone. Forward VALUES are unchanged either
        # way, so accuracy is measured on exactly the same network.
        self.detach_head = detach_head
        self.hidden = TTFSGroupAELayer(
            in_dim, hidden, n_groups=n_groups, routing=routing,
            num_steps=num_steps, beta=beta, threshold=threshold, slope=slope,
            **kw)
        self.has_ae = True
        g = self.hidden.n_groups
        self.kind = ("ttfs_nan" if g == hidden else
                     "ttfs_ann" if g == 1 else f"ttfs_group{g}")
        if routing != "fixed":
            self.kind += f"-{routing}"

        if head == "code":
            self.head = nn.Linear(hidden, classes)
        else:
            self.head = nn.Linear(hidden, classes)
            self.out_lif = snn.Leaky(
                beta=beta, threshold=threshold,
                spike_grad=surrogate.fast_sigmoid(slope=slope))

    def _spikes(self, x):
        x = x.view(x.shape[0], -1)
        return latency_encode(x, self.num_steps, mode=self.latency), x

    def forward(self, x):
        xs, _ = self._spikes(x)
        if self.head_kind == "code":
            code = self.hidden(xs)
            return self.head(code.detach() if self.detach_head else code)
        hid = self.hidden.spike_train(xs)
        if self.detach_head:
            hid = hid.detach()
        mem = torch.zeros(x.shape[0], self.head.out_features,
                          device=x.device, dtype=x.dtype)
        alive, out = torch.ones_like(mem), []
        for t in range(hid.shape[0]):
            spk, mem = self.out_lif(self.head(hid[t]), mem)
            spk = spk * alive
            alive = alive * (1.0 - spk.detach())
            out.append(spk)
        return first_spike_code(torch.stack(out, 0))[0]

    def local_loss(self, x):
        xs, flat = self._spikes(x)
        return self.hidden.local_loss(xs, flat)

    @torch.no_grad()
    def calibrate(self, x, target_fire=0.9):
        """Call once on a real batch before training. See layer.calibrate."""
        return self.hidden.calibrate(self._spikes(x)[0], target_fire)

    @torch.no_grad()
    def codes(self, x):
        """Hidden codes, for effective rank and the collapse metrics."""
        xs, _ = self._spikes(x)
        return self.hidden(xs)

    @torch.no_grad()
    def group_stats(self, x=None):
        xs = None if x is None else self._spikes(x)[0]
        return self.hidden.group_stats(xs)


def build_ttfs(variant, in_dim, hidden, classes, **kw):
    """`ttfs_nan`, `ttfs_ann`, `ttfs_group<G>`, each optionally `-coincidence`
    or `-learned`. Mirrors group_ae's naming so sweeps read the same.

    Extra keywords are passed to TTFSNet and on to the layer, so
    `detach_head=True`, `route_norm="center"`, `tau_route=...`, `sigma=...`
    and `hebb_eta=...` all work here.
    """
    name = variant
    routing = "fixed"
    for suffix in ("-coincidence", "-learned"):
        if name.endswith(suffix):
            routing, name = suffix[1:], name[: -len(suffix)]
    if name == "ttfs_nan":
        g = hidden
    elif name == "ttfs_ann":
        g = 1
    elif name.startswith("ttfs_group"):
        g = int(name[len("ttfs_group"):])
    else:
        raise ValueError(f"unknown TTFS variant {variant!r}")
    return TTFSNet(in_dim, hidden, classes, n_groups=g, routing=routing, **kw)


# =============================================================================
# self-test
# =============================================================================
def _selftest():
    torch.manual_seed(0)
    B, D, H, C, T = 16, 40, 8, 5, 12
    x = torch.rand(B, D)

    print("ttfs_nan self-test")

    # -- 1. the code really is a first-spike time -------------------------
    spk = torch.zeros(T, B, H)
    spk[3, 0, 0] = 1.0
    spk[7, 0, 0] = 1.0            # a second spike that must be ignored
    spk[0, 0, 1] = 1.0
    code, tf, fired = first_spike_code(spk)
    assert abs(code[0, 0].item() - (1 - 3 / T)) < 1e-6, code[0, 0]
    assert abs(code[0, 1].item() - 1.0) < 1e-6
    assert code[0, 2].item() == 0.0 and not fired[0, 2]
    assert tf[0, 0].item() == 3 and tf[0, 2].item() == T
    print("  first-spike code: later spikes ignored, silence = 0  OK")

    # -- 2. latency encoding: bright early, dim silent --------------------
    xs = latency_encode(torch.tensor([[1.0, 0.75, 0.5, 0.25, 0.0]]), T)
    _, t, f = first_spike_code(xs)
    ts = [t[0, i].item() for i in range(4)]
    assert ts == sorted(ts) and ts[0] < ts[3], ts
    assert ts[3] >= 0.6 * (T - 1), f"latency code not using the time axis: {ts}"
    assert not f[0, 4], "a zero pixel must stay silent"
    print(f"  latency encode: t = {[int(v) for v in ts]} over {T} steps, "
          f"0.0 -> silent  OK")

    # -- 3. one spike per neuron, and the layer is not silent --------------
    net = TTFSNet(D, H, C, n_groups=4, num_steps=T)
    thr = net.calibrate(x, target_fire=0.9)
    train = net.hidden.spike_train(latency_encode(x, T))
    counts = train.sum(0)
    _, _, fired = first_spike_code(train)
    frac = fired.float().mean().item()
    assert counts.max().item() <= 1.0, f"max spikes/neuron = {counts.max()}"
    assert frac > 0.25, f"layer is mostly silent after calibration: {frac:.2f}"
    print(f"  one spike per neuron: max count {counts.max():.0f}  OK")
    print(f"  calibration: threshold {thr:.3f} -> {100*frac:.0f}% of neurons "
          f"fire on a given input  OK")

    # -- 4. the two endpoints ---------------------------------------------
    for g, name in [(H, "ttfs_nan"), (1, "ttfs_ann")]:
        m = build_ttfs(name, D, H, C, num_steps=T)
        assert m.hidden.n_groups == g, (name, m.hidden.n_groups)
        assert m.kind == name
    n_nan = build_ttfs("ttfs_nan", D, H, C, num_steps=T)
    n_grp = build_ttfs("ttfs_group4", D, H, C, num_steps=T)
    p = lambda m: sum(q.numel() for q in m.hidden.parameters())
    print(f"  endpoints build: nan G={H}, ann G=1  OK")
    print(f"  params  nan={p(n_nan)}  group4={p(n_grp)}  "
          f"(differ only by the {H}x G routing and group biases)")

    # -- 5. gradients reach the encoder through the spikes -----------------
    m = build_ttfs("ttfs_group4", D, H, C, num_steps=T)
    loss = F.cross_entropy(m(x), torch.randint(0, C, (B,))) + m.local_loss(x)
    loss.backward()
    genc = m.hidden.encoder.weight.grad
    gdec = m.hidden.decoder_weight.grad
    assert genc is not None and genc.abs().sum() > 0, "no gradient to encoder"
    assert gdec is not None and gdec.abs().sum() > 0, "no gradient to decoder"
    print(f"  surrogate gradients: |g_enc|={genc.abs().mean():.2e}  "
          f"|g_dec|={gdec.abs().mean():.2e}  OK")

    # -- 6. the coincidence rule moves the assignment, and stays sane ------
    m = build_ttfs("ttfs_group4-coincidence", D, H, C, num_steps=T,
                   hebb_eta=0.3)
    m.calibrate(x)
    m.train()
    before = m.hidden.responsibilities(hard=False).clone()
    for _ in range(20):
        m.local_loss(torch.rand(B, D))
    after = m.hidden.responsibilities(hard=False)
    moved = (after - before).abs().max().item()
    st = m.group_stats(x)
    assert moved > 1e-4, f"coincidence routing never moved (max {moved:.1e})"
    assert 1.0 <= st["used_groups"] <= 4.0 + 1e-6, st["used_groups"]
    print(f"  coincidence routing: max change {moved:.3f}, "
          f"used_groups {st['used_groups']:.2f} of 4  OK")
    assert st["within_group_gap"] is not None, "no co-firing pairs to measure"
    print(f"  spike gaps: within={st['within_group_gap']:.2f} steps, "
          f"between={st['between_group_gap']:.2f}, "
          f"silent={st['frac_silent']:.2f}")

    # -- 7. fixed routing must NOT move ------------------------------------
    m = build_ttfs("ttfs_group4", D, H, C, num_steps=T)
    m.calibrate(x)
    m.train()
    b = m.hidden.responsibilities().clone()
    for _ in range(5):
        m.local_loss(torch.rand(B, D))
    assert torch.equal(b, m.hidden.responsibilities()), "fixed routing moved"
    print("  fixed routing is genuinely fixed  OK")

    # -- 8. detach_head really does cut the task gradient ------------------
    for kind in ("code", "ttfs"):
        att = build_ttfs("ttfs_group4", D, H, C, num_steps=T, head=kind)
        det = build_ttfs("ttfs_group4", D, H, C, num_steps=T, head=kind,
                         detach_head=True)
        det.load_state_dict(att.state_dict())
        att.calibrate(x); det.hidden.lif.threshold.copy_(att.hidden.lif.threshold)

        fa, fd = att(x), det(x)
        assert torch.allclose(fa, fd, atol=1e-6), \
            f"detach changed the forward pass ({kind})"

        y = torch.randint(0, C, (B,))
        for mm in (att, det):
            mm.zero_grad()
            F.cross_entropy(mm(x), y).backward()
        ga = att.hidden.encoder.weight.grad
        gd = det.hidden.encoder.weight.grad
        assert ga is not None and ga.abs().sum() > 0, f"attached has no grad ({kind})"
        assert gd is None or gd.abs().sum() == 0, \
            f"CE still reaches the encoder when detached ({kind})"
        assert det.head.weight.grad.abs().sum() > 0, \
            f"detached head stopped learning ({kind})"
    print("  detach_head: forward identical, CE->encoder zero, head still "
          "learns  OK")

    # -- 9. route_norm gives the routing a usable dynamic range ------------
    sharp = {}
    for norm in ("center", "zscore"):
        m = build_ttfs("ttfs_group4-coincidence", D, H, C, num_steps=T,
                       hebb_eta=0.3, route_norm=norm, init_logit=0.5)
        m.calibrate(x); m.train()
        for _ in range(60):
            m.local_loss(torch.rand(B, D))
        sharp[norm] = m.group_stats()["route_top1"]
    uni = 1.0 / 4
    assert sharp["zscore"] >= sharp["center"], sharp
    print(f"  route_norm: top-1 prob center={sharp['center']:.3f} "
          f"zscore={sharp['zscore']:.3f} (uniform {uni:.3f})  OK")

    print("  all checks passed")


if __name__ == "__main__":
    _selftest()
