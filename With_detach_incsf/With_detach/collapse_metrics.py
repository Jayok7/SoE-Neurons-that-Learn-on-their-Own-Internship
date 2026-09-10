"""
collapse_metrics.py — numerical quantification of hidden-layer collapse.

The question this module answers: of the H neurons in the hidden layer, how
many are doing *distinct* work? A layer has collapsed when its neurons have
become redundant copies of one another (or have gone silent), so that the
layer's effective width is far below its nominal width H.

Every metric below is defined so that it can be compared across widths. Where
a metric naturally scales with H it is reported BOTH raw and normalised by H
(the `*_frac` fields), because the sweep compares H=4 against H=1024 and a raw
count of live neurons is not comparable across those.

--------------------------------------------------------------------------
THE METRICS
--------------------------------------------------------------------------
Group A — activation-space (what the layer actually outputs). These are the
metrics to trust for ALL variants, including ANN, since they only look at the
representation, not at any particular decoder parameterisation.

  eff_rank_act        Effective rank (entropy of the normalised singular value
                      spectrum, exp of Shannon entropy) of the mean-centred
                      activation matrix A (n_samples x H). Continuous analogue
                      of matrix rank: H if all neurons carry independent
                      variance, ~1 if they all carry the same direction.
                      Reported as eff_rank_act_frac = eff_rank / H.
  part_ratio_act      Participation ratio (sum s^2)^2 / sum s^4 of the same
                      spectrum. A second, harsher rank estimate — it discounts
                      small directions more aggressively than entropy does, so
                      the gap between the two is itself informative.
  pr90_act            Number of principal components needed to explain 90% of
                      activation variance. The most interpretable of the three.
  mean_abs_corr       Mean |Pearson correlation| over all off-diagonal neuron
                      pairs. 0 = perfectly decorrelated, 1 = every neuron is a
                      copy of every other. Direct "overlap" number.
  frac_pairs_corr_90  Fraction of neuron pairs with |corr| > 0.9. A blunt,
                      easily-reported "how much of the layer is duplicated".

Group B — dead / saturated units. The other failure mode: a neuron that is not
redundant but simply constant, contributing no information at all.

  frac_dead           Fraction of neurons whose activation std over the eval
                      set is below `dead_std_tol` (default 1e-3): the neuron
                      outputs essentially the same value for every input.
  frac_saturated      Fraction of neurons that are dead AND pinned at a sigmoid
                      extreme (mean activation < 0.01 or > 0.99). A subset of
                      frac_dead, separated out because it points at a different
                      cause (encoder weights blown up, not collapsed to zero).
  frac_alive          1 - frac_dead. The single headline number: the fraction
                      of the layer that is doing anything at all.
  eff_neurons         eff_rank_act itself, restated as "the layer of width H is
                      behaving like a layer of width eff_neurons".

Group C — decoder-space (per-neuron reconstruction). Meaningful for NAN only,
where row j of decoder_weight is neuron j's own template for the whole image.

  mean_abs_cos_dec    Mean |cosine similarity| over off-diagonal pairs of
                      per-neuron decoder weight vectors. This is the metric
                      that most directly captures "the neurons are all trying
                      to reconstruct the same thing".
  eff_rank_dec        Effective rank of the decoder weight matrix.
  frac_pairs_cos_90   Fraction of decoder pairs with |cosine| > 0.9.

  For ANN these are returned as None. In LayerAELayer a row of `decoder_weight`
  is an INPUT dimension, not a neuron, so the per-neuron interpretation does
  not hold — reporting a number there would be misleading. This is deliberate
  and matches the warning already in models.py.

Group D — a single scalar for plotting against hidden width.

  collapse_index      1 - eff_rank_act / H, in [0, 1].
                      0 = no collapse (every neuron independent),
                      1 = total collapse (layer behaves as a single neuron).
                      This is the quantity to put on the y-axis of the sweep.

--------------------------------------------------------------------------
NOTES ON CORRECTNESS
--------------------------------------------------------------------------
* Activations are mean-centred before the SVD. Without centring, a constant
  offset shared by all neurons appears as a large first singular value and
  fakes collapse where there is none.
* Effective rank uses only non-negligible singular values (> eps * s_max) to
  avoid 0*log(0) noise from numerical zeros.
* Correlations are computed with a std floor so dead neurons (std = 0) do not
  produce NaNs; a dead neuron's correlation with anything is defined as 0 here
  and it is instead counted by frac_dead. This keeps the two failure modes
  cleanly separated rather than letting one contaminate the other.
* All metrics are computed under torch.no_grad() on a fixed, deterministic
  subset of the test set (default 2000 images) so numbers are comparable
  across variants, seeds and widths.
"""

import numpy as np
import torch

try:
    from models import DEVICE
except ImportError:
    from nan.models import DEVICE


# ==============================================================================
# Spectrum helpers
# ==============================================================================
def _singular_values(matrix):
    """Singular values of a 2-D tensor, descending, as a float64 numpy array."""
    with torch.no_grad():
        s = torch.linalg.svdvals(matrix.double())
    return s.cpu().numpy()


def effective_rank(singular_values, eps=1e-12):
    """exp(Shannon entropy of the normalised singular value spectrum).

    Roy & Vetterli's effective rank. Equals k exactly when there are k equal
    singular values and the rest are zero, and varies continuously in between,
    which is what makes it usable as "how many neurons' worth of independent
    signal is present".
    """
    s = np.asarray(singular_values, dtype=np.float64)
    s = s[s > eps * (s.max() if s.size and s.max() > 0 else 1.0)]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * np.log(p)).sum()
    return float(np.exp(entropy))


def participation_ratio(singular_values):
    """(sum s^2)^2 / sum s^4 — a second rank estimate from the same spectrum.

    Harsher than effective rank: it weights by squared variance, so a spectrum
    with one dominant direction and a long weak tail scores much lower here
    than it does under the entropy measure. Reporting both makes the shape of
    the spectrum visible from two numbers.
    """
    s2 = np.asarray(singular_values, dtype=np.float64) ** 2
    denom = (s2 ** 2).sum()
    if denom <= 0:
        return 0.0
    return float((s2.sum() ** 2) / denom)


def components_for_variance(singular_values, target=0.90):
    """Number of principal components needed to reach `target` of the variance."""
    s2 = np.asarray(singular_values, dtype=np.float64) ** 2
    total = s2.sum()
    if total <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(s2) / total, target) + 1)


# ==============================================================================
# Pairwise overlap helpers
# ==============================================================================
def _offdiag(matrix):
    """The off-diagonal entries of a square matrix, flattened."""
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return matrix[mask]


def correlation_stats(activations, std_floor=1e-8):
    """Mean |correlation| and fraction of highly-correlated pairs.

    activations: (n_samples, H) numpy array.

    Dead neurons (std ~ 0) would give 0/0. Their correlations are set to 0 so
    they never inflate the overlap statistic; they are reported separately by
    dead_unit_stats instead.
    """
    a = np.asarray(activations, dtype=np.float64)
    h = a.shape[1]
    if h < 2:
        return 0.0, 0.0
    centred = a - a.mean(axis=0, keepdims=True)
    std = centred.std(axis=0)
    live = std > std_floor
    corr = np.zeros((h, h), dtype=np.float64)
    if live.sum() >= 2:
        z = centred[:, live] / std[live]
        c = (z.T @ z) / a.shape[0]
        idx = np.where(live)[0]
        corr[np.ix_(idx, idx)] = np.clip(c, -1.0, 1.0)
    off = np.abs(_offdiag(corr))
    return float(off.mean()), float((off > 0.9).mean())


def cosine_stats(weights):
    """Mean |cosine similarity| and fraction of near-parallel pairs, over the
    rows of `weights` (a (H, in_dim) numpy array of per-neuron decoders)."""
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] < 2:
        return 0.0, 0.0
    norms = np.linalg.norm(w, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    unit = w / norms
    cos = np.clip(unit @ unit.T, -1.0, 1.0)
    off = np.abs(_offdiag(cos))
    return float(off.mean()), float((off > 0.9).mean())


def dead_unit_stats(activations, dead_std_tol=1e-3):
    """Fraction of neurons that are constant, and of those, how many are pinned
    at a sigmoid extreme."""
    a = np.asarray(activations, dtype=np.float64)
    std = a.std(axis=0)
    mean = a.mean(axis=0)
    dead = std < dead_std_tol
    saturated = dead & ((mean < 0.01) | (mean > 0.99))
    return float(dead.mean()), float(saturated.mean())


# ==============================================================================
# Activation collection
# ==============================================================================
@torch.no_grad()
def collect_activations(model, loader, max_samples=2000):
    """Hidden-layer activations for the first `max_samples` test images.

    Deterministic: the test loader is unshuffled, so this is the same fixed
    subset for every variant, seed and width, which is what makes the metrics
    comparable. Returns a (n, H) numpy array, or None for variants with no
    hidden layer (the `linear` baseline).
    """
    model.eval()
    hidden = getattr(model, "hidden", None)
    if hidden is None:
        fc = getattr(model, "fc", None)          # plain MLP / frozen MLP
        if fc is None:
            return None                          # linear baseline: nothing to measure
        act_fn = getattr(model, "act", torch.sigmoid)

        def encode(x):
            return act_fn(fc(x))
    else:
        def encode(x):
            return hidden(x)

    chunks, n = [], 0
    for x, _ in loader:
        x = x.to(DEVICE)
        chunks.append(encode(x).detach().cpu())
        n += x.size(0)
        if n >= max_samples:
            break
    return torch.cat(chunks)[:max_samples].numpy()


def _per_neuron_decoder(model):
    """The (H, in_dim) per-neuron decoder matrix, or None where the concept
    does not apply (ANN's shared decoder, plain MLPs, linear baseline).

    A layer opts in either by being NAN, or by setting `per_neuron_decoder =
    True` (the group-AE layer does this: it keeps one decoder row per neuron
    and pools only in the loss, so row j really is neuron j's template).
    """
    hidden = getattr(model, "hidden", None)
    if hidden is None:
        return None
    opted_in = getattr(hidden, "per_neuron_decoder", False)
    if not opted_in and getattr(model, "kind", None) != "nan":
        return None                              # ANN rows are input dims, not neurons
    w = getattr(hidden, "decoder_weight", None)
    return None if w is None else w.detach().cpu().numpy()


# ==============================================================================
# The full report
# ==============================================================================
@torch.no_grad()
def collapse_report(model, loader, max_samples=2000, dead_std_tol=1e-3):
    """Every metric above, as one flat JSON-serialisable dict.

    Returns None for the `linear` variant, which has no hidden layer to assess.
    """
    acts = collect_activations(model, loader, max_samples)
    if acts is None:
        return None

    h = acts.shape[1]
    centred = acts - acts.mean(axis=0, keepdims=True)
    sv = _singular_values(torch.from_numpy(centred))

    eff_rank = effective_rank(sv)
    part_ratio = participation_ratio(sv)
    pr90 = components_for_variance(sv, 0.90)
    mean_corr, frac_corr90 = correlation_stats(acts)
    frac_dead, frac_sat = dead_unit_stats(acts, dead_std_tol)

    report = {
        "hidden_width": int(h),
        "n_eval_samples": int(acts.shape[0]),
        # Group A — activation space
        "eff_rank_act": eff_rank,
        "eff_rank_act_frac": eff_rank / h,
        "part_ratio_act": part_ratio,
        "part_ratio_act_frac": part_ratio / h,
        "pr90_act": pr90,
        "pr90_act_frac": pr90 / h,
        "mean_abs_corr": mean_corr,
        "frac_pairs_corr_90": frac_corr90,
        # Group B — dead / saturated
        "frac_dead": frac_dead,
        "frac_saturated": frac_sat,
        "frac_alive": 1.0 - frac_dead,
        "eff_neurons": eff_rank,
        # Group C — decoder space (filled below when applicable)
        "mean_abs_cos_dec": None,
        "eff_rank_dec": None,
        "eff_rank_dec_frac": None,
        "frac_pairs_cos_90": None,
        # Group D — headline
        "collapse_index": 1.0 - eff_rank / h,
        # spectrum kept for plotting; truncated so JSONs stay small
        "singular_values": sv[:64].tolist(),
    }

    dec = _per_neuron_decoder(model)
    if dec is not None:
        mean_cos, frac_cos90 = cosine_stats(dec)
        dec_rank = effective_rank(_singular_values(torch.from_numpy(dec)))
        report.update({
            "mean_abs_cos_dec": mean_cos,
            "eff_rank_dec": dec_rank,
            "eff_rank_dec_frac": dec_rank / h,
            "frac_pairs_cos_90": frac_cos90,
        })
    return report


def summarise(report):
    """One-line human-readable summary for stdout during a run."""
    if report is None:
        return "no hidden layer"
    s = (f"eff_rank={report['eff_rank_act']:.1f}/{report['hidden_width']} "
         f"({100 * report['eff_rank_act_frac']:.0f}%)  "
         f"PR90={report['pr90_act']}  "
         f"|corr|={report['mean_abs_corr']:.3f}  "
         f"dead={100 * report['frac_dead']:.0f}%  "
         f"collapse={report['collapse_index']:.3f}")
    if report["mean_abs_cos_dec"] is not None:
        s += f"  |cos_dec|={report['mean_abs_cos_dec']:.3f}"
    return s


def aggregate_reports(reports):
    """Mean and SEM across seeds for every scalar field.

    Fields that are None for a variant stay None; the singular-value spectrum
    is averaged elementwise where lengths agree and dropped otherwise.
    """
    reports = [r for r in reports if r is not None]
    if not reports:
        return None
    out = {}
    for key in reports[0]:
        vals = [r.get(key) for r in reports]
        if any(v is None for v in vals):
            out[key] = None
        elif key == "singular_values":
            lengths = {len(v) for v in vals}
            out[key] = (np.mean(np.array(vals), axis=0).tolist()
                        if len(lengths) == 1 else None)
        else:
            arr = np.array(vals, dtype=np.float64)
            out[key] = {"mean": float(arr.mean()),
                        "sem": float(arr.std() / np.sqrt(len(arr)))}
    return out
