"""
nan/experiments.py — combined experiment runner, schedules, metrics, and plotting.

Includes:
    - Collapse diagnostics (activation correlation, effective rank, decoder similarity)
    - Multi-seed training aggregation and SEM calculation
    - Training schedules (interleaved, sequential, and hybrid warm-up)
    - Matplotlib curve plotting, sweep plotting, and neuron weight/activation visualization
    - High-level orchestrators (exp_curves, exp_schedule_compare, exp_sweep, exp_seq_variants)
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F

# Fallback imports to support running both as a package and as a standalone script
try:
    from .models import DEVICE, HiddenLayerClassifier, get_loaders, PENALTIES, outdir
except ImportError:
    from models import DEVICE, HiddenLayerClassifier, get_loaders, PENALTIES, outdir

# Keep the old name working inside the experiment bodies
NAN_Classifier = HiddenLayerClassifier

# ==============================================================================
# 1. Collapse Diagnostics & Multi-Seed Aggregation
# ==============================================================================
def metrics(model, loader, probe_batch):
    """Compute test accuracy plus three 'neuron collapse' diagnostics.

    probe_batch: a fixed batch of inputs used to measure how diverse the neurons
    are. The three collapse metrics answer the supervisor's question 'do all the
    neurons end up learning the same thing?':
        decoder_similarity : how alike the neurons' decoder weights are (NAN only)
        activation_corr    : how correlated the neurons' outputs are
        effective_rank     : how many neurons do genuinely independent work
        recon_mse          : how well the autoencoders reconstruct the input,
                             measured on the test set (NAN/ANN only; -1 for NN).
                             This is the SAME quantity local_loss() optimises, so
                             it reads out reconstruction QUALITY directly rather
                             than inferring it from accuracy/diversity. It moves
                             independently of accuracy because the head and the
                             decoders are trained by different, non-overlapping
                             losses (cross-entropy vs reconstruction MSE).
    """
    model.eval()

    # ---- test accuracy ----
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    accuracy = 100.0 * correct / total

    layer = model.hidden

    # ---- reconstruction MSE + reconstruction ACCURACY on the TEST set ----
    # recon_mse: reuse the layer's own local_loss so the measured error is exactly
    #   the objective that was optimised. Lower = better.
    # recon_acc: a percentage that RISES with reconstruction quality, so it reads
    #   as an accuracy and its curve slopes UP as the model improves. Defined as
    #       recon_acc = 100 * (1 - mse / var(inputs))
    #   i.e. the fraction of input variance the autoencoder explains (the same
    #   "explained variance" idea as R^2), expressed as a percent and clamped to
    #   [0, 100]. 100% = perfect reconstruction; 0% = no better than predicting
    #   the dataset mean.
    # A plain NN has no autoencoder (local_loss is a constant zero with no gradient
    # path), so both report -1 ('n/a'), mirroring the dec_sim convention.
    #
    # Detecting "has an autoencoder": we must NOT rely on local_loss().requires_grad.
    # That is False both for a plain NN (no AE) AND for a real AE layer whose params
    # have been frozen (e.g. the frozen sequential schedule's Phase 2) — which would
    # wrongly mark a frozen ANN/NAN as n/a. Instead we test the layer structurally:
    # only PlainNNLayer has an all-zero, non-trainable decoder placeholder. Any layer
    # whose decoder has actual magnitude is a genuine autoencoder, frozen or not.
    has_autoencoder = bool(layer.decoder_weight.abs().sum() > 0)
    if has_autoencoder:
        with torch.no_grad():
            recon_sum = recon_count = 0
            sq_sum = lin_sum = elem_count = 0.0
            for x, _y in loader:
                x = x.to(DEVICE)
                recon_sum += layer.local_loss(x).item() * x.size(0)
                recon_count += x.size(0)
                # running totals for input variance (over all pixels & samples)
                sq_sum += (x ** 2).sum().item()
                lin_sum += x.sum().item()
                elem_count += x.numel()
            recon_mse = recon_sum / recon_count
            input_var = sq_sum / elem_count - (lin_sum / elem_count) ** 2
            input_var = max(input_var, 1e-12)            # guard against /0
            recon_acc = 100.0 * (1.0 - recon_mse / input_var)
            recon_acc = float(min(100.0, max(0.0, recon_acc)))   # clamp to [0,100]
    else:
        recon_mse = -1.0           # not applicable for a plain NN
        recon_acc = -1.0


    # ---- decoder similarity (only meaningful when each neuron has its own
    #      decoder row, i.e. the NAN layers). For NN the decoder is unused, for
    #      ANN it is shared and indexed by input dimension, so we report -1
    #      ('not applicable') in those cases. ----
    num_neurons = layer.encoder.out_features
    decoder_is_per_neuron = (layer.decoder_weight.shape[0] == num_neurons
                             and layer.decoder_weight.abs().sum() > 0)
    if decoder_is_per_neuron:
        unit_vectors = F.normalize(layer.decoder_weight, dim=1)
        similarity = unit_vectors @ unit_vectors.t()
        off_diagonal = similarity[~torch.eye(num_neurons, dtype=torch.bool,
                                             device=probe_batch.device)]
        decoder_similarity = off_diagonal.abs().mean().item()
    else:
        decoder_similarity = -1.0     # not applicable for NN / ANN

    # ---- activation correlation: centre and standardise each neuron's outputs
    #      over the batch, then average the off-diagonal of the correlation
    #      matrix (1 = neurons are clones, 0 = uncorrelated). ----
    activations = layer(probe_batch)
    activations = activations - activations.mean(0, keepdim=True)
    standardised = activations / (activations.std(0, keepdim=True) + 1e-8)
    correlation = (standardised.t() @ standardised) / standardised.size(0)
    n = correlation.size(0)
    activation_corr = correlation[~torch.eye(n, dtype=torch.bool,
                                             device=probe_batch.device)].abs().mean().item()

    # ---- effective rank: how many neurons are effectively independent. We take
    #      the singular values of the activation matrix, turn them into a
    #      probability distribution, and compute exp(entropy). Ranges from 1
    #      (total collapse onto one direction) up to num_neurons (full diversity). ----
    singular_values = torch.linalg.svdvals(activations)
    spectrum = singular_values / singular_values.sum()
    entropy = -(spectrum * (spectrum + 1e-12).log()).sum()
    effective_rank = torch.exp(entropy).item()

    return {"acc": accuracy, "dec_sim": decoder_similarity,
            "act_corr": activation_corr, "eff_rank": effective_rank,
            "recon_mse": recon_mse, "recon_acc": recon_acc}


def multiseed(make_and_run, seeds, in_dim, hidden, num_classes):
    """Run an experiment once per seed and aggregate the per-epoch metrics into
    a mean and standard-error curve.

    make_and_run(seed) -> a list (one entry per epoch) of metric dicts.
    Returns {metric_name: {'mean': [...per epoch...], 'sem': [...]}}.
    """
    runs_per_seed = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        runs_per_seed.append(make_and_run(seed))

    metric_names = [name for name in runs_per_seed[0][0].keys() if name != "phase"]
    aggregated = {}
    for name in metric_names:
        # shape (num_seeds, num_epochs): this metric's value at each epoch per seed
        values = np.array([[epoch[name] for epoch in run] for run in runs_per_seed])
        aggregated[name] = {
            "mean": values.mean(0).tolist(),
            "sem": (values.std(0) / np.sqrt(len(seeds))).tolist(),
        }
    # carry the phase markers (for the sequential schedule) straight through
    aggregated["phase"] = [epoch.get("phase", 0) for epoch in runs_per_seed[0]]
    return aggregated

# ==============================================================================
# 2. Training Schedules
# ==============================================================================
def run_interleaved(model, train_loader, test_loader, probe_batch, epochs,
                    local_weight, div_weight=0.0, penalty="decoder"):
    """Paper-style schedule: each batch is randomly EITHER a task-learning cycle
    OR an autoencoding cycle. Records the metrics once per epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    penalty_fn = PENALTIES[penalty]
    history = []
    for _epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            run_task_cycle = local_weight == 0 or torch.rand(1).item() < 0.5
            if run_task_cycle:
                optimizer.zero_grad()
                F.cross_entropy(model(x), y).backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                _, recon_loss = model(x, want_local=True)
                loss = local_weight * recon_loss
                if div_weight > 0:
                    loss = loss + div_weight * penalty_fn(model.hidden, x)
                # A plain NN has no autoencoder: its reconstruction loss is a
                # constant zero with no gradient path. There is nothing to learn
                # on an autoencoding cycle, so fall back to a task cycle rather
                # than calling backward() on a non-differentiable tensor.
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                else:
                    F.cross_entropy(model(x), y).backward()
                    optimizer.step()
        history.append(metrics(model, test_loader, probe_batch))
    return history


def run_sequential(model, train_loader, test_loader, probe_batch,
                   ae_epochs, task_epochs,
                   div_weight=0.0, penalty="decoder", finetune=False):
    """
    Sequential schedule (the variant the supervisor suggested):
      Phase 1 trains ONLY the autoencoders (optionally with a diversity penalty).
      Phase 2 trains the task. If finetune=True the encoder stays trainable at a
        low learning rate so the task signal can repair leftover collapse; if
        finetune=False the hidden layer is frozen, the strict 'is the AE-shaped
        representation good for the task?' test.
    Metrics are logged every epoch across BOTH phases, so a curve shows the
    Phase-1 -> Phase-2 transition.
    """
    penalty_fn = PENALTIES[penalty]
    history = []

    # ---- Phase 1: train the autoencoders (skipped for a plain NN, which has
    #      no autoencoder to train) ----
    first_batch = next(iter(train_loader))[0].to(DEVICE)
    has_autoencoder = model.hidden.local_loss(first_batch).requires_grad
    ae_optimizer = torch.optim.Adam(model.hidden.parameters(), lr=1e-3)
    for _epoch in range(ae_epochs):
        model.train()
        if has_autoencoder:
            for x, _y in train_loader:
                x = x.to(DEVICE)
                ae_optimizer.zero_grad()
                loss = model.hidden.local_loss(x)
                if div_weight > 0:
                    loss = loss + div_weight * penalty_fn(model.hidden, x)
                loss.backward()
                ae_optimizer.step()
        history.append({**metrics(model, test_loader, probe_batch), "phase": 1})

    # ---- Phase 2: train the task ----
    if finetune:
        # encoder keeps learning, but slowly, while the head learns at full speed
        task_optimizer = torch.optim.Adam([
            {"params": model.head.parameters(), "lr": 1e-3},
            {"params": model.hidden.parameters(), "lr": 1e-4},
        ])
    else:
        for param in model.hidden.parameters():
            param.requires_grad_(False)             # freeze the AE representation
        task_optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)

    for _epoch in range(task_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            task_optimizer.zero_grad()
            F.cross_entropy(model(x), y).backward()
            task_optimizer.step()
        history.append({**metrics(model, test_loader, probe_batch), "phase": 2})
    return history


def run_warmup_interleaved(model, train_loader, test_loader, probe_batch,
                           warmup_epochs, interleaved_epochs,
                           local_weight=1.0, div_weight=5.0, penalty="decoder"):
    """
    Hybrid schedule: a reconstruction WARM-UP, then full interleaved training.

    Phase 1 (warm-up): train ONLY the autoencoders, WITH the diversity penalty,
        to reach a good, decorrelated starting representation. No task signal,
        head idle (accuracy at chance) -- same as sequential's phase 1.
    Phase 2 (interleaved): hand off to the normal interleaved schedule (coin-flip
        task / reconstruction every batch), which keeps neurons diverse on its
        own. Nothing is frozen here -- the whole network trains.

    Hypothesis: warm-starting from a structured, decorrelated representation, then
    letting interleaved training take over, gets the benefit of a good init
    without the collapse risk of freezing. Sits between the existing two schedules.
    Metrics logged every epoch across both phases (phase markers 1 then 2).
    """
    penalty_fn = PENALTIES[penalty]
    history = []

    # ---- Phase 1: AE warm-up with the diversity penalty -------------------
    first_batch = next(iter(train_loader))[0].to(DEVICE)
    has_autoencoder = model.hidden.local_loss(first_batch).requires_grad
    warmup_optimizer = torch.optim.Adam(model.hidden.parameters(), lr=1e-3)
    for _epoch in range(warmup_epochs):
        model.train()
        if has_autoencoder:
            for x, _y in train_loader:
                x = x.to(DEVICE)
                warmup_optimizer.zero_grad()
                loss = model.hidden.local_loss(x)
                if div_weight > 0:
                    loss = loss + div_weight * penalty_fn(model.hidden, x)
                loss.backward()
                warmup_optimizer.step()
        history.append({**metrics(model, test_loader, probe_batch), "phase": 1})

    # ---- Phase 2: full interleaved training (whole network trainable) -----
    # one optimiser over ALL parameters, exactly like run_interleaved
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _epoch in range(interleaved_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            run_task_cycle = local_weight == 0 or torch.rand(1).item() < 0.5
            if run_task_cycle:
                optimizer.zero_grad()
                F.cross_entropy(model(x), y).backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                _, recon_loss = model(x, want_local=True)
                loss = local_weight * recon_loss
                if div_weight > 0:
                    loss = loss + div_weight * penalty_fn(model.hidden, x)
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                else:
                    F.cross_entropy(model(x), y).backward()
                    optimizer.step()
        history.append({**metrics(model, test_loader, probe_batch), "phase": 2})
    return history

# ==============================================================================
# 3. Plotting & Serialization Helper Functions
# ==============================================================================
def plot_curves(curve_dict, metric, ylabel, fname, title, subdir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.5))
    for label, agg in curve_dict.items():
        m = np.array(agg[metric]["mean"]); s = np.array(agg[metric]["sem"])
        ep = np.arange(1, len(m) + 1)
        plt.plot(ep, m, label=label, linewidth=2)
        plt.fill_between(ep, m - s, m + s, alpha=0.2)
    plt.xlabel("epoch"); plt.ylabel(ylabel); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    path = os.path.join(outdir(subdir), fname); plt.savefig(path, dpi=130); plt.close()
    print(f"  saved {path}")


def plot_sweep(xs, ys_dict, xlabel, fname, title, subdir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    am, asem = ys_dict["acc_mean"], ys_dict["acc_sem"]
    rm, rsem = ys_dict["rank_mean"], ys_dict["rank_sem"]
    ax1.errorbar(xs, am, yerr=asem, marker="o", color="tab:blue", label="test acc")
    ax2.errorbar(xs, rm, yerr=rsem, marker="s", color="tab:red", label="eff. rank")
    ax1.set_xlabel(xlabel); ax1.set_ylabel("test accuracy (%)", color="tab:blue")
    ax2.set_ylabel("effective rank", color="tab:red")
    ax1.set_title(title); ax1.grid(alpha=0.3); fig.tight_layout()
    path = os.path.join(outdir(subdir), fname); plt.savefig(path, dpi=130); plt.close()
    print(f"  saved {path}")


def save_json(obj, fname, subdir=None):
    path = os.path.join(outdir(subdir), fname)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  saved {path}")


def plot_neuron_decoders(model, resolution, fname, title, subdir=None,
                         max_neurons=40, ncols=8):
    """Visualise each neuron's decoder weights reshaped to (res x res).

    For NAN layers, each row of decoder_weight IS that neuron's reconstruction
    template — the spatial pattern it tries to reconstruct from its single
    scalar code. If neurons have learned independent features, these should
    look visibly different. If they've collapsed, they'll all look alike.

    Uses a red-blue diverging colourmap (like the reference image): blue for
    negative weights, red for positive, white for zero.

    For NN/ANN architectures the decoder weights aren't per-neuron, so this
    function will skip gracefully.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    layer = model.hidden
    num_neurons = layer.encoder.out_features
    decoder_is_per_neuron = (layer.decoder_weight.shape[0] == num_neurons
                             and layer.decoder_weight.abs().sum() > 0)
    if not decoder_is_per_neuron:
        print(f"  [plot_neuron_decoders] skipping — not a per-neuron decoder ({type(layer).__name__})")
        return

    weights = layer.decoder_weight.detach().cpu().numpy()   # (num_neurons, in_features)
    n_show = min(num_neurons, max_neurons)
    nrows = int(np.ceil(n_show / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 2.0))
    axes = np.array(axes).reshape(-1)   # flatten for easy indexing

    # Global colour normalisation so all subplots share the same scale
    vmax = np.abs(weights[:n_show]).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for i in range(n_show):
        w = weights[i].reshape(resolution, resolution)
        axes[i].imshow(w, cmap="RdBu_r", norm=norm, interpolation="nearest")
        axes[i].set_title(f"neuron {i}", fontsize=8)
        axes[i].axis("off")

    # Hide any leftover axes
    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir(subdir), fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


@torch.no_grad()
def plot_neuron_activations(model, test_loader, resolution, fname, title,
                            subdir=None, max_neurons=40, ncols=8, topk=50):
    """Visualise what each neuron responds to by averaging the top-k inputs
    that produce the highest activation for that neuron.

    This is the activation-based complement to the decoder-weight view: instead
    of showing WHAT the neuron reconstructs (its template), this shows WHICH
    input patterns maximally excite it. If neurons have specialised, each
    panel should highlight different digit structures.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    model.eval()
    layer = model.hidden

    # Collect all test activations and inputs
    all_x, all_act = [], []
    for x, _y in test_loader:
        x = x.to(DEVICE)
        act = layer(x)                           # (batch, num_neurons)
        all_x.append(x.cpu())
        all_act.append(act.cpu())
    all_x = torch.cat(all_x, dim=0)              # (N, in_dim)
    all_act = torch.cat(all_act, dim=0)           # (N, num_neurons)

    num_neurons = all_act.shape[1]
    n_show = min(num_neurons, max_neurons)
    nrows = int(np.ceil(n_show / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 2.0))
    axes = np.array(axes).reshape(-1)

    for i in range(n_show):
        # Find top-k inputs that most excite this neuron
        _, top_indices = all_act[:, i].topk(min(topk, all_act.shape[0]))
        # Average those inputs to get the neuron's "preferred stimulus"
        preferred = all_x[top_indices].mean(dim=0).numpy().reshape(resolution, resolution)
        vmax = max(np.abs(preferred).max(), 1e-8)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        axes[i].imshow(preferred, cmap="RdBu_r", norm=norm, interpolation="nearest")
        axes[i].set_title(f"neuron {i}", fontsize=8)
        axes[i].axis("off")

    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir(subdir), fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def run_and_visualize(arch, res, nc, hidden, epochs, local_weight=1.0,
                      div_weight=0.0, schedule="interleaved", subdir="visualize"):
    """Train a single model and produce both decoder-weight and activation
    visualizations.  Returns the trained model for further inspection."""
    tr, te, in_dim = get_loaders(resolution=res, num_classes=nc)
    xb = next(iter(te))[0].to(DEVICE)
    model = HiddenLayerClassifier(in_dim, hidden, nc, arch=arch).to(DEVICE)

    if schedule == "interleaved":
        run_interleaved(model, tr, te, xb, epochs, local_weight, div_weight)
    else:
        run_sequential(model, tr, te, xb, epochs // 2, epochs - epochs // 2,
                       div_weight=div_weight, finetune=False)

    # Decoder weight fields — what each neuron reconstructs
    plot_neuron_decoders(
        model, res,
        fname=f"decoder_fields_{arch}.png",
        title=f"Decoder weight fields — {arch} (what each neuron reconstructs)",
        subdir=subdir)

    # Activation maps — what inputs each neuron responds to most
    plot_neuron_activations(
        model, te, res,
        fname=f"activation_maps_{arch}.png",
        title=f"Preferred stimuli — {arch} (inputs that maximally excite each neuron)",
        subdir=subdir)

    # Also print the final metrics
    m = metrics(model, te, xb)
    ds = "n/a" if m["dec_sim"] < 0 else f"{m['dec_sim']:.3f}"
    rm = "n/a" if m["recon_mse"] < 0 else f"{m['recon_mse']:.4f}"
    print(f"  [{arch}] final: acc={m['acc']:.1f}%  eff_rank={m['eff_rank']:.1f}  "
          f"dec_sim={ds}  act_corr={m['act_corr']:.3f}  recon_mse={rm}")
    return model

# ==============================================================================
# 4. High-Level Experiment Orchestrations
# ==============================================================================
def exp_curves(seeds, epochs, res=14, nc=10, hidden=128, arch="static"):
    """Per-epoch curves: interleaved, sequential(frozen), sequential(fixed+pen).
    arch: "static" or "spiking" — selects the hidden-layer neuron model."""
    tr, te, in_dim = get_loaders(resolution=res, num_classes=nc)
    xb = next(iter(te))[0].to(DEVICE)
    print(f"exp_curves [{arch}]: interleaved ...")
    inter = multiseed(lambda s: run_interleaved(
        NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE), tr, te, xb, epochs, 1.0),
        seeds, in_dim, hidden, nc)
    print(f"exp_curves [{arch}]: sequential (frozen, no penalty) ...")
    seq = multiseed(lambda s: run_sequential(
        NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE), tr, te, xb,
        epochs // 2, epochs - epochs // 2, div_weight=0.0, finetune=False),
        seeds, in_dim, hidden, nc)
    print(f"exp_curves [{arch}]: sequential (frozen + diversity penalty) ...")
    seqp = multiseed(lambda s: run_sequential(
        NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE), tr, te, xb,
        epochs // 2, epochs - epochs // 2, div_weight=5.0, finetune=False),
        seeds, in_dim, hidden, nc)
    print(f"exp_curves [{arch}]: warm-up then interleaved ...")
    warm = multiseed(lambda s: run_warmup_interleaved(
        NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE), tr, te, xb,
        epochs // 2, epochs - epochs // 2, div_weight=5.0),
        seeds, in_dim, hidden, nc)
    curves = {"interleaved": inter, "sequential": seq,
              "sequential+penalty": seqp, "warmup+interleaved": warm}
    save_json(curves, f"curves_{arch}.json", subdir="curves")
    plot_curves(curves, "acc", "test accuracy (%)", f"curve_acc_{arch}.png",
                f"Test accuracy over training ({arch})", subdir="curves")
    plot_curves(curves, "eff_rank", "effective rank", f"curve_rank_{arch}.png",
                f"Neuron diversity (effective rank) over training ({arch})", subdir="curves")
    # Reconstruction accuracy over training. NN has no autoencoder, so its
    # recon_acc is -1 every epoch; plot_curves will just draw a flat -1 line for
    # it if NN is ever in the dict. For the NAN/ANN archs this rises toward 100%
    # as the autoencoders learn to reconstruct the input.
    plot_curves(curves, "recon_acc", "reconstruction accuracy (%)",
                f"curve_recon_{arch}.png",
                f"Reconstruction accuracy over training ({arch})", subdir="curves")
    return curves


def exp_recon_curves(seeds, epochs, res=14, nc=10, hidden=128,
                     archs=("ann", "static", "spiking")):
    """Dedicated reconstruction-quality experiment.

    Trains each autoencoder architecture (interleaved) and records, per epoch,
    BOTH reconstruction measures that metrics() now emits:
        recon_acc : reconstruction ACCURACY (%) — 1 - mse/var(input), RISES toward
                    100% as the autoencoders improve.
        recon_mse : reconstruction MSE — the raw error, FALLS toward 0.
    Produces two overlay figures (one per measure) with all architectures on the
    same axes, plus a JSON of the mean/SEM curves.

    NN is intentionally excluded by default: it has no autoencoder, so its
    reconstruction is undefined (metrics() returns -1). Pass archs=(...,"nn") only
    if you explicitly want the flat n/a line drawn for contrast.

    This is the minimal, purpose-built run for the reconstruction curves — it does
    no sweeps, schedules, penalties, or visualizations.
    """
    tr, te, in_dim = get_loaders(resolution=res, num_classes=nc)
    xb = next(iter(te))[0].to(DEVICE)

    curves = {}
    for arch in archs:
        print(f"exp_recon_curves [{arch}]: interleaved ...")
        curves[arch] = multiseed(lambda s: run_interleaved(
            NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE),
            tr, te, xb, epochs, local_weight=1.0),
            seeds, in_dim, hidden, nc)

    save_json(curves, "recon_curves.json", subdir="recon_curves")

    # Curve 1 — reconstruction ACCURACY (rises toward 100%)
    plot_curves(curves, "recon_acc", "reconstruction accuracy (%)",
                "recon_accuracy_curve.png",
                "Reconstruction accuracy over training", subdir="recon_curves")
    # Curve 2 — reconstruction MSE (falls toward 0)
    plot_curves(curves, "recon_mse", "reconstruction MSE",
                "recon_mse_curve.png",
                "Reconstruction MSE over training", subdir="recon_curves")

    # final-epoch summary to stdout
    print(f"\n  reconstruction curves (final epoch, hidden={hidden}):")
    print(f"  {'arch':10} {'recon_acc(%)':>13} {'recon_mse':>11}")
    for arch, agg in curves.items():
        ra = agg["recon_acc"]["mean"][-1]
        rm = agg["recon_mse"]["mean"][-1]
        ra_s = "n/a" if ra < 0 else f"{ra:.2f}"
        rm_s = "n/a" if rm < 0 else f"{rm:.4f}"
        print(f"  {arch:10} {ra_s:>13} {rm_s:>11}")
    return curves


def exp_schedule_compare(seeds, epochs, res=14, nc=10, hidden=128, arch="static"):
    """Focused three-way comparison of the training SCHEDULES, your hybrid idea
    vs the two it sits between:
        interleaved        - coin-flip task/reconstruction every batch (baseline)
        sequential         - pure sequential: AE-only phase, then FROZEN task phase
        warmup+interleaved - the hybrid: penalised AE warm-up, THEN interleaved
    Pure sequential here is the frozen, no-penalty variant (the collapse case),
    so the plot shows the hybrid against both the healthy baseline and the failure
    mode. Output goes to its own 'schedule_compare' subdir, separate from the
    four-line curves plot, so this comparison reads cleanly on its own.
    """
    tr, te, in_dim = get_loaders(resolution=res, num_classes=nc)
    xb = next(iter(te))[0].to(DEVICE)

    print(f"exp_schedule_compare [{arch}]: interleaved ...")
    inter = multiseed(lambda s: run_interleaved(
        NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE), tr, te, xb, epochs, 1.0),
        seeds, in_dim, hidden, nc)
    print(f"exp_schedule_compare [{arch}]: pure sequential (frozen) ...")
    seq = multiseed(lambda s: run_sequential(
        NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE), tr, te, xb,
        epochs // 2, epochs - epochs // 2, div_weight=0.0, finetune=False),
        seeds, in_dim, hidden, nc)
    print(f"exp_schedule_compare [{arch}]: hybrid (warm-up then interleaved) ...")
    warm = multiseed(lambda s: run_warmup_interleaved(
        NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE), tr, te, xb,
        epochs // 2, epochs - epochs // 2, div_weight=5.0),
        seeds, in_dim, hidden, nc)

    curves = {"interleaved": inter, "pure sequential": seq, "hybrid (warm-up+interleaved)": warm}
    save_json(curves, f"schedule_compare_{arch}.json", subdir="schedule_compare")
    plot_curves(curves, "acc", "test accuracy (%)", f"schedule_acc_{arch}.png",
                f"Schedule comparison: accuracy over training ({arch})", subdir="schedule_compare")
    plot_curves(curves, "eff_rank", "effective rank", f"schedule_rank_{arch}.png",
                f"Schedule comparison: neuron diversity over training ({arch})", subdir="schedule_compare")
    # concise final-epoch summary to the log
    print(f"\n  schedule comparison ({arch}, final epoch):")
    print(f"  {'schedule':30} {'acc':>8} {'eff_rank':>9}")
    for name, agg in curves.items():
        print(f"  {name:30} {agg['acc']['mean'][-1]:>8.2f} {agg['eff_rank']['mean'][-1]:>9.1f}")
    return curves


def exp_sweep(kind, values, seeds, epochs, res=14, nc=10, hidden=128, arch="static",
              penalty="decoder"):
    """Sweep div_weight (kind='div') or local_weight (kind='local').
    Records FINAL accuracy and eff-rank (mean+sem over seeds) per value.
    arch: "static" or "spiking".  penalty: which diversity penalty to apply when
    kind='div' ("decoder" = weight decorrelation, "activation" = Barlow-Twins)."""
    tr, te, in_dim = get_loaders(resolution=res, num_classes=nc)
    xb = next(iter(te))[0].to(DEVICE)
    acc_m, acc_s, rank_m, rank_s = [], [], [], []
    for v in values:
        print(f"exp_sweep[{kind}/{arch}/{penalty}]: value={v} ...")
        def run(s):
            m = NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE)
            if kind == "div":   # interleaved, vary diversity penalty (chosen formulation)
                return run_interleaved(m, tr, te, xb, epochs, 1.0, div_weight=v,
                                       penalty=penalty)
            else:               # vary local (reconstruction) weight, no penalty
                return run_interleaved(m, tr, te, xb, epochs, v, div_weight=0.0)
        agg = multiseed(run, seeds, in_dim, hidden, nc)
        acc_m.append(agg["acc"]["mean"][-1]);  acc_s.append(agg["acc"]["sem"][-1])
        rank_m.append(agg["eff_rank"]["mean"][-1]); rank_s.append(agg["eff_rank"]["sem"][-1])
    res_obj = {"values": values, "acc_mean": acc_m, "acc_sem": acc_s,
               "rank_mean": rank_m, "rank_sem": rank_s, "penalty": penalty}
    sweep_subdir = f"{kind}_sweep"
    # tag div-sweep files by penalty so "decoder" and "activation" runs coexist
    tag = f"{arch}_{penalty}" if kind == "div" else arch
    save_json(res_obj, f"sweep_{kind}_{tag}.json", subdir=sweep_subdir)
    pen_label = f", {penalty} penalty" if kind == "div" else ""
    xlabel = "diversity weight" if kind == "div" else "local (reconstruction) weight"
    plot_sweep(values, res_obj, xlabel, f"sweep_{kind}_{tag}.png",
              f"Final accuracy & neuron diversity vs {xlabel} ({arch}{pen_label})", subdir=sweep_subdir)
    return res_obj


def _plot_penalty_overlay(values, series, ylabel, fname, title, subdir=None):
    """Overlay one line per schedule: y (with SEM) vs decoder-penalty strength.

    series maps schedule_name -> {"mean": [...], "sem": [...]} over `values`.
    Used for both reconstruction-accuracy-vs-penalty and test-accuracy-vs-penalty.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    markers = {"interleaved": "o", "hybrid": "s", "sequential": "^"}
    for name, s in series.items():
        ax.errorbar(values, s["mean"], yerr=s["sem"], marker=markers.get(name, "o"),
                    capsize=3, label=name)
    ax.set_xlabel("decoder-penalty strength (div_weight)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(title="schedule")
    fig.tight_layout()
    path = os.path.join(outdir(subdir), fname)
    plt.savefig(path, dpi=130); plt.close()
    print(f"  saved {path}")


def exp_recon_penalty_sweep(seeds, epochs, values=(0, 0.5, 1, 2, 5, 10, 20),
                            res=14, nc=10, hidden=128,
                            archs=("static", "spiking", "ann")):
    """Decoder-penalty sweep vs FINAL reconstruction accuracy AND test accuracy,
    across all three training schedules.

    For each architecture and each schedule (interleaved / hybrid / sequential),
    sweeps the decoder-diversity penalty strength and records the final-epoch
    reconstruction accuracy (recon_acc, %), reconstruction MSE (recon_mse), and
    test accuracy (acc, %), each with SEM over seeds.

    The decoder penalty pushes per-neuron decoder vectors apart, which fights the
    reconstruction objective — so this reveals the tradeoff: how much reconstruction
    (and by extension task accuracy) you give up as you force more neuron diversity,
    and whether that cost depends on WHEN the penalty is applied (constantly, in a
    warm-up only, or throughout a frozen AE phase).

    ANN is included as a CONTROL: it has one shared decoder (not per-neuron), so
    the decoder-orthogonality penalty acts on input-indexed rows rather than neurons
    — a useful baseline for whether any effect is specific to the NAN structure.

    Produces, per architecture: a reconstruction-accuracy-vs-penalty figure and a
    test-accuracy-vs-penalty figure (one line per schedule), plus a combined JSON.
    """
    penalty = "decoder"
    values = list(values)
    tr, te, in_dim = get_loaders(resolution=res, num_classes=nc)
    xb = next(iter(te))[0].to(DEVICE)
    p1 = epochs // 2                       # phase-1 epochs for hybrid/sequential
    p2 = epochs - p1                       # phase-2 epochs

    def schedule_runners(v):
        """Return {schedule_name: run_fn(seed)} at penalty strength v."""
        return {
            "interleaved": lambda s: run_interleaved(
                NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE),
                tr, te, xb, epochs, local_weight=1.0,
                div_weight=v, penalty=penalty),
            "hybrid": lambda s: run_warmup_interleaved(
                NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE),
                tr, te, xb, p1, p2, div_weight=v, penalty=penalty),
            "sequential": lambda s: run_sequential(
                NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE),
                tr, te, xb, p1, p2, div_weight=v, penalty=penalty, finetune=False),
        }

    all_results = {}
    for arch in archs:
        # per-schedule accumulators of final-epoch metrics across the sweep
        recon = {sch: {"mean": [], "sem": []} for sch in ("interleaved", "hybrid", "sequential")}
        testacc = {sch: {"mean": [], "sem": []} for sch in ("interleaved", "hybrid", "sequential")}
        mse = {sch: {"mean": [], "sem": []} for sch in ("interleaved", "hybrid", "sequential")}
        for v in values:
            runners = schedule_runners(v)
            for sch, run_fn in runners.items():
                print(f"exp_recon_penalty_sweep [{arch}/{sch}]: div_weight={v} ...")
                agg = multiseed(run_fn, seeds, in_dim, hidden, nc)
                recon[sch]["mean"].append(agg["recon_acc"]["mean"][-1])
                recon[sch]["sem"].append(agg["recon_acc"]["sem"][-1])
                mse[sch]["mean"].append(agg["recon_mse"]["mean"][-1])
                mse[sch]["sem"].append(agg["recon_mse"]["sem"][-1])
                testacc[sch]["mean"].append(agg["acc"]["mean"][-1])
                testacc[sch]["sem"].append(agg["acc"]["sem"][-1])

        all_results[arch] = {"values": values, "recon_acc": recon,
                             "recon_mse": mse, "acc": testacc}
        save_json(all_results[arch], f"recon_penalty_sweep_{arch}.json",
                  subdir="recon_penalty_sweep")

        ctrl = " [control]" if arch == "ann" else ""
        _plot_penalty_overlay(
            values, recon, "reconstruction accuracy (%)",
            f"recon_acc_vs_penalty_{arch}.png",
            f"Reconstruction accuracy vs decoder penalty ({arch}{ctrl})",
            subdir="recon_penalty_sweep")
        _plot_penalty_overlay(
            values, testacc, "test accuracy (%)",
            f"test_acc_vs_penalty_{arch}.png",
            f"Test accuracy vs decoder penalty ({arch}{ctrl})",
            subdir="recon_penalty_sweep")

        # concise stdout summary
        print(f"\n  {arch}{ctrl} — final metrics vs decoder penalty:")
        print(f"  {'dw':>5} {'schedule':12} {'recon_acc%':>11} {'test_acc%':>10} {'recon_mse':>10}")
        for i, v in enumerate(values):
            for sch in ("interleaved", "hybrid", "sequential"):
                ra = recon[sch]["mean"][i]; ta = testacc[sch]["mean"][i]; rm = mse[sch]["mean"][i]
                ra_s = "n/a" if ra < 0 else f"{ra:.2f}"
                rm_s = "n/a" if rm < 0 else f"{rm:.4f}"
                print(f"  {v:>5} {sch:12} {ra_s:>11} {ta:>10.2f} {rm_s:>10}")
    return all_results


def exp_seq_variants(seeds, epochs, res=14, nc=10, hidden=128, arch="static"):
    """The 2x2: {freeze, finetune} x {no penalty, penalty}. Final metrics.
    arch: "static" or "spiking"."""
    tr, te, in_dim = get_loaders(resolution=res, num_classes=nc)
    xb = next(iter(te))[0].to(DEVICE)
    grid = {}
    for finetune in (False, True):
        for dw in (0.0, 5.0):
            name = f"{'finetune' if finetune else 'freeze'}+{'pen' if dw>0 else 'nopen'}"
            print(f"exp_seq_variants [{arch}]: {name} ...")
            agg = multiseed(lambda s: run_sequential(
                NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE), tr, te, xb,
                epochs // 2, epochs - epochs // 2, div_weight=dw, finetune=finetune),
                seeds, in_dim, hidden, nc)
            grid[name] = {"acc": agg["acc"]["mean"][-1],
                          "acc_sem": agg["acc"]["sem"][-1],
                          "eff_rank": agg["eff_rank"]["mean"][-1]}
    save_json(grid, f"seq_variants_{arch}.json", subdir="seq_variants")
    print(f"\n  2x2 sequential variants ({arch}, final):")
    print(f"  {'variant':20} {'acc':>8} {'+-sem':>7} {'eff_rank':>9}")
    for k, v in grid.items():
        print(f"  {k:20} {v['acc']:>8.2f} {v['acc_sem']:>7.2f} {v['eff_rank']:>9.1f}")
    return grid


def plot_schedule_grid(curve_dict, arch, div_weight, subdir=None):
    """3×2 grid figure: rows = schedules, cols = accuracy / effective rank.

    Each subplot shows a per-epoch curve with SEM shading. For sequential and
    hybrid schedules, a vertical dashed line marks the Phase-1 → Phase-2
    transition (derived from the 'phase' markers in the aggregated data).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schedules = list(curve_dict.keys())          # guaranteed insertion order (3.7+)
    metrics_cfg = [
        ("acc",      "Test accuracy (%)",  "tab:blue"),
        ("eff_rank", "Effective rank",     "tab:red"),
    ]

    fig, axes = plt.subplots(len(schedules), len(metrics_cfg),
                             figsize=(12, 3.5 * len(schedules)),
                             sharex=True, squeeze=False)

    for row, sched_name in enumerate(schedules):
        agg = curve_dict[sched_name]
        phases = agg.get("phase", [])

        for col, (metric_key, ylabel, colour) in enumerate(metrics_cfg):
            ax = axes[row][col]
            m = np.array(agg[metric_key]["mean"])
            s = np.array(agg[metric_key]["sem"])
            ep = np.arange(1, len(m) + 1)

            ax.plot(ep, m, color=colour, linewidth=2)
            ax.fill_between(ep, m - s, m + s, color=colour, alpha=0.2)

            # Phase-transition marker: find the last Phase-1 epoch
            if phases:
                phase_arr = np.array(phases)
                p1_epochs = np.where(phase_arr == 1)[0]
                if len(p1_epochs) > 0 and len(p1_epochs) < len(phase_arr):
                    boundary = p1_epochs[-1] + 1.5     # between last P1 and first P2
                    ax.axvline(boundary, color="grey", linestyle="--",
                               linewidth=1, alpha=0.7)
                    ax.text(boundary, ax.get_ylim()[1], " P1→P2",
                            fontsize=7, color="grey", va="top")

            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(ylabel, fontsize=11, fontweight="bold")
            if row == len(schedules) - 1:
                ax.set_xlabel("Epoch", fontsize=9)

            # Row label on left column
            if col == 0:
                ax.annotate(sched_name, xy=(0, 0.5),
                            xytext=(-ax.yaxis.labelpad - 10, 0),
                            xycoords=ax.yaxis.label, textcoords="offset points",
                            fontsize=10, fontweight="bold", ha="right", va="center",
                            rotation=90)

    fig.suptitle(f"Schedule comparison with decoder penalty "
                 f"(div_weight={div_weight}, {arch})",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    dw_tag = str(div_weight).replace(".", "p")
    path = os.path.join(outdir(subdir), f"penalised_grid_{arch}_dw{dw_tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def exp_penalised_schedule_compare(seeds, epochs, div_weight, res=14, nc=10,
                                   hidden=128, arch="static"):
    """Compare schedules at the SAME decoder penalty strength.

    Three schedules, all using penalty='decoder' at the given div_weight:
        interleaved        — coin-flip task/AE every batch
        sequential         — AE-only Phase 1 (penalised), then task Phase 2 (finetune)
        hybrid             — penalised AE warm-up, then full interleaved

    Output files are tagged with the div_weight so multiple runs (e.g. dw=0, 2, 5)
    can coexist in the same directory.
    """
    tr, te, in_dim = get_loaders(resolution=res, num_classes=nc)
    xb = next(iter(te))[0].to(DEVICE)
    dw_tag = str(div_weight).replace(".", "p")
    sub = "penalised_compare"

    print(f"exp_penalised_schedule_compare [{arch}, dw={div_weight}]: interleaved ...")
    inter = multiseed(lambda s: run_interleaved(
        NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE),
        tr, te, xb, epochs, local_weight=1.0, div_weight=div_weight,
        penalty="decoder"),
        seeds, in_dim, hidden, nc)

    print(f"exp_penalised_schedule_compare [{arch}, dw={div_weight}]: sequential (finetune) ...")
    seq = multiseed(lambda s: run_sequential(
        NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE),
        tr, te, xb, epochs // 2, epochs - epochs // 2,
        div_weight=div_weight, penalty="decoder", finetune=True),
        seeds, in_dim, hidden, nc)

    print(f"exp_penalised_schedule_compare [{arch}, dw={div_weight}]: hybrid (warm-up + interleaved) ...")
    warm = multiseed(lambda s: run_warmup_interleaved(
        NAN_Classifier(in_dim, hidden, nc, arch=arch).to(DEVICE),
        tr, te, xb, epochs // 2, epochs - epochs // 2,
        local_weight=1.0, div_weight=div_weight, penalty="decoder"),
        seeds, in_dim, hidden, nc)

    curves = {"interleaved": inter, "sequential (finetune)": seq,
              "hybrid (warm-up+interleaved)": warm}

    # ---- save JSON ----
    save_json(curves, f"penalised_compare_{arch}_dw{dw_tag}.json", subdir=sub)

    # ---- overlay line plots (accuracy and rank, same style as existing) ----
    plot_curves(curves, "acc", "test accuracy (%)",
                f"penalised_acc_{arch}_dw{dw_tag}.png",
                f"Accuracy over training — decoder penalty dw={div_weight} ({arch})",
                subdir=sub)
    plot_curves(curves, "eff_rank", "effective rank",
                f"penalised_rank_{arch}_dw{dw_tag}.png",
                f"Neuron diversity over training — decoder penalty dw={div_weight} ({arch})",
                subdir=sub)

    # ---- the 3×2 grid figure ----
    plot_schedule_grid(curves, arch, div_weight, subdir=sub)

    # ---- final-epoch summary to stdout ----
    print(f"\n  penalised schedule comparison ({arch}, dw={div_weight}, final epoch):")
    print(f"  {'schedule':35} {'acc':>8} {'eff_rank':>9}")
    for name, agg in curves.items():
        print(f"  {name:35} {agg['acc']['mean'][-1]:>8.2f} "
              f"{agg['eff_rank']['mean'][-1]:>9.1f}")
    return curves