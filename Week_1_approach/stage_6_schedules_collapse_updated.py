"""
================================================================================
stage6_schedules_and_collapse.py
--------------------------------------------------------------------------------
Directly addresses the supervisor's email. Two additions on top of stage5:

  (A) SEQUENTIAL training schedule. The email: "In the paper they are
      interleaved, while you could train them sequentially (first the
      autoencoders, and then the task learning)." We implement BOTH schedules
      behind one switch so they can be compared head to head:
          - "interleaved" : coin-flip task/AE per batch (the paper's style)
          - "sequential"  : Phase 1 trains ONLY the autoencoders to convergence,
                            then Phase 2 freezes them and trains ONLY the task.

  (B) NEURON-COLLAPSE DIAGNOSTIC. The email: "all autoencoders in the hidden
      layer learn the same set of characteristics, which would not be good...
      but it is not clear whether this happens." So we MEASURE it, three ways:
          1. pairwise cosine similarity of the per-neuron decoder weight vectors
          2. pairwise correlation of the neurons' activations over a data batch
          3. effective rank of the activation matrix (how many neurons are
             doing genuinely independent work)
      A collapsed layer => decoders near-identical, activations highly
      correlated, effective rank ~1. A healthy layer => low similarity, rank ~H.

Reuses NeuronAELayer from the earlier stages (inlined here so the file is
standalone; import it in your repo instead).

Author scaffolding: Jamal Idrissou
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# ==============================================================================
# REUSED: the neuron-AE layer (unchanged from stage3_4 / stage5)
# ==============================================================================
class NeuronAELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.encoder = nn.Linear(in_features, out_features)
        self.dec_w = nn.Parameter(torch.empty(out_features, in_features))
        self.dec_b = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.dec_w, a=5 ** 0.5)

    def forward(self, x):
        return torch.sigmoid(self.encoder(x))

    def local_loss(self, x):
        a = torch.sigmoid(self.encoder(x))
        x_hat = a.unsqueeze(-1) * self.dec_w.unsqueeze(0) + self.dec_b.unsqueeze(0)
        return F.mse_loss(x_hat, x.unsqueeze(1).expand_as(x_hat))


class NAN_Classifier(nn.Module):
    """Neuron-AE hidden layer + linear classification head."""
    def __init__(self, in_dim, hidden=128, num_classes=10):
        super().__init__()
        self.hidden = NeuronAELayer(in_dim, hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x, want_local=False):
        local = self.hidden.local_loss(x) if want_local else None
        logits = self.head(self.hidden(x))
        return (logits, local) if want_local else logits


# ==============================================================================
# Anti-collapse mechanisms (creative direction #1)
# ==============================================================================
def diversity_penalty(layer):
    """
    Optional regulariser that PUSHES neurons apart: penalise the mean squared
    off-diagonal cosine similarity of decoder weights. Add lambda*this to the
    autoencoding-cycle loss to actively discourage collapse. Returns a scalar.
    """
    Wn = F.normalize(layer.dec_w, dim=1)
    sim = Wn @ Wn.t()
    H = sim.size(0)
    off = sim[~torch.eye(H, dtype=torch.bool, device=layer.dec_w.device)]
    return (off ** 2).mean()


# ==============================================================================
# (A) TWO TRAINING SCHEDULES behind one switch
# ==============================================================================
def train_interleaved(model, loader, epochs, local_weight, device, div_weight=0.0):
    """Paper-style: each batch is randomly a task OR an autoencoding cycle."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if local_weight == 0 or torch.rand(1).item() < 0.5:        # task
                opt.zero_grad()
                F.cross_entropy(model(x), y).backward()
                opt.step()
            else:                                                      # AE
                opt.zero_grad()
                _, local_loss = model(x, want_local=True)
                
                # --- APPLY DIVERSITY PENALTY HERE ---
                div_loss = diversity_penalty(model.hidden) if div_weight > 0 else 0.0
                total_ae_loss = (local_weight * local_loss) + (div_weight * div_loss)
                
                total_ae_loss.backward()
                opt.step()
    return model


def train_sequential(model, loader, ae_epochs, task_epochs, device, div_weight=0.0):
    """
    Supervisor's suggested variant: FIRST train the autoencoders alone, THEN
    freeze them and train the task head.
    """
    # ---- Phase 1: autoencoders only ----------------------------------------
    ae_params = list(model.hidden.parameters())
    opt_ae = torch.optim.Adam(ae_params, lr=1e-3)
    for _ in range(ae_epochs):
        for x, _y in loader:
            x = x.to(device)
            opt_ae.zero_grad()
            
            local_loss = model.hidden.local_loss(x)
            
            # --- APPLY DIVERSITY PENALTY HERE ---
            div_loss = diversity_penalty(model.hidden) if div_weight > 0 else 0.0
            total_ae_loss = local_loss + (div_weight * div_loss)
            
            total_ae_loss.backward()
            opt_ae.step()

    # ---- freeze the AE-trained hidden layer --------------------------------
    for p in model.hidden.parameters():
        p.requires_grad_(False)

    # ---- Phase 2: task head only, on the frozen representation -------------
    opt_task = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    for _ in range(task_epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt_task.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt_task.step()
    return model


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
    return 100.0 * correct / total


# ==============================================================================
# (B) NEURON-COLLAPSE DIAGNOSTICS  — "do all neurons learn the same thing?"
# ==============================================================================
@torch.no_grad()
def decoder_similarity(layer):
    W = layer.dec_w                                  # (H, in)
    Wn = F.normalize(W, dim=1)                        # unit vectors per neuron
    sim = Wn @ Wn.t()                                 # (H, H) cosine sims
    H = sim.size(0)
    off_diag = sim[~torch.eye(H, dtype=torch.bool, device=W.device)]
    return off_diag.abs().mean().item()


@torch.no_grad()
def activation_redundancy(layer, x):
    a = layer(x)                                      # (B, H) activations
    a = a - a.mean(0, keepdim=True)                   # centre per neuron
    # pairwise correlation
    std = a.std(0, keepdim=True) + 1e-8
    z = a / std
    corr = (z.t() @ z) / z.size(0)                    # (H, H)
    H = corr.size(0)
    off = corr[~torch.eye(H, dtype=torch.bool, device=x.device)]
    mean_abs_corr = off.abs().mean().item()
    # effective rank
    sv = torch.linalg.svdvals(a)                      # singular values
    p = sv / sv.sum()
    entropy = -(p * (p + 1e-12).log()).sum()
    eff_rank = torch.exp(entropy).item()
    return mean_abs_corr, eff_rank, H


def collapse_report(model, x, tag=""):
    sim = decoder_similarity(model.hidden)
    corr, rank, H = activation_redundancy(model.hidden, x)
    print(f"  [{tag}] decoder cos-sim={sim:.3f} | act |corr|={corr:.3f} | "
          f"eff.rank={rank:.1f}/{H}")
    return sim, corr, rank


# ==============================================================================
# Synthetic-or-real MNIST loader (same fallback approach as stage5)
# ==============================================================================
def get_loaders(resolution=14, num_classes=10, batch_size=128, root="./data"):
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.Resize((resolution, resolution)),
                             transforms.ToTensor()])
    def prep(ds):
        X, y = [], []
        for img, lab in ds:
            if lab < num_classes:
                X.append(img.view(-1)); y.append(lab)
        return TensorDataset(torch.stack(X), torch.tensor(y))
    try:
        tr = prep(datasets.MNIST(root, train=True, download=True, transform=tf))
        te = prep(datasets.MNIST(root, train=False, download=True, transform=tf))
    except Exception as e:
        print(f"[get_loaders] real MNIST unavailable ({type(e).__name__}); synthetic stand-in.")
        d = resolution * resolution
        def synth(n):
            y = torch.randint(0, num_classes, (n,))
            X = (torch.rand(n, d) * 0.1 + (y.float().unsqueeze(1) / num_classes)).clamp(0, 1)
            return TensorDataset(X, y)
        tr, te = synth(2000), synth(1000)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(te, batch_size=batch_size, shuffle=False),
            resolution * resolution)


# ==============================================================================
# RUNNER: interleaved vs sequential vs baseline, each with a collapse report
# ==============================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    tr, te, in_dim = get_loaders(resolution=14, num_classes=10)
    xb = next(iter(te))[0].to(device)        # a batch for the diagnostics

    print("=== BASELINE (traditional model, no autoencoding) ===")
    base = NAN_Classifier(in_dim).to(device)
    train_interleaved(base, tr, epochs=3, local_weight=0.0, device=device)
    print(f"  test acc: {accuracy(base, te, device):.2f}%")

    print("\n=== INTERLEAVED neuron-AE (paper style) ===")
    inter = NAN_Classifier(in_dim).to(device)
    train_interleaved(inter, tr, epochs=3, local_weight=1.0, device=device)
    print(f"  test acc: {accuracy(inter, te, device):.2f}%")
    collapse_report(inter, xb, tag="interleaved")

    print("\n=== SEQUENTIAL neuron-AE (AEs first, then frozen task) ===")
    seq = NAN_Classifier(in_dim).to(device)
    train_sequential(seq, tr, ae_epochs=3, task_epochs=3, device=device)
    print(f"  test acc: {accuracy(seq, te, device):.2f}%")
    collapse_report(seq, xb, tag="sequential")
    
    print("\n=== SEQUENTIAL neuron-AE (WITH DIVERSITY PENALTY) ===")
    seq_fixed = NAN_Classifier(in_dim).to(device)
    # Note the div_weight=5.0 here to forcefully push the neurons apart
    train_sequential(seq_fixed, tr, ae_epochs=3, task_epochs=3, device=device, div_weight=5.0)
    print(f"  test acc: {accuracy(seq_fixed, te, device):.2f}%")
    collapse_report(seq_fixed, xb, tag="sequential_fixed")

    print("\nNote: Compare 'sequential' against 'sequential_fixed'.")
    print("The diversity penalty should dramatically increase the effective rank ")
    print("and rescue the test accuracy.")