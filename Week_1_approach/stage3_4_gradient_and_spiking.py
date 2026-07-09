"""
================================================================================
stage3_4_gradient_and_spiking.py
--------------------------------------------------------------------------------
STAGE 3: the gradient-descent reinterpretation of Bull's idea (scales to real
         data, connects to your existing MNIST repo).
STAGE 4: extending neuron-level local autoencoding to SPIKING neural networks
         (SNNs), which is the integration your internship asked for.

Read stage1_2_bull_faithful.py FIRST. That file is the paper as written
(hill-climbing, no gradients). THIS file is the practical engineering version:
we keep Bull's *structure* (each neuron reconstructs its own input, training
alternates task vs autoencoding cycles) but swap his random hill-climber for
gradient descent so it scales beyond tiny N and runs on GPUs.

WHY BOTH VERSIONS MATTER (the "how to learn how to learn" answer):
  - The faithful version teaches you what the paper actually claims and why the
    NK tuneability matters. You cannot critique or extend a method you have only
    seen through a reframing.
  - The gradient version is what you will actually experiment with at scale and
    what plugs into your skin-cancer-era PyTorch skills.
Showing your supervisor that you understand the difference is itself a result.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# STAGE 3 (APPLY): gradient-based neuron-autoencoder layer
# ==============================================================================
class NeuronAELayer(nn.Module):
    """
    Gradient-trained version of Bull's neuron-autoencoder. Identical idea to the
    faithful file, but the weights are nn.Parameters so autograd can train them.

    Each of the `out_features` neurons:
        encode : x (I,) -> a_j (scalar)         [the ordinary forward synapse]
        decode : a_j    -> x_hat_j (I,)         [its own dendritic 'echo']
        local loss_j = MSE(x_hat_j, x)

    Shapes (B batch, I in, N neurons): x (B,I) -> a (B,N), x_hat (B,N,I).
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.encoder = nn.Linear(in_features, out_features)      # forward synapse
        # per-neuron decoder: neuron j maps its scalar -> I-dim reconstruction
        self.dec_w = nn.Parameter(torch.empty(out_features, in_features))
        self.dec_b = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.dec_w, a=5 ** 0.5)

    def forward(self, x):
        # soma signal each neuron passes forward (this is the normal layer out)
        return torch.sigmoid(self.encoder(x))

    def local_loss(self, x):
        a = torch.sigmoid(self.encoder(x))                       # (B,N) codes
        # outer-product style broadcast -> one reconstruction per neuron
        x_hat = a.unsqueeze(-1) * self.dec_w.unsqueeze(0) + self.dec_b.unsqueeze(0)
        return F.mse_loss(x_hat, x.unsqueeze(1).expand_as(x_hat))


class GradientNAN(nn.Module):
    """A small regression MLP whose hidden layer is a NeuronAELayer.
    Returns the task prediction, and on request the hidden layer's local loss."""

    def __init__(self, in_dim, hidden=10, out_dim=1):
        super().__init__()
        self.hidden = NeuronAELayer(in_dim, hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, want_local=False):
        local = self.hidden.local_loss(x) if want_local else None
        a = self.hidden(x)
        y = torch.sigmoid(self.head(a))      # NK targets are in [0,1)
        return (y, local) if want_local else y


def train_gradient_nan(model, X, y, steps=3000, local_weight=1.0):
    """
    Alternating gradient version of Bull's two cycle types:
        task cycle  -> minimise supervised MSE (head + encoder get gradients)
        local cycle -> minimise neuron reconstruction MSE (encoder + decoder)
    We alternate on a coin flip per step, mirroring his 0.5 probability, but use
    one Adam optimiser and real gradients instead of single-weight hill-climbing.
    """
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(steps):
        if torch.rand(1).item() < 0.5:        # ---- task cycle ----
            opt.zero_grad()
            pred = model(X)
            loss = F.mse_loss(pred, y)
            loss.backward(); opt.step()
        else:                                  # ---- autoencoding cycle ----
            opt.zero_grad()
            _, local = model(X, want_local=True)
            (local_weight * local).backward(); opt.step()
    with torch.no_grad():
        return F.mse_loss(model(X), y).item()


# ==============================================================================
# STAGE 4 (LEARN -> APPLY -> EXPERIMENT): neuron-AE inside a SPIKING network
# ==============================================================================
"""
WHY SNNs FIT BULL'S IDEA UNUSUALLY WELL
---------------------------------------
Bull motivates neuron-AE biologically: a real neuron sends a forward spike down
its axon and a backward "echo" up its dendrites. SNNs are the ANN family that
actually models neurons as spiking units over time, so neuron-level local
learning is a more natural fit here than in a plain MLP. Two honest challenges:

  (1) Spikes are non-differentiable (a threshold crossing is a step function),
      so standard backprop doesn't apply directly. The field's standard fix is
      the SURROGATE GRADIENT: use the hard threshold on the forward pass but a
      smooth function's derivative on the backward pass. (See sources below.)

  (2) "Reconstruct the input" must be defined over TIME. A spiking neuron's
      output is a spike train, and its natural scalar summary is its firing
      rate (spikes / timesteps). So our neuron-AE decodes from that rate.

This makes the SNN neuron-AE a clean research question: does forcing each
spiking neuron to retain enough information to reconstruct its input (via its
firing rate) shape better representations, exactly as Bull found for MLPs?
"""


class SurrogateSpike(torch.autograd.Function):
    """
    Leaky-integrate-and-fire spike with a surrogate gradient.

    forward : hard threshold. Output is 1.0 if membrane potential v >= 0 (we
              pass v already shifted by the threshold), else 0.0 — a real spike.
    backward: pretend the step was a smooth fast-sigmoid so a usable gradient
              flows. This is the standard surrogate-gradient trick that makes
              SNNs trainable with autograd. grad ~ 1/(1+|v|*scale)^2.
    """
    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        scale = 10.0
        sg = 1.0 / (scale * torch.abs(v) + 1.0) ** 2   # surrogate derivative
        return grad_output * sg


spike_fn = SurrogateSpike.apply


class SpikingNeuronAELayer(nn.Module):
    """
    A layer of leaky-integrate-and-fire (LIF) neurons, where each neuron ALSO
    autoencodes the (time-averaged) input from its OWN firing rate.

    Over T timesteps each neuron:
        - integrates input current into a leaky membrane potential v,
        - emits a spike when v crosses threshold, then resets,
        - its firing RATE (mean spikes over T) is its scalar 'code' a_j,
        - a per-neuron decoder maps a_j -> reconstruction of the mean input.

    So this is Bull's neuron-AE with the scalar bottleneck being the neuron's
    firing rate rather than a static activation — the temporal analogue.
    """

    def __init__(self, in_features, out_features, T=20, beta=0.9, thresh=1.0):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.T = T            # number of simulation timesteps
        self.beta = beta      # membrane leak (0=no memory, 1=perfect integrator)
        self.thresh = thresh  # firing threshold

        self.fc = nn.Linear(in_features, out_features)   # input -> current
        # per-neuron decoder from firing rate (scalar) back to the input vector
        self.dec_w = nn.Parameter(torch.empty(out_features, in_features))
        self.dec_b = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.dec_w, a=5 ** 0.5)

    def simulate(self, x):
        """
        Run the LIF dynamics for T steps with a constant input current derived
        from x (rate-coding the static input). Returns:
            rate    : (B, N) mean firing rate per neuron  (the AE 'code')
            spikes  : (B, T, N) full spike train (kept for inspection/plots)
        """
        B = x.size(0)
        current = self.fc(x)                       # (B, N) constant drive
        v = torch.zeros(B, self.out_features, device=x.device)
        spike_sum = torch.zeros(B, self.out_features, device=x.device)
        spikes = []
        for _ in range(self.T):
            v = self.beta * v + current            # leaky integration
            s = spike_fn(v - self.thresh)          # spike if v >= thresh
            v = v * (1.0 - s)                      # hard reset where a spike fired
            spike_sum = spike_sum + s
            spikes.append(s)
        rate = spike_sum / self.T                  # (B, N) firing rate in [0,1]
        return rate, torch.stack(spikes, dim=1)

    def forward(self, x):
        rate, _ = self.simulate(x)
        return rate                                # passed forward to next layer

    def local_loss(self, x):
        """Neuron-AE loss: each neuron rebuilds the input from its firing rate."""
        rate, _ = self.simulate(x)                 # (B, N)
        x_hat = rate.unsqueeze(-1) * self.dec_w.unsqueeze(0) + self.dec_b.unsqueeze(0)
        return F.mse_loss(x_hat, x.unsqueeze(1).expand_as(x_hat))


class SpikingNAN(nn.Module):
    """Spiking regression network with a neuron-autoencoding hidden layer."""

    def __init__(self, in_dim, hidden=10, out_dim=1, T=20):
        super().__init__()
        self.hidden = SpikingNeuronAELayer(in_dim, hidden, T=T)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, want_local=False):
        local = self.hidden.local_loss(x) if want_local else None
        rate = self.hidden(x)            # firing rates -> readout
        y = torch.sigmoid(self.head(rate))
        return (y, local) if want_local else y


# ==============================================================================
# RUNNER: confirm both stages train on a toy regression target
# ==============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    # toy data standing in for the NK regression (import NKModel from stage 1/2
    # for the real thing). Inputs in {-1,1}, target in [0,1).
    B, I = 256, 20
    X = (torch.rand(B, I) > 0.5).float() * 2 - 1
    y = torch.sigmoid(X.sum(dim=1, keepdim=True) * 0.3)   # smooth-ish target

    print("=== STAGE 3: gradient neuron-AE (NAN) ===")
    g = GradientNAN(in_dim=I, hidden=10)
    mse_g = train_gradient_nan(g, X, y, steps=1500, local_weight=1.0)
    print(f"  final task MSE: {mse_g:.5f}")

    print("\n=== STAGE 4: spiking neuron-AE (Spiking NAN) ===")
    s = SpikingNAN(in_dim=I, hidden=10, T=20)
    opt = torch.optim.Adam(s.parameters(), lr=1e-2)
    for step in range(800):
        if torch.rand(1).item() < 0.5:                  # task cycle
            opt.zero_grad(); F.mse_loss(s(X), y).backward(); opt.step()
        else:                                            # autoencoding cycle
            opt.zero_grad(); s(X, want_local=True)[1].backward(); opt.step()
    with torch.no_grad():
        print(f"  final task MSE: {F.mse_loss(s(X), y).item():.5f}")
        rate, spk = s.hidden.simulate(X)
        print(f"  mean firing rate: {rate.mean().item():.3f} "
              f"(spike train shape {tuple(spk.shape)} = B x T x neurons)")
    print("\nExperiment next: sweep T (timesteps), beta (leak), and local_weight;")
    print("compare Spiking NAN vs a spiking baseline with no neuron-AE, NK-style.")
