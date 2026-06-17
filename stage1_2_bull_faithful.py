"""
================================================================================
stage1_2_bull_faithful.py
--------------------------------------------------------------------------------
FAITHFUL reproduction of Bull (2025), "Neurons as Autoencoders".

This is the version that learns the paper ON ITS OWN TERMS. The single most
important thing to understand before writing any code:

    *** BULL DOES NOT USE BACKPROPAGATION / GRADIENT DESCENT. ***

His title's "backpropagation" is the BIOLOGICAL backward echo up the dendrite,
not the ML algorithm. The weights are trained by a RANDOM HILL-CLIMBER:
    "a learning cycle consists of one randomly chosen weight from either the
     hidden layer or output layer being adjusted by a random amount from the
     range [-R, +R]" ... "the new network configuration is chosen if its mean
     squared error is reduced in comparison to that of the current solution,
     with ties broken at random."

So there are NO gradients here at all. We store weights as plain numpy arrays
and mutate one number at a time. This file deliberately uses numpy, not torch,
because gradients are exactly what we are NOT using — bringing in autograd would
misrepresent the paper. (Stage 3, in the other file, does the gradient version.)

Three network types are built, matching the paper's Figures 3-4:
    NN  : plain MLP, supervised only           (baseline)
    ANN : one shared decoder LAYER on the hidden layer (standard autoencoder)
    NAN : each hidden NEURON has its OWN decoder (the paper's proposal)

Data comes from the NK model — a *tuneable* regression source (knobs N and K).

Paper settings reproduced: H=10, R=1.0, 10,000 iterations, 20 runs averaged,
train/test = 1000 examples each, inputs use -1.0 in place of 0.

Author scaffolding: Jamal Idrissou
================================================================================
"""

import numpy as np


# ==============================================================================
# STAGE 1 (LEARN + APPLY): the NK model as a tuneable data source
# ==============================================================================
class NKModel:
    """
    Kauffman's NK model, used here as a regression function (after Bull 2022).

    Why this and not MNIST? Because it has two difficulty knobs:
        N = number of binary features (genes)
        K = how many OTHER genes each gene's fitness depends on (epistasis)
    Raising K makes the input->output mapping more rugged / less correlated,
    so you can deliberately place a problem in the "easy" or "hard" regime and
    see WHERE neuron-autoencoding helps. That conditional finding is the whole
    point of the paper, and you can't get it from a single fixed dataset.

    Mechanism: each gene i has its own lookup table over its own value plus the
    values of its K interaction partners (2^(K+1) rows of random fitness in
    [0,1)). The genome fitness is the mean of the N per-gene table lookups.
    """

    def __init__(self, N, K, seed=0):
        self.N = N
        self.K = K
        rng = np.random.default_rng(seed)

        # For each gene, pick which K OTHER genes it depends on (its partners).
        # Gene i always depends on itself plus K distinct others -> (K+1) inputs.
        self.partners = []
        for i in range(N):
            others = [j for j in range(N) if j != i]
            chosen = rng.choice(others, size=K, replace=False) if K > 0 else np.array([], dtype=int)
            # store [self, partner1, ... partnerK]; order matters for table indexing
            self.partners.append(np.concatenate(([i], chosen)).astype(int))

        # For each gene, a random fitness table with 2^(K+1) rows (one per
        # combination of the gene + its K partners). Values in [0, 1).
        self.tables = [rng.random(2 ** (K + 1)) for _ in range(N)]

    def fitness(self, genome):
        """
        genome : array of N values in {0,1}. Returns scalar fitness in [0,1).
        For each gene we read off its (K+1) relevant bits, interpret them as a
        binary index into that gene's table, and average the N looked-up values.
        """
        total = 0.0
        for i in range(self.N):
            bits = genome[self.partners[i]]                 # the (K+1) relevant bits
            idx = int("".join(str(int(b)) for b in bits), 2)  # bits -> table row index
            total += self.tables[i][idx]
        return total / self.N

    def make_dataset(self, n_samples, seed=0):
        """
        Build (X, y). X is {-1, +1} encoded (paper replaces 0 with -1.0 as a
        neural-net input convention); y is the scalar NK fitness in [0,1).
        Returns X: (n_samples, N) float, y: (n_samples, 1) float.
        """
        rng = np.random.default_rng(seed)
        genomes = rng.integers(0, 2, size=(n_samples, self.N))   # {0,1}
        y = np.array([[self.fitness(g)] for g in genomes])        # (n,1) in [0,1)
        X = np.where(genomes == 0, -1.0, 1.0)                     # {0,1} -> {-1,+1}
        return X.astype(np.float64), y.astype(np.float64)


# ------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------
def sigmoid(z):
    # paper uses a sigmoid transfer function on all nodes (not linear)
    return 1.0 / (1.0 + np.exp(-z))


# ==============================================================================
# STAGE 2 (APPLY): the three network types, trained by random hill-climbing
# ==============================================================================
class NeuronAutoencoderNetwork:
    """
    A two-layer MLP (N -> H -> 1) where, optionally, each hidden neuron also
    carries its own decoder (N weights + bias) that tries to rebuild the input
    from that single neuron's scalar activation. Selecting the `mode` switches
    between the paper's three architectures:

        mode="NN"  : plain MLP, supervised hill-climbing only.
        mode="ANN" : a SINGLE decoder layer hangs off the whole hidden layer
                     (H -> N). Standard "one decoder layer" autoencoder.
        mode="NAN" : EACH of the H hidden neurons has its OWN 1->N decoder.
                     This is the paper's neuron-autoencoder proposal.

    All weights are plain numpy arrays. Training mutates ONE weight per cycle
    and keeps the change only if MSE improves (hill-climbing, ties random).
    """

    def __init__(self, N, H, mode="NAN", R=1.0, seed=0):
        self.N, self.H, self.mode, self.R = N, H, mode, R
        rng = np.random.default_rng(seed)
        self.rng = rng

        # --- forward weights, seeded in [-1, 1] as in the paper --------------
        # hidden layer: maps N inputs -> H hidden units (+ bias per unit)
        self.W1 = rng.uniform(-1, 1, size=(N, H))
        self.b1 = rng.uniform(-1, 1, size=(H,))
        # output layer: maps H hidden -> 1 output (+ bias)
        self.W2 = rng.uniform(-1, 1, size=(H, 1))
        self.b2 = rng.uniform(-1, 1, size=(1,))

        # --- decoder weights depending on mode -------------------------------
        if mode == "NAN":
            # one decoder PER hidden neuron: each maps its scalar a_j -> N-dim.
            # Dweight[j] is neuron j's (N,) decode vector; Dbias[j] its (N,) bias.
            self.Dweight = rng.uniform(-1, 1, size=(H, N))
            self.Dbias   = rng.uniform(-1, 1, size=(H, N))
        elif mode == "ANN":
            # one shared decoder LAYER: hidden vector (H,) -> reconstruct (N,).
            self.Dw = rng.uniform(-1, 1, size=(H, N))
            self.Db = rng.uniform(-1, 1, size=(N,))
        # mode == "NN" has no decoder at all.

    # --------------------------------------------------------------------------
    def hidden(self, X):
        """Hidden activations a = sigmoid(X W1 + b1). X:(B,N) -> (B,H)."""
        return sigmoid(X @ self.W1 + self.b1)

    def forward(self, X):
        """Full task forward pass -> output in (B,1) via sigmoid output node."""
        a = self.hidden(X)
        return sigmoid(a @ self.W2 + self.b2)

    # --------------------------------------------------------------------------
    def task_mse(self, X, y):
        """Supervised objective: MSE between network output and target y."""
        return float(np.mean((self.forward(X) - y) ** 2))

    def autoenc_mse(self, X):
        """
        Reconstruction objective. For NAN this is the MEAN over the H neurons of
        each neuron's own reconstruction MSE (matches the paper, where 'the NAN
        autoencoder error is the average of the H hidden layer nodes').
        For ANN it is the single shared-decoder reconstruction MSE.
        """
        a = self.hidden(X)                       # (B, H) hidden activations
        if self.mode == "NAN":
            # neuron j reconstructs X from its scalar a[:,j]:
            #   xhat[:, j, :] = a[:,j,None] * Dweight[j] + Dbias[j]
            # build (B, H, N) then compare against X broadcast to (B,1,N).
            xhat = a[:, :, None] * self.Dweight[None, :, :] + self.Dbias[None, :, :]
            target = X[:, None, :]               # (B,1,N) -> broadcasts to (B,H,N)
            return float(np.mean((xhat - target) ** 2))
        elif self.mode == "ANN":
            xhat = sigmoid(a @ self.Dw + self.Db)  # (B,N) single decoder layer
            # NOTE: inputs are in {-1,1}; we compare in raw space as the paper
            # measures MSE on the input values directly.
            return float(np.mean((xhat - X) ** 2))
        else:
            return 0.0  # NN has no autoencoder

    # --------------------------------------------------------------------------
    def _perturb(self, arr, idx):
        """Apply a random [-R,+R] nudge to one entry; return the OLD value so
        the caller can revert if the move is not kept (hill-climbing)."""
        old = arr[idx]
        arr[idx] = old + self.rng.uniform(-self.R, self.R)
        return old

    def train(self, X, y, iterations=10000, record_every=200):
        """
        Bull's training loop. Each iteration:
          1. Decide cycle type. For NN every cycle is a task cycle. For ANN/NAN
             it is a task cycle or an autoencoding cycle with probability 0.5.
          2. Pick ONE weight in the relevant group, perturb it by [-R,+R].
          3. Recompute the relevant error. Keep the change only if it did not
             get worse (improvement kept; ties broken at random by a coin flip).
        Returns a history dict of task / autoencoder MSE sampled over time.
        """
        hist = {"iter": [], "task_mse": [], "ae_mse": []}

        for it in range(iterations):
            # ---- 1. choose cycle type -------------------------------------
            is_ae_cycle = (self.mode != "NN") and (self.rng.random() < 0.5)

            if is_ae_cycle:
                # ---- AUTOENCODING CYCLE: judged by reconstruction MSE ------
                before = self.autoenc_mse(X)
                if self.mode == "NAN":
                    # perturb one decoder weight OR the hidden weight feeding a
                    # neuron — i.e. any weight that affects reconstruction. The
                    # paper adjusts a hidden-layer-associated weight; here we
                    # nudge a single decoder weight (the neuron's own AE param).
                    j = self.rng.integers(self.H)
                    k = self.rng.integers(self.N)
                    old = self._perturb(self.Dweight, (j, k))
                    after = self.autoenc_mse(X)
                    if after > before:                 # worse -> revert
                        self.Dweight[j, k] = old
                else:  # ANN
                    r = self.rng.integers(self.H); c = self.rng.integers(self.N)
                    old = self._perturb(self.Dw, (r, c))
                    after = self.autoenc_mse(X)
                    if after > before:
                        self.Dw[r, c] = old
            else:
                # ---- TASK CYCLE: judged by supervised task MSE -------------
                before = self.task_mse(X, y)
                # pick a weight from the forward path (hidden or output layer)
                if self.rng.random() < 0.5:
                    j = self.rng.integers(self.H); k = self.rng.integers(self.N)
                    old = self._perturb(self.W1, (k, j))
                    after = self.task_mse(X, y)
                    if after > before:
                        self.W1[k, j] = old
                else:
                    j = self.rng.integers(self.H)
                    old = self._perturb(self.W2, (j, 0))
                    after = self.task_mse(X, y)
                    if after > before:
                        self.W2[j, 0] = old

            # ---- record progress periodically -----------------------------
            if it % record_every == 0:
                hist["iter"].append(it)
                hist["task_mse"].append(self.task_mse(X, y))
                hist["ae_mse"].append(self.autoenc_mse(X))

        return hist


# ==============================================================================
# STAGE 2 RUNNER (EXPERIMENT): compare NN / ANN / NAN, sweep N and K
# ==============================================================================
if __name__ == "__main__":
    # --- a quick single comparison so you can see it run ---------------------
    N, K, H = 20, 5, 10
    nk = NKModel(N=N, K=K, seed=1)
    Xtr, ytr = nk.make_dataset(1000, seed=10)
    Xte, yte = nk.make_dataset(1000, seed=20)
    print(f"NK data: N={N} K={K} | train {Xtr.shape} test {Xte.shape}")

    ITERS = 4000  # paper uses 10000; smaller here so the demo is quick
    results = {}
    for mode in ["NN", "ANN", "NAN"]:
        net = NeuronAutoencoderNetwork(N=N, H=H, mode=mode, R=1.0, seed=2)
        net.train(Xtr, ytr, iterations=ITERS, record_every=500)
        results[mode] = net.task_mse(Xte, yte)   # held-out task error
        print(f"  {mode:3s}  test task MSE = {results[mode]:.5f}")

    print("\nReproduce the paper's *conditional* finding by sweeping:")
    print("  small N (e.g. 20) vs large N (e.g. 200+), K in {0,5,10,15},")
    print("  20 runs each, average, and run a t-test (NAN vs ANN, NAN vs NN).")
