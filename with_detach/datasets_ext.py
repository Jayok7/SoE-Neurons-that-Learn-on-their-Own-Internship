"""
datasets_ext.py — MNIST / EMNIST / Fashion-MNIST loaders for the Week 7 scripts.

Drop-in replacement for models.get_loaders. Same signature plus a `dataset`
argument, same return type (train_loader, test_loader, in_dim), so nothing else
in sweep_hidden.py or collapse_analysis.py changes.

--------------------------------------------------------------------------
WHY THIS EXISTS RATHER THAN EDITING models.get_loaders
--------------------------------------------------------------------------
models.get_loaders is what Weeks 4-6 ran against. Leaving it untouched means
any old script still reproduces its original numbers exactly. This module
defers to it for `fashion` so the Week 7 Fashion-MNIST results stay bit-identical.

--------------------------------------------------------------------------
THE THREE GOTCHAS, ALL HANDLED BELOW
--------------------------------------------------------------------------
1. EMNIST IMAGES ARE TRANSPOSED. The raw EMNIST files store each digit/letter
   rotated 90 degrees and mirrored, and torchvision does NOT correct this. Train
   on them uncorrected and everything still "works" -- accuracy looks plausible,
   nothing errors -- but every reconstruction figure is sideways and the
   decoder templates are meaningless. We transpose the last two axes.

2. EMNIST 'letters' LABELS START AT 1, not 0, so a 26-class head would need 27
   outputs and class 0 would never fire. We subtract 1 for that split only.

3. EMNIST IS BIG. `balanced` is 112,800 train images against Fashion-MNIST's
   60,000, and `byclass` is 697,932. The original prep loop iterated the
   torchvision Dataset one image at a time through a PIL transform, which costs
   minutes per run -- unacceptable across a 135-run sweep. The prep here works
   directly on the underlying uint8 tensor and is roughly two orders of
   magnitude faster.

--------------------------------------------------------------------------
AVAILABLE DATASETS
--------------------------------------------------------------------------
  fashion            10 classes,  60,000 train   (the Week 5-7 default)
  mnist              10 classes,  60,000 train   handwritten digits
  emnist-digits      10 classes, 240,000 train   digits, more data
  emnist-letters     26 classes, 124,800 train   letters only
  emnist-balanced    47 classes, 112,800 train   digits + letters, class-balanced
  emnist-bymerge     47 classes, 697,932 train   same classes, unbalanced
  emnist-byclass     62 classes, 697,932 train   case-sensitive, unbalanced

`emnist-balanced` is the one to reach for first: it is the natural
higher-class-count comparison to Fashion-MNIST, and being class-balanced it
avoids the confound that comes with byclass/bymerge, where accuracy can move
simply because the class priors changed.

--------------------------------------------------------------------------
USE
--------------------------------------------------------------------------
    python sweep_hidden.py --dataset mnist
    python sweep_hidden.py --dataset emnist-balanced
    python sweep_hidden.py --dataset emnist-letters --hidden_sizes 16 64 256

--classes now defaults to the dataset's own class count, so nothing needs to be
passed alongside --dataset. Passing it explicitly still subsets from class 0.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---- import bootstrap (same as the other Week 7 scripts) -------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_HERE,
                   os.path.join(_HERE, os.pardir, "Week_5"),
                   os.path.join(_HERE, os.pardir, "Week_4")):
    _candidate = os.path.abspath(_candidate)
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.append(_candidate)

try:
    from models import DATA_ROOT, get_loaders as _fashion_loaders
except ImportError:
    from nan.models import DATA_ROOT, get_loaders as _fashion_loaders


# ==============================================================================
# Registry
# ==============================================================================
# name -> (torchvision class name, EMNIST split or None, n_classes, label_offset)
DATASETS = {
    "fashion":         ("FashionMNIST", None,       10, 0),
    "mnist":           ("MNIST",        None,       10, 0),
    "emnist-digits":   ("EMNIST",       "digits",   10, 0),
    "emnist-letters":  ("EMNIST",       "letters",  26, 1),   # labels are 1..26
    "emnist-balanced": ("EMNIST",       "balanced", 47, 0),
    "emnist-bymerge":  ("EMNIST",       "bymerge",  47, 0),
    "emnist-byclass":  ("EMNIST",       "byclass",  62, 0),
}


def default_classes(dataset):
    """Natural class count for a dataset name."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}. "
                         f"Choose from: {', '.join(sorted(DATASETS))}")
    return DATASETS[dataset][2]


def is_emnist(dataset):
    return DATASETS[dataset][0] == "EMNIST"


# ==============================================================================
# Fast preparation straight from the raw tensors
# ==============================================================================
def _prep(ds, resolution, num_classes, transpose, label_offset):
    """(N, res*res) float in [0,1] and (N,) long labels, without a Python loop.

    torchvision exposes the raw uint8 images as `.data` and the labels as
    `.targets`, so the per-image PIL transform can be skipped entirely.
    """
    X = ds.data
    y = ds.targets
    if not torch.is_tensor(X):
        X = torch.as_tensor(X)
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    X = X.float().div_(255.0)
    y = y.long()

    if label_offset:
        y = y - label_offset

    if transpose:                      # EMNIST is stored rotated + mirrored
        X = X.transpose(-2, -1)

    keep = y < num_classes
    X, y = X[keep], y[keep]

    if X.shape[-1] != resolution or X.shape[-2] != resolution:
        X = F.interpolate(X.unsqueeze(1), size=(resolution, resolution),
                          mode="bilinear", align_corners=False).squeeze(1)

    return TensorDataset(X.reshape(X.shape[0], -1).contiguous(), y.contiguous())


# ==============================================================================
# The loader
# ==============================================================================
def get_loaders(dataset="fashion", resolution=28, num_classes=None,
                batch_size=128, root=None, verbose=True):
    """Train/test DataLoaders for any registered dataset.

    Returns (train_loader, test_loader, in_dim) exactly like models.get_loaders,
    so it is a drop-in replacement. Falls back to models.get_loaders' synthetic
    stand-in behaviour if the real data cannot be fetched, so the scripts still
    run in an offline sandbox.
    """
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}. "
                         f"Choose from: {', '.join(sorted(DATASETS))}")
    cls_name, split, n_default, label_offset = DATASETS[dataset]
    num_classes = n_default if num_classes is None else num_classes
    root = root if root is not None else DATA_ROOT

    # fashion keeps the original code path, so Week 5-7 numbers stay identical
    if dataset == "fashion":
        return _fashion_loaders(resolution=resolution, num_classes=num_classes,
                                batch_size=batch_size, root=root)

    try:
        from torchvision import datasets as tvd
        ctor = getattr(tvd, cls_name)
        kw = {"split": split} if split else {}
        tr_raw = ctor(root, train=True, download=True, **kw)
        te_raw = ctor(root, train=False, download=True, **kw)
        transpose = is_emnist(dataset)
        tr = _prep(tr_raw, resolution, num_classes, transpose, label_offset)
        te = _prep(te_raw, resolution, num_classes, transpose, label_offset)
        if verbose:
            print(f"[datasets_ext] {dataset}: {len(tr)} train / {len(te)} test, "
                  f"{num_classes} classes, {resolution}x{resolution}"
                  + ("  [EMNIST transpose applied]" if transpose else ""))
    except Exception as e:
        print(f"[datasets_ext] real {dataset} unavailable ({type(e).__name__}: {e}); "
              f"synthetic stand-in.")
        d = resolution * resolution

        def synth(n):
            y = torch.randint(0, num_classes, (n,))
            X = (torch.rand(n, d) * 0.1
                 + (y.float().unsqueeze(1) / num_classes)).clamp(0, 1)
            return TensorDataset(X, y)
        tr, te = synth(2000), synth(1000)

    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(te, batch_size=batch_size, shuffle=False),
            resolution * resolution)


# ==============================================================================
# Self-test: the prep logic, without needing the real downloads
# ==============================================================================
def _self_test():
    class Fake:
        def __init__(self, data, targets):
            self.data, self.targets = data, targets

    # a marker image that is obviously asymmetric, so a transpose is visible
    img = torch.zeros(3, 28, 28, dtype=torch.uint8)
    img[:, 0, :] = 255                                   # bright TOP row
    ds = Fake(img, torch.tensor([0, 1, 2]))

    plain = _prep(ds, 28, 10, transpose=False, label_offset=0)
    X = plain.tensors[0].reshape(-1, 28, 28)
    assert X[0, 0, :].mean() > 0.9 and X[0, :, 0].mean() < 0.1, "no-transpose wrong"

    flipped = _prep(ds, 28, 10, transpose=True, label_offset=0)
    Xf = flipped.tensors[0].reshape(-1, 28, 28)
    assert Xf[0, :, 0].mean() > 0.9 and Xf[0, 0, :].mean() < 0.1, "transpose wrong"
    print("  transpose: top row -> left column, as EMNIST needs")

    off = _prep(Fake(img, torch.tensor([1, 2, 3])), 28, 26,
                transpose=True, label_offset=1)
    assert off.tensors[1].tolist() == [0, 1, 2], "letters offset wrong"
    print("  letters label offset: 1,2,3 -> 0,1,2")

    sub = _prep(Fake(img, torch.tensor([0, 5, 9])), 28, 6,
                transpose=False, label_offset=0)
    assert sub.tensors[1].tolist() == [0, 5], "class subsetting wrong"
    print("  class subsetting: kept labels < 6")

    small = _prep(ds, 14, 10, transpose=False, label_offset=0)
    assert small.tensors[0].shape == (3, 196), small.tensors[0].shape
    print("  resize 28 -> 14 gives 196-dim inputs")

    for name in DATASETS:
        assert default_classes(name) == DATASETS[name][2]
    print(f"  registry: {len(DATASETS)} datasets, class counts resolve")
    print("  all checks passed")


if __name__ == "__main__":
    print("datasets_ext self-test")
    _self_test()
