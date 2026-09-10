#!/bin/bash
# =============================================================================
# fetch_datasets.sh — download MNIST / EMNIST once, on the login node.
#
# Run this BEFORE submitting any Week 8 job:
#     cd $PROJECT_DIR/Week_8-9 && bash fetch_datasets.sh
#
# Why a separate script rather than letting the job download:
#   * Compute nodes often have no outbound internet, so a download inside the
#     job fails and datasets_ext silently substitutes synthetic data. The job
#     then "succeeds" and produces a full set of meaningless numbers.
#   * Downloading once on the login node means every later job starts instantly
#     and three parallel jobs cannot race to write the same files.
#
# torchvision's EMNIST mirror frequently returns HTTP 403. If that happens this
# script falls back to fetching the NIST zip directly with curl.
# =============================================================================

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship}"
DATA_DIR="${NAN_DATA_DIR:-$PROJECT_DIR/data}"
mkdir -p "$DATA_DIR"

module load apps/binapps/anaconda3/2021.11 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate torch_gpu 2>/dev/null || true

echo "Data directory: $DATA_DIR"
echo

# ---- MNIST and Fashion-MNIST: torchvision handles these reliably ------------
python - "$DATA_DIR" <<'PY'
import sys
from torchvision import datasets
root = sys.argv[1]
for name, ctor in [("MNIST", datasets.MNIST),
                   ("FashionMNIST", datasets.FashionMNIST)]:
    try:
        ctor(root, train=True,  download=True)
        ctor(root, train=False, download=True)
        print(f"  OK   {name}")
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
PY

# ---- EMNIST: try torchvision first ------------------------------------------
echo
echo "EMNIST via torchvision..."
if python - "$DATA_DIR" <<'PY'
import sys
from torchvision import datasets
root = sys.argv[1]
try:
    datasets.EMNIST(root, split="balanced", train=True, download=True)
    print("  OK   EMNIST (torchvision)")
except Exception as e:
    print(f"  FAIL EMNIST via torchvision: {type(e).__name__}: {e}")
    sys.exit(1)
PY
then
    echo "EMNIST ready."
else
    # ---- Fallback: fetch the NIST zip directly ------------------------------
    echo
    echo "Falling back to a direct download of the EMNIST archive."
    RAW="$DATA_DIR/EMNIST/raw"
    mkdir -p "$RAW"
    cd "$RAW"
    URLS=(
      "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
      "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
    )
    OK=0
    for u in "${URLS[@]}"; do
        echo "  trying $u"
        if curl -fL --retry 3 --connect-timeout 30 -o gzip.zip "$u"; then OK=1; break; fi
    done
    if [ "$OK" -ne 1 ]; then
        echo
        echo "Automatic download failed. Fetch gzip.zip manually on your laptop from"
        echo "  https://www.nist.gov/itl/products-and-services/emnist-dataset"
        echo "then copy it up with:"
        echo "  scp gzip.zip m84149ji@csf3.itservices.manchester.ac.uk:$RAW/"
        echo "and re-run this script."
        exit 1
    fi
    unzip -o -q gzip.zip && rm -f gzip.zip
    # the zip nests everything under gzip/ — flatten it and decompress
    if [ -d gzip ]; then mv -f gzip/* . && rmdir gzip; fi
    gunzip -f ./*.gz 2>/dev/null || true
    echo "  extracted into $RAW"
fi

# ---- Verify every dataset actually loads ------------------------------------
echo
echo "Verifying..."
cd "$(dirname "$0")"
python - "$DATA_DIR" <<'PY'
import os, sys
os.environ["NAN_DATA_DIR"] = sys.argv[1]
from datasets_ext import get_loaders, default_classes, DATASETS
bad = []
for name in ["fashion", "mnist", "emnist-digits", "emnist-letters", "emnist-balanced"]:
    try:
        tr, te, d = get_loaders(dataset=name, resolution=28,
                                num_classes=default_classes(name),
                                batch_size=256, verbose=False)
        n = len(tr.dataset)
        synthetic = n <= 2000
        print(f"  {'SYNTH' if synthetic else 'OK   '} {name:18} "
              f"{n:>7} train / {len(te.dataset):>6} test  "
              f"{default_classes(name):>2} classes")
        if synthetic:
            bad.append(name)
    except Exception as e:
        print(f"  FAIL  {name:18} {type(e).__name__}: {e}")
        bad.append(name)
if bad:
    print(f"\nNOT usable yet: {', '.join(bad)}")
    sys.exit(1)
print("\nAll datasets ready.")
PY
