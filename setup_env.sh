#!/bin/bash
# =============================================================================
# setup_env.sh  —  RUN ONCE on the CSF3 LOGIN NODE (not via sbatch).
# -----------------------------------------------------------------------------
# Purpose:
#   1. Create a dedicated PyTorch conda environment for the autoencoder project
#      (separate from the skin-cancer `tf_gpu` env, which is TensorFlow).
#   2. Pre-download MNIST while we still have internet — COMPUTE NODES HAVE NO
#      NETWORK ACCESS, so the data must be fetched here, on the login node.
#
# Usage:
#   cd ~/autoencoder-intership
#   bash setup_env.sh
#
# This takes a few minutes. You only ever need to run it once; the env and the
# downloaded MNIST persist between jobs.
# =============================================================================
set -e   # stop on first error

PROJECT_DIR="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-intership"
ENV_NAME="torch_gpu"
cd "$PROJECT_DIR"

echo "=== 1/5  Loading modules ==="
module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2

# --------------------------------------------------------------------------
# Use conda's shell hook for reliable activation (source activate is fragile
# on CSF and often fails to switch the Python binary).
# --------------------------------------------------------------------------
eval "$(conda shell.bash hook)"

echo "=== 2/5  Creating conda env '$ENV_NAME' (python 3.10) ==="
# If it already exists this is a no-op-ish; remove first if you want a clean slate:
#   conda env remove -n torch_gpu
conda create -y -n "$ENV_NAME" python=3.10 || echo "env may already exist, continuing"

# activate it (using conda activate, NOT source activate)
conda activate "$ENV_NAME"

# Verify we are in the right env
echo "  Python binary: $(which python)"
echo "  Python version: $(python --version 2>&1)"

echo "=== 3/5  Installing PyTorch + deps ==="
# NumPy pinned <2.0 to match the system's compiled packages (matplotlib etc.),
# exactly as the skin-cancer guide warns.
pip install "numpy<2.0"
# CUDA 12.1 wheels work fine against the cuda/12.6.2 module on CSF.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib scipy

echo "=== 4/5  Pre-downloading MNIST to ./data (needs login-node internet) ==="
# The original host (yann.lecun.com) is dead (404). Download the four gzipped
# files from the AWS S3 mirror that modern torchvision uses internally.
MNIST_DIR="./data/MNIST/raw"
mkdir -p "$MNIST_DIR"

MIRROR="https://ossci-datasets.s3.amazonaws.com/mnist"
for FILE in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
            t10k-images-idx3-ubyte.gz  t10k-labels-idx1-ubyte.gz; do
    if [ ! -f "$MNIST_DIR/$FILE" ]; then
        echo "  downloading $FILE ..."
        wget -q -O "$MNIST_DIR/$FILE" "$MIRROR/$FILE"
    else
        echo "  $FILE already cached, skipping"
    fi
done

echo "=== 5/5  Verifying MNIST loads ==="
python - <<'PY'
import sys, os
print(f"  Using Python: {sys.executable}")
print(f"  Python version: {sys.version}")

import torch
print(f"  PyTorch: {torch.__version__}")
import torchvision
print(f"  torchvision: {torchvision.__version__}")

from torchvision import datasets, transforms
tf = transforms.ToTensor()
train = datasets.MNIST("./data", train=True,  download=False, transform=tf)
test  = datasets.MNIST("./data", train=False, download=False, transform=tf)
print(f"  MNIST loaded OK — {len(train)} train, {len(test)} test samples")
PY

echo ""
echo "=========================================================="
echo "Setup complete."
echo "  Env:   $ENV_NAME"
echo "  Data:  $PROJECT_DIR/data/MNIST"
echo ""
echo "To activate this env in your shell:"
echo "  module load apps/binapps/anaconda3/2021.11"
echo "  module load cuda/12.6.2"
echo '  eval "\$(conda shell.bash hook)"'
echo "  conda activate $ENV_NAME"
echo ""
echo "To run the experiments:"
echo "  sbatch submit_experiments.sh"
echo "=========================================================="
