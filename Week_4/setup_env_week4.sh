#!/bin/bash
# =============================================================================
# setup_env_fashion.sh  —  RUN ONCE on the CSF3 LOGIN NODE (not via sbatch).
# -----------------------------------------------------------------------------
# Fashion-MNIST variant of the autoencoder-internship setup script.
#
# Purpose:
#   1. Create (or reuse) the PyTorch conda environment for the project.
#   2. Pre-download FASHION-MNIST while we still have internet — COMPUTE NODES
#      HAVE NO NETWORK ACCESS, so the data must be fetched here, on the login
#      node.
#
# Usage:
#   cd ~/autoencoder-internship
#   bash setup_env_fashion.sh
#
# You only need to run this once per dataset; the env and the downloaded data
# persist between jobs. If you already ran the MNIST setup_env.sh, the conda
# env (torch_gpu) is reused and only the Fashion-MNIST download step is new.
# =============================================================================
set -e   # stop on first error

PROJECT_DIR="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship"
ENV_NAME="torch_gpu"
cd "$PROJECT_DIR"

echo "=== 1/5  Loading modules ==="
module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2

eval "$(conda shell.bash hook)"

echo "=== 2/5  Creating / reusing conda env '$ENV_NAME' (python 3.10) ==="
conda create -y -n "$ENV_NAME" python=3.10 || echo "env may already exist, continuing"
conda activate "$ENV_NAME"
echo "  Python binary: $(which python)"
echo "  Python version: $(python --version 2>&1)"

echo "=== 3/5  Installing PyTorch + deps ==="
pip install "numpy<2.0"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib scipy

echo "=== 4/5  Pre-downloading FASHION-MNIST to ./data (needs login-node internet) ==="
# Fashion-MNIST uses the same four gzipped IDX files as MNIST, but a different
# mirror. torchvision expects them under FashionMNIST/raw/ (NOT MNIST/raw/).
FASHION_DIR="./data/FashionMNIST/raw"
mkdir -p "$FASHION_DIR"

MIRROR="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
for FILE in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
            t10k-images-idx3-ubyte.gz  t10k-labels-idx1-ubyte.gz; do
    if [ ! -f "$FASHION_DIR/$FILE" ]; then
        echo "  downloading $FILE ..."
        wget -q -O "$FASHION_DIR/$FILE" "$MIRROR/$FILE"
    else
        echo "  $FILE already cached, skipping"
    fi
done

echo "=== 5/5  Verifying Fashion-MNIST loads ==="
python - <<'PY'
import sys
print(f"  Using Python: {sys.executable}")
import torch, torchvision
print(f"  PyTorch: {torch.__version__} | torchvision: {torchvision.__version__}")
from torchvision import datasets, transforms
tf = transforms.ToTensor()
train = datasets.FashionMNIST("./data", train=True,  download=False, transform=tf)
test  = datasets.FashionMNIST("./data", train=False, download=False, transform=tf)
print(f"  Fashion-MNIST loaded OK — {len(train)} train, {len(test)} test samples")
print(f"  classes: {train.classes}")
PY

echo ""
echo "=========================================================="
echo "Fashion-MNIST setup complete."
echo "  Env:   $ENV_NAME"
echo "  Data:  $PROJECT_DIR/data/FashionMNIST"
echo ""
echo "To run the experiments:"
echo "  cd $PROJECT_DIR/Week_4"
echo "  sbatch submit_penalised_compare.sh"
echo "=========================================================="
