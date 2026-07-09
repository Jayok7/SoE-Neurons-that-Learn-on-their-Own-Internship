#!/bin/bash
# =============================================================================
# submit_penalised_compare.sh  (Fashion-MNIST variant)  —  submit with:
#     sbatch submit_penalised_compare.sh                        # dw=2, h=128
#     sbatch --export=ALL,DIV_WEIGHT=5 submit_penalised_compare.sh
#     sbatch --export=ALL,DIV_WEIGHT=0,HIDDEN=20 submit_penalised_compare.sh
# -----------------------------------------------------------------------------
# Runs the penalised schedule comparison (interleaved vs sequential vs hybrid)
# for both NAN architectures, on FASHION-MNIST. Produces per-epoch accuracy /
# rank curves and a 3x2 grid figure for each NAN architecture.
#
# Overridable variables (via --export):
#     DIV_WEIGHT  — decoder penalty strength (default: 2)
#     HIDDEN      — hidden-layer width        (default: 128)
#
# Prerequisite: run `bash setup_env_fashion.sh` ONCE first
# (creates the env, caches Fashion-MNIST).
# =============================================================================

#SBATCH --job-name=nan-fashion-pen-cmp
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/fashion_pen_cmp_%j.log
#SBATCH --error=logs/fashion_pen_cmp_%j.err

set -e
PROJECT_DIR="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship"
cd "$PROJECT_DIR/Week_4"
mkdir -p logs

module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2
eval "$(conda shell.bash hook)"
conda activate torch_gpu

# DATA stays at the project root; the loader reads FashionMNIST/raw from here.
export NAN_DATA_DIR="$PROJECT_DIR/data"

# Overridable parameters
HIDDEN="${HIDDEN:-128}"
DIV_WEIGHT="${DIV_WEIGHT:-2}"

# Fashion-specific output folder so it never clashes with the MNIST results.
export NAN_OUT_DIR="$PROJECT_DIR/experiments_fashion_h${HIDDEN}"
mkdir -p "$NAN_OUT_DIR"

SCRIPT=run_experiments.py
SEEDS=10
EPOCHS=15

echo "=========================================="
echo "Job: $SLURM_JOB_NAME | ID: $SLURM_JOB_ID"
echo "Node: $(hostname) | Started: $(date)"
echo "Dataset=Fashion-MNIST  Hidden=$HIDDEN  DIV_WEIGHT=$DIV_WEIGHT"
echo "=========================================="
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

for ARCH in static spiking; do
    echo ">>> penalised_compare arch=$ARCH dw=$DIV_WEIGHT hidden=$HIDDEN (Fashion-MNIST)"
    python "$SCRIPT" --mode penalised_compare --arch "$ARCH" \
        --div_weight "$DIV_WEIGHT" \
        --seeds "$SEEDS" --epochs "$EPOCHS" --res 28 --classes 10 --hidden "$HIDDEN"
done

echo "=========================================="
echo "Fashion-MNIST penalised schedule comparison done — $(date)"
echo "Results in: $NAN_OUT_DIR/penalised_compare/"
echo "=========================================="
