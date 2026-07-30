#!/bin/bash
# =============================================================================
# submit_fill_gap.sh — fill in missing small-lambda points for combined sweep
#
# Usage:
#     sbatch submit_fill_gap.sh                             # static, h=128
#     sbatch --export=ALL,ARCH=spiking submit_fill_gap.sh   # spiking variant
# =============================================================================

#SBATCH --job-name=fill-gap
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/fill_gap_%j.log
#SBATCH --error=logs/fill_gap_%j.err

set -e
PROJECT_DIR="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship"
cd "$PROJECT_DIR/Week_4"
mkdir -p logs

module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2
eval "$(conda shell.bash hook)"
conda activate torch_gpu

export NAN_DATA_DIR="$PROJECT_DIR/data"

ARCH="${ARCH:-static}"
HIDDEN="${HIDDEN:-128}"
SEEDS="${SEEDS:-5}"
EPOCHS="${EPOCHS:-9}"

export NAN_OUT_DIR="$PROJECT_DIR/experiments_fashion_h${HIDDEN}/extra_sweeps"
mkdir -p "$NAN_OUT_DIR"

echo "=========================================="
echo "Job: $SLURM_JOB_NAME | ID: $SLURM_JOB_ID"
echo "Node: $(hostname) | Started: $(date)"
echo "arch=$ARCH hidden=$HIDDEN seeds=$SEEDS epochs=$EPOCHS"
echo "output -> $NAN_OUT_DIR"
echo

python fill_combined_gap.py \
    --arch "$ARCH" \
    --seeds "$SEEDS" \
    --epochs "$EPOCHS" \
    --hidden "$HIDDEN"

echo
echo "Finished: $(date)"
echo "=========================================="
