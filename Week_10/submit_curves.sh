#!/bin/bash
# =============================================================================
# submit_curves.sh  -- per-epoch training curves for every group layout.
#
#   sbatch --export=ALL,DATASET=emnist-balanced,HIDDEN=128,EPOCHS=40 submit_curves.sh
#   sbatch --export=ALL,DATASET=mnist,HIDDEN=20,EPOCHS=40            submit_curves.sh
#
# Interleaved only, matching Week 10 and Weeks 5-9. There is no SCHEDULE knob
# here on purpose: two schedules on one epoch axis would be two different
# training budgets drawn as if they were comparable.
#
# PREREQUISITE
#   Week_10/ next to this directory, supplying group_ae.py, models.py,
#   simple_core.py, datasets_ext.py and collapse_metrics.py.
# =============================================================================
#SBATCH --job-name=curves
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/curves_%j.log

# NOT `set -u`: the anaconda module's activate.d scripts reference unbound
# variables (ADDR2LINE), so -u makes `conda activate` fail before Python runs.
set -eo pipefail
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate torch_gpu

DATASET="${DATASET:-emnist-balanced}"
HIDDEN="${HIDDEN:-128}"
EPOCHS="${EPOCHS:-40}"
SEEDS="${SEEDS:-3}"
BATCH="${BATCH:-128}"
LR="${LR:-0.001}"
ACT="${ACT:-sigmoid}"
AE_PROB="${AE_PROB:-0.5}"
RANK_SAMPLES="${RANK_SAMPLES:-2000}"
RANK_EVERY="${RANK_EVERY:-1}"
LAYOUTS="${LAYOUTS:-}"                 # empty = every partition of HIDDEN

cd /mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship/Week_10

echo "============================================================"
echo "job      : ${SLURM_JOB_ID:-manual}   $(date)"
echo "dataset  : $DATASET"
echo "width    : $HIDDEN"
echo "epochs   : $EPOCHS   seeds: $SEEDS"
echo "layouts  : ${LAYOUTS:-all partitions of $HIDDEN}"
echo "schedule : interleaved (ae_prob=$AE_PROB)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

python train_curves.py \
    --dataset "$DATASET" \
    --hidden "$HIDDEN" \
    --epochs "$EPOCHS" \
    --seeds "$SEEDS" \
    --batch "$BATCH" \
    --lr "$LR" \
    --activation "$ACT" \
    --ae_prob "$AE_PROB" \
    --rank_samples "$RANK_SAMPLES" \
    --rank_every "$RANK_EVERY" \
    ${LAYOUTS:+--layouts $LAYOUTS}

echo
echo "finished $(date)"
