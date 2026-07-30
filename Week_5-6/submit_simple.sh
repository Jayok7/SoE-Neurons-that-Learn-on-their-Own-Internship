#!/bin/bash
# =============================================================================
# submit_simple.sh  (Week 5 foundation / Fashion-MNIST)  -  submit with:
#     sbatch submit_simple.sh                                    # interleaved, default variants
#     sbatch --export=ALL,MODE=combined submit_simple.sh         # combined-loss script
#     sbatch --export=ALL,MODE=both submit_simple.sh             # run both
#     sbatch --export=ALL,VARIANTS="linear frozen mlp nan ann" submit_simple.sh
#     sbatch --export=ALL,ACT=relu submit_simple.sh              # ReLU MLP variants
#     sbatch --export=ALL,MODE=combined,LAM=5 submit_simple.sh
# -----------------------------------------------------------------------------
# Runs the simplified foundation scripts:
#   simple_interleaved.py : the hidden-layer test (linear / frozen-reservoir /
#                           mlp / nan / ann) with interleaved(AE_PROB) training
#                           for the AE variants and pure task training otherwise.
#   simple_combined.py    : identical foundation, combined loss
#                           CE + LAM * recon on every batch for AE variants.
#
# Metrics: accuracy, top-2 accuracy, CE loss, reconstruction MSE. Every run's
# full configuration (batch size, lr, optimizer, sizes, versions, ...) is
# saved inside the results JSON. Outputs:
#   experiments_fashion_h<HIDDEN>/simple_interleaved/
#   experiments_fashion_h<HIDDEN>/simple_combined/
#
# Prerequisite: setup_env_week4.sh run once on the login node. Place
# simple_core.py + both runners NEXT TO models.py.
# =============================================================================

#SBATCH --job-name=fashion-simple
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/simple_%j.log
#SBATCH --error=logs/simple_%j.err

set -e
PROJECT_DIR="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship"
cd "$PROJECT_DIR/Week_5"
mkdir -p logs

module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2
eval "$(conda shell.bash hook)"
conda activate torch_gpu

export NAN_DATA_DIR="$PROJECT_DIR/data"

# ---- Overridable parameters -------------------------------------------------
MODE="${MODE:-interleaved}"          # interleaved | combined | both
VARIANTS="${VARIANTS:-}"             # empty = each script's default set
SEEDS="${SEEDS:-5}"
EPOCHS="${EPOCHS:-15}"
BATCH="${BATCH:-128}"
LR="${LR:-0.001}"
HIDDEN="${HIDDEN:-128}"
RES="${RES:-28}"                     # native Fashion-MNIST resolution
CLASSES="${CLASSES:-10}"
ACT="${ACT:-sigmoid}"                # sigmoid | relu (mlp/frozen only)
AE_PROB="${AE_PROB:-0.5}"            # interleaved script
LAM="${LAM:-2}"                      # combined script

export NAN_OUT_DIR="$PROJECT_DIR/experiments_fashion_h${HIDDEN}"
mkdir -p "$NAN_OUT_DIR"

COMMON="--seeds $SEEDS --epochs $EPOCHS --batch_size $BATCH --lr $LR \
        --hidden $HIDDEN --res $RES --classes $CLASSES --activation $ACT"
[ -n "$VARIANTS" ] && COMMON="$COMMON --variants $VARIANTS"

echo "=========================================="
echo "Job: $SLURM_JOB_NAME | ID: $SLURM_JOB_ID"
echo "Node: $(hostname) | Started: $(date)"
echo "mode=$MODE variants=${VARIANTS:-default} hidden=$HIDDEN res=$RES act=$ACT"
echo "seeds=$SEEDS epochs=$EPOCHS batch=$BATCH lr=$LR ae_prob=$AE_PROB lam=$LAM"
echo

if [ "$MODE" = "interleaved" ] || [ "$MODE" = "both" ]; then
    python simple_interleaved.py $COMMON --ae_prob "$AE_PROB"
fi
if [ "$MODE" = "combined" ] || [ "$MODE" = "both" ]; then
    python simple_combined.py $COMMON --lam "$LAM"
fi

echo
echo "Finished: $(date)"
echo "=========================================="
