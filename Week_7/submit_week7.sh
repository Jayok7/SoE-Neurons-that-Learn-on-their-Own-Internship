#!/bin/bash
# =============================================================================
# submit_week7.sh  (Week 7 / Fashion-MNIST)  -  submit with:
#     sbatch submit_week7.sh                                  # sweep only
#     sbatch --export=ALL,MODE=collapse submit_week7.sh        # collapse only
#     sbatch --export=ALL,MODE=both submit_week7.sh            # both
#     sbatch --export=ALL,HIDDEN_SIZES="2 4 8 16 32 64 128 256 512 1024" submit_week7.sh
#     sbatch --export=ALL,MODE=collapse,HIDDEN=512 submit_week7.sh
#     sbatch --export=ALL,VARIANTS="mlp nan ann",SEEDS=5 submit_week7.sh
# -----------------------------------------------------------------------------
#   sweep_hidden.py      : sweeps hidden width, recording accuracy/top-2/CE/
#                          recon AND the full collapse report at each width.
#   collapse_analysis.py : trains at one width and produces the numeric report
#                          plus the diagnostic images (per-neuron
#                          reconstructions, decoder templates, correlation
#                          heatmap, spectrum, activation statistics).
#
# Prerequisite: conda env torch_gpu (setup_env_week4.sh). Place
# collapse_metrics.py, sweep_hidden.py, collapse_analysis.py and the Week 5
# simple_core.py NEXT TO models.py.
# =============================================================================

#SBATCH --job-name=fashion-week7
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/week7_%j.log
#SBATCH --error=logs/week7_%j.err

set -e
PROJECT_DIR="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship"
cd "$PROJECT_DIR/Week_7"
mkdir -p logs

module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2
eval "$(conda shell.bash hook)"
conda activate torch_gpu

export NAN_DATA_DIR="$PROJECT_DIR/data"

# ---- Overridable parameters -------------------------------------------------
MODE="${MODE:-sweep}"                # sweep | collapse | both
HIDDEN_SIZES="${HIDDEN_SIZES:-2 4 8 16 32 64 128 256 512}"   # sweep script
HIDDEN="${HIDDEN:-128}"              # collapse script (single width)
VARIANTS="${VARIANTS:-}"             # empty = each script's default set
SEEDS="${SEEDS:-3}"
EPOCHS="${EPOCHS:-15}"
BATCH="${BATCH:-128}"
LR="${LR:-0.001}"
RES="${RES:-28}"                     # native Fashion-MNIST resolution
CLASSES="${CLASSES:-10}"
ACT="${ACT:-sigmoid}"                # sigmoid | relu (mlp/frozen only)
AE_PROB="${AE_PROB:-0.5}"
COLLAPSE_SAMPLES="${COLLAPSE_SAMPLES:-2000}"

export NAN_OUT_DIR="$PROJECT_DIR/experiments_week7"
mkdir -p "$NAN_OUT_DIR"

COMMON="--seeds $SEEDS --epochs $EPOCHS --batch_size $BATCH --lr $LR \
        --res $RES --classes $CLASSES --activation $ACT --ae_prob $AE_PROB \
        --collapse_samples $COLLAPSE_SAMPLES"
[ -n "$VARIANTS" ] && COMMON="$COMMON --variants $VARIANTS"

echo "=========================================="
echo "Job: $SLURM_JOB_NAME | ID: $SLURM_JOB_ID"
echo "Node: $(hostname) | Started: $(date)"
echo "mode=$MODE variants=${VARIANTS:-default} res=$RES act=$ACT"
echo "seeds=$SEEDS epochs=$EPOCHS batch=$BATCH lr=$LR ae_prob=$AE_PROB"
echo "sweep widths=[$HIDDEN_SIZES] | collapse hidden=$HIDDEN"
echo

if [ "$MODE" = "sweep" ] || [ "$MODE" = "both" ]; then
    python sweep_hidden.py $COMMON --hidden_sizes $HIDDEN_SIZES
fi
if [ "$MODE" = "collapse" ] || [ "$MODE" = "both" ]; then
    python collapse_analysis.py $COMMON --hidden "$HIDDEN"
fi

echo
echo "Finished: $(date)"
echo "=========================================="
