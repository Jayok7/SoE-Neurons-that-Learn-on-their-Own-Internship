#!/bin/bash
# =============================================================================
# submit_week89.sh  (Weeks 8-9 / MNIST + EMNIST)
#
# Same experiments as Week 7, now across datasets. The only structural change
# is that outputs and logs are separated PER DATASET, so an EMNIST run can
# never overwrite an MNIST one and the Week 7 Fashion-MNIST results stay
# untouched in experiments_week7/.
#
#   $PROJECT_DIR/
#     experiments_week7/                 <- Week 7, left alone
#     experiments_week8-9/
#       mnist/           sweep_hidden_mnist/  collapse_analysis_mnist_h128/
#       emnist-balanced/ sweep_hidden_emnist-balanced/  ...
#       fashion/         (only if you re-run fashion under a new schedule)
#     Week_8-9/logs/     week89_<dataset>_<jobid>.log
#
# SUBMIT WITH
#     sbatch --export=ALL,DATASET=mnist submit_week89.sh
#     sbatch --export=ALL,DATASET=emnist-balanced submit_week89.sh
#     sbatch --export=ALL,DATASET=emnist-letters,MODE=both submit_week89.sh
#     sbatch --export=ALL,DATASET=mnist,MODE=collapse,HIDDEN=128 submit_week89.sh
#     sbatch --export="ALL,DATASET=mnist,SCHEDULE=interleaved combined" submit_week89.sh
#     sbatch --export=ALL,DATASET=emnist-balanced,SEEDS=5,TIME=16:00:00 submit_week89.sh
#
# DATASETS
#     fashion           10 cls,  60k train   (the Week 7 baseline)
#     mnist             10 cls,  60k train
#     emnist-digits     10 cls, 240k train
#     emnist-letters    26 cls, 125k train
#     emnist-balanced   47 cls, 113k train   <- the interesting one
#     emnist-bymerge    47 cls, 698k train
#     emnist-byclass    62 cls, 698k train
#
#   --classes is left unset by default, so each dataset uses its own class
#   count. The scripts read it from datasets_ext.DATASETS.
#
# PREREQUISITE
#   conda env torch_gpu. Week_8-9/ must contain:
#     models.py  simple_core.py  (copied or symlinked from Week_4 / Week_5)
#     datasets_ext.py  collapse_metrics.py  sweep_hidden.py
#     collapse_analysis.py  group_ae.py
# =============================================================================

#SBATCH --job-name=week89-data
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/week89_%j.log
#SBATCH --error=logs/week89_%j.err

set -euo pipefail

PROJECT_DIR="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship"
WORK_DIR="$PROJECT_DIR/Week_8-9"
cd "$WORK_DIR"
mkdir -p logs

module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2
eval "$(conda shell.bash hook)"
conda activate torch_gpu

# ---- Overridable parameters -------------------------------------------------
DATASET="${DATASET:-mnist}"          # see the list above
MODE="${MODE:-sweep}"                # sweep | collapse | both
HIDDEN_SIZES="${HIDDEN_SIZES:-2 4 8 16 32 64 128 256 512}"
HIDDEN="${HIDDEN:-128}"              # collapse script (single width)
VARIANTS="${VARIANTS:-}"             # empty = each script's default set
SEEDS="${SEEDS:-3}"
EPOCHS="${EPOCHS:-15}"
BATCH="${BATCH:-128}"
LR="${LR:-0.001}"
RES="${RES:-28}"
CLASSES="${CLASSES:-}"               # empty = the dataset's own class count
ACT="${ACT:-sigmoid}"                # sigmoid | relu (mlp/frozen only)
AE_PROB="${AE_PROB:-0.5}"            # interleaved only
SCHEDULE="${SCHEDULE:-interleaved}"  # "interleaved", "combined", or both
LAM="${LAM:-2.0}"                    # combined only: CE + LAM*recon
COLLAPSE_SAMPLES="${COLLAPSE_SAMPLES:-2000}"

# ---- Per-dataset input / output separation ----------------------------------
# Datasets are shared across weeks, so they stay in one place and are only
# downloaded once. Outputs are namespaced by dataset so parallel jobs on
# different datasets cannot collide.
export NAN_DATA_DIR="$PROJECT_DIR/data"
export NAN_OUT_DIR="$PROJECT_DIR/experiments_week8-9/$DATASET"
mkdir -p "$NAN_DATA_DIR" "$NAN_OUT_DIR"

# Rename the SLURM log to include the dataset, so a directory listing is
# readable at a glance rather than a wall of job IDs. Best-effort only: if the
# rename fails the job carries on writing to the original file.
JOBID="${SLURM_JOB_ID:-manual}"
TAGGED_LOG="logs/week89_${DATASET}_${JOBID}.log"

COMMON="--seeds $SEEDS --epochs $EPOCHS --batch_size $BATCH --lr $LR \
        --res $RES --dataset $DATASET --activation $ACT --ae_prob $AE_PROB \
        --schedule $SCHEDULE --lam $LAM \
        --collapse_samples $COLLAPSE_SAMPLES"
[ -n "$VARIANTS" ] && COMMON="$COMMON --variants $VARIANTS"
[ -n "$CLASSES" ]  && COMMON="$COMMON --classes $CLASSES"

echo "=========================================="
echo "Job: ${SLURM_JOB_NAME:-week89} | ID: $JOBID"
echo "Node: $(hostname) | Started: $(date)"
echo "dataset=$DATASET   mode=$MODE   res=$RES   act=$ACT"
echo "variants=${VARIANTS:-default}   classes=${CLASSES:-auto}"
echo "schedule=$SCHEDULE   ae_prob=$AE_PROB   lam=$LAM"
echo "seeds=$SEEDS   epochs=$EPOCHS   batch=$BATCH   lr=$LR"
echo "sweep widths=[$HIDDEN_SIZES]   collapse hidden=$HIDDEN"
echo "data in  : $NAN_DATA_DIR"
echo "output to: $NAN_OUT_DIR"
echo

# ---- Fail fast if a required file is missing --------------------------------
for f in models.py simple_core.py datasets_ext.py collapse_metrics.py \
         sweep_hidden.py collapse_analysis.py; do
    if [ ! -e "$f" ]; then
        echo "ERROR: $f is missing from $WORK_DIR."
        echo "Copy models.py from Week_4 and simple_core.py from Week_5."
        exit 1
    fi
done

# ---- Confirm the dataset is actually present before burning the allocation --
# datasets_ext silently falls back to a synthetic stand-in when a download
# fails, which would otherwise produce a full run of meaningless numbers.
# EMNIST in particular often 403s from torchvision's mirror.
echo "Checking dataset availability..."
python - "$DATASET" <<'PYCHECK'
import sys
from datasets_ext import get_loaders, default_classes
name = sys.argv[1]
tr, te, in_dim = get_loaders(dataset=name, resolution=28,
                             num_classes=default_classes(name),
                             batch_size=256, verbose=True)
n_tr, n_te = len(tr.dataset), len(te.dataset)
if n_tr <= 2000 and n_te <= 1000:
    print(f"ERROR: '{name}' fell back to the synthetic stand-in "
          f"({n_tr} train / {n_te} test).")
    print("The real data was not downloadable. Fetch it manually into "
          "$NAN_DATA_DIR and resubmit.")
    sys.exit(1)
print(f"OK: {name} -> {n_tr} train / {n_te} test, in_dim={in_dim}")
PYCHECK
echo

if [ "$MODE" = "sweep" ] || [ "$MODE" = "both" ]; then
    echo "--- sweep_hidden.py ---"
    python sweep_hidden.py $COMMON --hidden_sizes $HIDDEN_SIZES
fi
if [ "$MODE" = "collapse" ] || [ "$MODE" = "both" ]; then
    echo "--- collapse_analysis.py ---"
    python collapse_analysis.py $COMMON --hidden "$HIDDEN"
fi

echo
echo "Finished: $(date)"
echo "Outputs in: $NAN_OUT_DIR"
echo "=========================================="

# Tag the log with the dataset name for easier browsing.
cp -f "logs/week89_${JOBID}.log" "$TAGGED_LOG" 2>/dev/null || true
