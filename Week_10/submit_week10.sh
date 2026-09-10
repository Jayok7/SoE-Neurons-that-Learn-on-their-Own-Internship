#!/bin/bash
# =============================================================================
# submit_week10.sh  (Week 10 / node width and node depth)
#
# The supervisor proposed two independent ways to make a NAN node richer:
#
#   WIDTH  how many units a node emits.   784 -> k -> 784
#          Already implemented as group_ae. k = 1 reproduces NAN exactly,
#          k = H reproduces ANN exactly. PARAMETER-NEUTRAL at every k, so no
#          capacity confound.
#
#   DEPTH  how much computation happens inside a node before it emits its one
#          scalar.  784 -> m -> 1 -> m -> 784   (dendritic_ae).
#          NOT parameter-neutral -- see EXPT=matched below.
#
# EXPT selects which sweep to run and fills in the variant list for you:
#
#   nodes20  nan group10 group5 group4 group2 ann   at H = 20
#            The supervisor's framing exactly: 20x1, 10x2, 5x4, 4x5, 2x10, 1x20.
#            Total hidden units fixed at 20; only the partition changes.
#   width    nan group64 group32 group16 group8 group4 group2 ann   at H = 128
#            The same sweep at the width where Weeks 7-9 measured the strongest
#            collapse, so the two scales can be compared.
#   annshrink  ann alone at H = 2 ... 512
#            "Decrease the number of hidden nodes in the ANN and see where it
#            breaks." NOTE: already answered by the Week 7-9 sweeps -- only
#            re-run this if you want it in one place with the others.
#   penalty  nan and group5 at H = 20, swept over diversity-penalty weights
#   depth    nan dend2 dend4 dend8                                  at H = 128
#   matched  nan at H = 128, 214, 385, 727
#            The parameter-matched controls for depth. dend2 at H=128 has the
#            same hidden-parameter count as NAN at H=214; dend4 matches 385;
#            dend8 matches 727. Without this arm, a win from depth cannot be
#            separated from a win from simply having more parameters.
#   both     nodes20 then width
#   all      nodes20, width, depth, matched
#
# SUBMIT WITH
#     sbatch --export=ALL,DATASET=emnist-balanced,EXPT=width   submit_week10.sh
#     sbatch --export=ALL,DATASET=emnist-balanced,EXPT=depth   submit_week10.sh
#     sbatch --export=ALL,DATASET=emnist-balanced,EXPT=matched submit_week10.sh
#     sbatch --export=ALL,DATASET=mnist,EXPT=all               submit_week10.sh
#     sbatch --export=ALL,DATASET=emnist-balanced,EXPT=width,MODE=both submit_week10.sh
#
# WHY EMNIST FIRST
#   Weeks 8-9 measured the hidden layer as worth +4.1 points on Fashion-MNIST
#   but +15.4 on EMNIST-balanced. On Fashion there is barely four points of
#   headroom for an architectural change to show up in, so a null result there
#   would be uninformative. Run EMNIST first, confirm on MNIST and Fashion.
#
# OUTPUTS
#   $PROJECT_DIR/experiments_week10/<dataset>/<expt>/sweep_hidden_<dataset>/
#   $PROJECT_DIR/Week_10/logs/week10_<dataset>_<expt>_<jobid>.log
#
# PREREQUISITE
#   Week_10/ must contain: models.py  simple_core.py  datasets_ext.py
#   collapse_metrics.py  sweep_hidden.py  collapse_analysis.py
#   group_ae.py  dendritic_ae.py
# =============================================================================

#SBATCH --job-name=week10-arch
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship/Week_10/logs/week10_%j.log
#SBATCH --error=/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship/Week_10/logs/week10_%j.err

# NOT `set -u`: the anaconda module's activate.d scripts reference unbound
# variables (ADDR2LINE), so -u makes `conda activate` fail before Python runs.
set -eo pipefail

PROJECT_DIR="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship"
WORK_DIR="$PROJECT_DIR/Week_10"
cd "$WORK_DIR"
mkdir -p logs

module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2
eval "$(conda shell.bash hook)"
conda activate torch_gpu

# ---- Overridable parameters -------------------------------------------------
DATASET="${DATASET:-emnist-balanced}"
EXPT="${EXPT:-nodes20}"              # nodes20 | width | annshrink | penalty
                                     # | depth | matched | both | all
PENALTY="${PENALTY:-decoder}"        # decoder | activation
PENALTY_WEIGHTS="${PENALTY_WEIGHTS:-0 0.01 0.1 1 10}"
MODE="${MODE:-sweep}"                # sweep | collapse | both
HIDDEN="${HIDDEN:-128}"              # the fixed width for width/depth sweeps
SEEDS="${SEEDS:-3}"
EPOCHS="${EPOCHS:-15}"
BATCH="${BATCH:-128}"
LR="${LR:-0.001}"
RES="${RES:-28}"
CLASSES="${CLASSES:-}"               # empty = the dataset's own class count
ACT="${ACT:-sigmoid}"
AE_PROB="${AE_PROB:-0.5}"
SCHEDULE="${SCHEDULE:-interleaved}"
LAM="${LAM:-2.0}"
COLLAPSE_SAMPLES="${COLLAPSE_SAMPLES:-2000}"

JOBID="${SLURM_JOB_ID:-manual}"
LOG_DIR="$WORK_DIR/logs"

export NAN_DATA_DIR="$PROJECT_DIR/data"
mkdir -p "$NAN_DATA_DIR"

COMMON="--seeds $SEEDS --epochs $EPOCHS --batch_size $BATCH --lr $LR \
        --res $RES --dataset $DATASET --activation $ACT --ae_prob $AE_PROB \
        --schedule $SCHEDULE --lam $LAM \
        --collapse_samples $COLLAPSE_SAMPLES"
[ -n "$CLASSES" ] && COMMON="$COMMON --classes $CLASSES"

echo "=========================================="
echo "Job: ${SLURM_JOB_NAME:-week10} | ID: $JOBID"
echo "Node: $(hostname) | Started: $(date)"
echo "dataset=$DATASET   expt=$EXPT   mode=$MODE   H=$HIDDEN"
[ "$EXPT" = "penalty" ] && echo "penalty=$PENALTY  weights=[$PENALTY_WEIGHTS]"
echo "seeds=$SEEDS  epochs=$EPOCHS  batch=$BATCH  lr=$LR  schedule=$SCHEDULE"
echo "data in : $NAN_DATA_DIR"
echo "logs in : $LOG_DIR"
echo

# ---- Fail fast if a required file is missing --------------------------------
for f in models.py simple_core.py datasets_ext.py collapse_metrics.py \
         sweep_hidden.py collapse_analysis.py group_ae.py dendritic_ae.py; do
    if [ ! -e "$f" ]; then
        echo "ERROR: $f is missing from $WORK_DIR."
        echo "Copy models.py from Week_4, simple_core.py from Week_5, and the"
        echo "rest from Week_8-9."
        exit 1
    fi
done

# ---- Refuse to run on the synthetic stand-in --------------------------------
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
    print(f"ERROR: '{name}' fell back to the synthetic stand-in.")
    print("Run fetch_datasets.sh on the login node and resubmit.")
    sys.exit(1)
print(f"OK: {name} -> {n_tr} train / {n_te} test, in_dim={in_dim}")
PYCHECK
echo

# ---- One sweep ---------------------------------------------------------------
run_one () {
    local tag="$1" variants="$2" widths="$3" extra="${4:-}"
    export NAN_OUT_DIR="$PROJECT_DIR/experiments_week10/$DATASET/$tag"
    mkdir -p "$NAN_OUT_DIR"
    echo "-------------------------------------------------------------"
    echo "[$tag]  variants: $variants"
    echo "[$tag]  widths  : $widths"
    echo "[$tag]  output  : $NAN_OUT_DIR"
    echo "-------------------------------------------------------------"
    if [ "$MODE" = "sweep" ] || [ "$MODE" = "both" ]; then
        python sweep_hidden.py $COMMON --variants $variants \
               --hidden_sizes $widths $extra
    fi
    if [ "$MODE" = "collapse" ] || [ "$MODE" = "both" ]; then
        python collapse_analysis.py $COMMON --variants $variants --hidden "$HIDDEN"
    fi
}

# H = 20: 20x1, 10x2, 5x4, 4x5, 2x10, 1x20  (G x k, total fixed at 20)
NODES20_VARIANTS="nan group10 group5 group4 group2 ann"
# H = 128: k = 1, 2, 4, 8, 16, 32, 64, 128 units per node
WIDTH_VARIANTS="nan group64 group32 group16 group8 group4 group2 ann"
ALL_WIDTHS="2 4 8 16 32 64 128 256 512"
DEPTH_VARIANTS="nan dend2 dend4 dend8"
MATCHED_WIDTHS="128 214 385 727"     # parameter-matched to dend1/2/4/8

PEN_ARGS="--penalty $PENALTY --penalty_sweep $PENALTY_WEIGHTS"

case "$EXPT" in
    nodes20)   run_one nodes20   "$NODES20_VARIANTS" "20" ;;
    width)     run_one width     "$WIDTH_VARIANTS"   "$HIDDEN" ;;
    annshrink) run_one annshrink "ann"               "$ALL_WIDTHS" ;;
    penalty)   run_one penalty   "nan group5"        "20" "$PEN_ARGS" ;;
    depth)     run_one depth     "$DEPTH_VARIANTS"   "$HIDDEN" ;;
    matched)   run_one matched   "nan"               "$MATCHED_WIDTHS" ;;
    both)      run_one nodes20   "$NODES20_VARIANTS" "20"
               run_one width     "$WIDTH_VARIANTS"   "$HIDDEN" ;;
    all)       run_one nodes20   "$NODES20_VARIANTS" "20"
               run_one width     "$WIDTH_VARIANTS"   "$HIDDEN"
               run_one depth     "$DEPTH_VARIANTS"   "$HIDDEN"
               run_one matched   "nan"               "$MATCHED_WIDTHS" ;;
    *) echo "ERROR: EXPT must be nodes20 | width | annshrink | penalty | depth | matched | both | all"
       exit 1 ;;
esac

echo
echo "Finished: $(date)"
echo "Outputs under: $PROJECT_DIR/experiments_week10/$DATASET/"
echo "=========================================="

cp -f "$LOG_DIR/week10_${JOBID}.log" \
      "$LOG_DIR/week10_${DATASET}_${EXPT}_${JOBID}.log" 2>/dev/null || true
