#!/bin/bash
# =============================================================================
# submit_week10.sh  (Week 10)
#
# TWO experiments, both at a fixed hidden width. Nothing else is run.
#
#   EXPT=groups   How a fixed budget of neurons is PARTITIONED into groups that
#                 share a reconstruction target. At H = 20 that is
#                     20x1   10x2   5x4   4x5   2x10   1x20     (groups x units)
#                 i.e. variants  nan group10 group5 group4 group2 ann.
#                 `nan` IS group20 and `ann` IS group1 exactly (group_ae's
#                 self-test proves both), so the endpoints are the same code
#                 used in Weeks 5-9. Parameter-neutral at every G: the decoder
#                 count is H x in_dim throughout and only the routing matrix
#                 (H x G, negligible) changes. So a difference across the sweep
#                 is a difference in the objective, not in capacity.
#
#   EXPT=penalty  The diversity penalty turned up from 0, reading off accuracy,
#                 effective rank and reconstruction MSE at each weight. Run on
#                 `nan` only: the penalty acts on per-neuron decoders, which is
#                 what NAN has. p=0 is the control; sweep_hidden.py adds it if
#                 it is missing. PENALTY=decoder acts on the per-neuron decoder
#                 vectors; PENALTY=activation decorrelates activations over the
#                 batch. Runs at $HIDDEN, the same width as the group sweep.
#
#   EXPT=both     groups, then penalty. THE RECOMMENDED WAY TO RUN WEEK 10:
#                 one job means both sweeps share the same width, schedule,
#                 epochs and seeds by construction, so the two are directly
#                 comparable and the accuracy levels should line up.
#
#   EXPT=depth    Depth INSIDE a node (supervisor's suggestion): each node is a
#                 two-layer stacked autoencoder 784 -> m -> 1 -> m -> 784 rather
#                 than a single linear step. The node still emits one scalar, so
#                 the classifier input is unchanged. NOT parameter-neutral --
#                 dend4 at H=128 costs 3.0x NAN -- so this arm automatically
#                 also runs NAN at the parameter-matched widths as its control.
#
# EVERY RUN IS INTERLEAVED
#   SCHEDULE defaults to interleaved and should stay there. Both sweeps, and
#   Weeks 5-9, then sit on one schedule, so any difference between them is the
#   thing being varied rather than the training budget. Overriding SCHEDULE
#   prints a warning; the results are still written, but they are not
#   comparable with anything else in the project.
#
# SUBMIT WITH
#     sbatch --export=ALL,DATASET=emnist-balanced,EXPT=both    submit_week10.sh
#     sbatch --export=ALL,DATASET=mnist,EXPT=both              submit_week10.sh
#     sbatch --export=ALL,DATASET=emnist-balanced,EXPT=groups  submit_week10.sh
#     sbatch --export=ALL,DATASET=emnist-balanced,EXPT=penalty,PENALTY=activation submit_week10.sh
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
#   collapse_metrics.py  sweep_hidden.py  collapse_analysis.py  group_ae.py
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
EXPT="${EXPT:-groups}"               # groups | penalty | both | depth
PENALTY="${PENALTY:-decoder}"        # decoder | activation
PENALTY_WEIGHTS="${PENALTY_WEIGHTS:-0 0.01 0.1 1 10}"
MODE="${MODE:-sweep}"                # sweep | collapse | both
HIDDEN="${HIDDEN:-128}"              # the fixed width. Drives BOTH sweeps, so
                                     # they stay comparable. Must be composite
                                     # for the group sweep to have interior
                                     # points. 128 is above EMNIST's break-even
                                     # width of 32; 20 is below it.
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
         sweep_hidden.py collapse_analysis.py group_ae.py; do
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

# ---- Fairness guard ----------------------------------------------------------
# Week 10's sweeps are only comparable to each other, and to Weeks 5-9, if the
# schedule, width, epochs and seeds are identical everywhere. The script drives
# both sweeps from the same variables, so the only way they can diverge is an
# explicit override on the sbatch line. Say so loudly rather than discovering it
# in the results three hours later.
if [ "$SCHEDULE" != "interleaved" ]; then
    echo "*****************************************************************"
    echo "WARNING: SCHEDULE=$SCHEDULE, not interleaved."
    echo "  Week 10 results are only comparable across sweeps, and against"
    echo "  Weeks 5-9, when every run is interleaved. A penalty weight does"
    echo "  not mean the same thing under combined training, because every"
    echo "  batch carries a reconstruction update instead of about half."
    echo "  Continuing, but do not compare these outputs to interleaved runs."
    echo "*****************************************************************"
    echo
fi

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

# Every partition of HIDDEN into equal groups, largest G (= NAN) first.
# Fixed routing requires G to divide H, so the list is the divisors of HIDDEN:
# at H = 20 this is 20x1, 10x2, 5x4, 4x5, 2x10, 1x20. G = H is emitted as `nan`
# and G = 1 as `ann` because those variants ARE those endpoints exactly, and
# reusing them keeps the sweep tied to the Week 5-9 baselines.
GROUP_VARIANTS=$(python - "$HIDDEN" <<'PYVAR'
import sys
H = int(sys.argv[1])
out = []
for g in sorted((g for g in range(1, H + 1) if H % g == 0), reverse=True):
    out.append("nan" if g == H else "ann" if g == 1 else f"group{g}")
print(" ".join(out))
PYVAR
)

# The penalty sweep runs on NAN only. The penalty acts on per-neuron decoders,
# which is exactly what NAN has; adding a grouped variant mixes the two axes in
# one job and doubles the cost to answer a question the group sweep already
# answers on its own.
PENALTY_VARIANTS="${PENALTY_VARIANTS:-nan}"
PEN_ARGS="--penalty $PENALTY --penalty_sweep $PENALTY_WEIGHTS"

# Depth inside the node (supervisor's suggestion). NOT parameter-neutral, so
# it is always run together with NAN at the widths that match its parameter
# count -- without that control a win from depth cannot be told apart from a
# win from simply having more parameters.
DEND_VARIANTS="${DEND_VARIANTS:-nan dend2 dend4}"
MATCHED_WIDTHS=$(python - "$HIDDEN" "$DEND_VARIANTS" <<'PYDEND'
import sys, re
from dendritic_ae import matched_width_for_depth
H = int(sys.argv[1])
ms = [int(m.group(1)) for m in
      (re.match(r"dend(\d+)", v) for v in sys.argv[2].split()) if m]
print(" ".join(str(w) for w in
                sorted({matched_width_for_depth(m, 784, H) for m in ms})))
PYDEND
)

case "$EXPT" in
    groups)  run_one groups  "$GROUP_VARIANTS"    "$HIDDEN" ;;
    penalty) run_one penalty "$PENALTY_VARIANTS"  "$HIDDEN" "$PEN_ARGS" ;;
    both)    run_one groups  "$GROUP_VARIANTS"    "$HIDDEN"
             run_one penalty "$PENALTY_VARIANTS"  "$HIDDEN" "$PEN_ARGS" ;;
    depth)   [ -f dendritic_ae.py ] || { echo "ERROR: EXPT=depth needs dendritic_ae.py"; exit 1; }
             run_one depth         "$DEND_VARIANTS" "$HIDDEN"
             run_one depth_matched "nan"            "$MATCHED_WIDTHS" ;;
    *) echo "ERROR: EXPT must be groups | penalty | both | depth"
       exit 1 ;;
esac

echo
echo "Finished: $(date)"
echo "Outputs under: $PROJECT_DIR/experiments_week10/$DATASET/"
echo "=========================================="

cp -f "$LOG_DIR/week10_${JOBID}.log" \
      "$LOG_DIR/week10_${DATASET}_${EXPT}_${JOBID}.log" 2>/dev/null || true