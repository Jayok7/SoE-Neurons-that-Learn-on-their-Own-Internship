#!/bin/bash
# =============================================================================
# submit_ttfs_detach.sh -- time-to-first-spike NAN with coincidence grouping,
#                          under the detached (Bull-faithful) learning rule.
#
# WHY DETACHED
#   The rate-based re-run measured the group dial on both sides of the task
#   gradient. Attached, going from one shared decoder to one per neuron -- a
#   128-fold change in layout -- moved EMNIST accuracy by 0.22 points. Detached,
#   the same dial spans 57 points and the collapse index rises monotonically
#   along it.
#
#   Choosing the partition differently at FIXED G is a smaller intervention
#   than changing G. An attached routing comparison therefore sits inside a
#   region already measured to be flat, and a null there would say nothing
#   about the coincidence rule. Every experiment below runs detached by
#   default. DETACH=0 is available as a deliberate control, not as the default.
#
# A GRADIENT PROBE RUNS FIRST
#   One batch, cross-entropy only, assert the hidden gradient is exactly zero
#   when detached and non-zero when not. This replaces the old equivalence
#   check, which compared two training runs and flagged a mismatch that was an
#   artefact of the two schedules delivering different numbers of
#   reconstruction updates. train_ttfs.py checks the real model before each run.
#
# EVERYTHING LANDS UNDER
#   $ROOT/With_detach/experiments/ttfs/<dataset>/<tag>/
#   $ROOT/With_detach/logs/
#   so it tars up with the rest:
#     tar czf with_detach.tar.gz -C $ROOT With_detach
#
# EXPERIMENTS
#   EXPT=baseline   Does a TTFS layer train at all detached, and does the fixed
#                   group dial behave like the rate-based one? ttfs_ann,
#                   ttfs_group<G>, ttfs_nan. RUN THIS FIRST.
#   EXPT=routing    THE EXPERIMENT. Same G, three ways of choosing the
#                   partition: fixed (control), coincidence, learned.
#   EXPT=tau        Routing sharpness. The parameter most worth sweeping --
#                   see the note below.
#   EXPT=sigma      Coincidence window width, in timesteps.
#   EXPT=steps      Time resolution T.
#   EXPT=fire       Calibration target: how sparse the spike code is.
#
# PARAMETERS CHANGED FROM THE ORIGINAL DEFAULTS
#   route_norm=zscore   was: centre only. Measured, the centred coincidence
#                       score gave a logit spread of about 0.5, so
#                       softmax(logits/0.5) came out at mean top-1 probability
#                       0.082 against a uniform 0.0625 -- the soft routing
#                       carried almost no information and the hard argmax was
#                       decided by noise. Worse, the diagnostics read healthy
#                       throughout (used_groups 15.97/16, entropy 0.997),
#                       because neither can tell a balanced partition from an
#                       uninformative one. Standardising per neuron raises
#                       top-1 to 0.36 and drops entropy to 0.72.
#   tau_route=0.25      was 0.5. With zscore the logits sit near unit scale, so
#                       this is now a real sharpness knob. Sweep it: EXPT=tau.
#   route_top1          new diagnostic. Watch it, not used_groups. If it sits
#                       near 1/G the routing has learned nothing whatever the
#                       occupancy says.
#
# COST AND EPOCHS
#   A TTFS pass unrolls the layer T times, so T=25 is roughly 25 forward passes
#   per batch. On EMNIST-balanced at H=128 that lands near 20-30s an epoch, so
#   25 epochs x 5 seeds x 3 variants is about 3 hours. 25 epochs is a good
#   choice here and not wasteful -- the rate-based runs were still climbing
#   0.15-0.53 points per epoch at 15, and a spiking layer with a surrogate
#   gradient converges slower, not faster. SCHEDULE=combined roughly doubles
#   the cost, because the spike train is unrolled once per objective.
#
# SUBMIT
#   sbatch --export=ALL,EXPT=baseline submit_ttfs_detach.sh
#   sbatch --export=ALL,EXPT=routing,SEEDS=5,EPOCHS=25 submit_ttfs_detach.sh
#   sbatch --export=ALL,EXPT=tau submit_ttfs_detach.sh
#
# LAYOUT
#   Everything now lives under With_detach/, which is BOTH the code root and
#   the output root. The TTFS scripts sit in a subdirectory of it:
#
#     With_detach/
#         simple_core.py  models.py  datasets_ext.py  collapse_metrics.py
#         group_ae.py  dendritic_ae.py  inhibition.py
#         sweep_hidden.py  collapse_analysis.py  submit_with_detach.sh
#         ttfs/
#             ttfs_nan.py  train_ttfs.py  submit_ttfs_detach.sh
#         experiments/    logs/    MANIFEST.tsv
#
#   Nothing is read from Week_10/ any more. train_ttfs.py finds the shared
#   modules by looking in its parent; NAN_CODE_ROOT overrides that if the
#   scripts ever move.
#
#   snntorch must be in the environment.
# =============================================================================
#SBATCH --job-name=ttfs-det
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship/With_detach/logs/ttfs_%j.log
#SBATCH --error=/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship/With_detach/logs/ttfs_%j.err

# NOT `set -u`: the anaconda module's activate.d scripts reference unbound
# variables (ADDR2LINE), so -u makes `conda activate` fail before Python runs.
set -eo pipefail

PROJECT_DIR="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship"
OUT_ROOT="$PROJECT_DIR/With_detach"
# With_detach is both the code root and the output root; the TTFS scripts are
# a subdirectory of it.
CODE_ROOT="${CODE_ROOT:-$OUT_ROOT}"
TTFS_DIR="${TTFS_DIR:-$OUT_ROOT/ttfs}"
mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/experiments/ttfs"

module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2
eval "$(conda shell.bash hook)"
conda activate torch_gpu

DATASET="${DATASET:-emnist-balanced}"
EXPT="${EXPT:-baseline}"
HIDDEN="${HIDDEN:-128}"
NGROUPS="${NGROUPS:-16}"
STEPS="${STEPS:-25}"
EPOCHS="${EPOCHS:-25}"
SEEDS="${SEEDS:-5}"
BATCH="${BATCH:-128}"
LR="${LR:-0.001}"
BETA="${BETA:-0.9}"
SLOPE="${SLOPE:-25}"
HEAD="${HEAD:-code}"
LATENCY="${LATENCY:-linear}"
TARGET_FIRE="${TARGET_FIRE:-0.9}"
HEBB_ETA="${HEBB_ETA:-0.02}"
TAU_ROUTE="${TAU_ROUTE:-0.25}"
ROUTE_NORM="${ROUTE_NORM:-zscore}"
INIT_LOGIT="${INIT_LOGIT:-0.5}"
DETACH="${DETACH:-1}"
SCHEDULE="${SCHEDULE:-interleaved}"
AE_PROB="${AE_PROB:-0.5}"
LAM="${LAM:-2.0}"
RANK_SAMPLES="${RANK_SAMPLES:-2000}"

JOBID="${SLURM_JOB_ID:-manual}"
export NAN_DATA_DIR="$PROJECT_DIR/data"
mkdir -p "$NAN_DATA_DIR"
export PYTHONPATH="$CODE_ROOT:$PYTHONPATH"
export NAN_CODE_ROOT="$CODE_ROOT"
cd "$TTFS_DIR"

# Fail immediately rather than half-way through, and say which file is missing.
for f in ttfs_nan.py train_ttfs.py; do
    [ -e "$f" ] || { echo "ERROR: $f missing from $TTFS_DIR"; exit 1; }
done
for f in datasets_ext.py collapse_metrics.py; do
    [ -e "$CODE_ROOT/$f" ] || { echo "ERROR: $f missing from $CODE_ROOT"; exit 1; }
done

TAG="$([ "$DETACH" = "1" ] && echo det || echo att)"
TAG="${TAG}_${SCHEDULE}_H${HIDDEN}_T${STEPS}_G${NGROUPS}"
[ "$ROUTE_NORM" != "zscore" ] && TAG="${TAG}_${ROUTE_NORM}"

FLAGS=""
[ "$DETACH" = "1" ] && FLAGS="$FLAGS --detach_head"
[ "$SCHEDULE" = "combined" ] && FLAGS="$FLAGS --schedule combined --lam $LAM"

echo "=============================================================="
echo "job $JOBID on $(hostname)   $(date)"
echo "expt=$EXPT  dataset=$DATASET  detach=$DETACH  schedule=$SCHEDULE"
echo "H=$HIDDEN  G=$NGROUPS  T=$STEPS  epochs=$EPOCHS  seeds=$SEEDS"
echo "route_norm=$ROUTE_NORM  tau_route=$TAU_ROUTE  init_logit=$INIT_LOGIT"
echo "code root=$CODE_ROOT"
echo "ttfs dir =$TTFS_DIR"
echo "tag=$TAG"
echo "outputs -> $OUT_ROOT/experiments/ttfs/$DATASET/$TAG/"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import snntorch, torch; print('snntorch', snntorch.__version__, '| torch', torch.__version__)"
echo

echo "--- model self-test ---"
python ttfs_nan.py
echo

run () {
    local tag="$1"; shift
    local out="$OUT_ROOT/experiments/ttfs/$DATASET/$TAG/$tag"
    mkdir -p "$out"
    echo "------------------------------------------------------------"
    echo "[$tag] $*"
    echo "[$tag] -> $out"
    echo "------------------------------------------------------------"
    python train_ttfs.py \
        --dataset "$DATASET" --hidden "$HIDDEN" --num_steps "$STEPS" \
        --epochs "$EPOCHS" --seeds "$SEEDS" --batch "$BATCH" --lr "$LR" \
        --beta "$BETA" --slope "$SLOPE" --head "$HEAD" --latency "$LATENCY" \
        --target_fire "$TARGET_FIRE" --hebb_eta "$HEBB_ETA" \
        --tau_route "$TAU_ROUTE" --route_norm "$ROUTE_NORM" \
        --ae_prob "$AE_PROB" --rank_samples "$RANK_SAMPLES" \
        $FLAGS --outdir "$out" "$@"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$JOBID" ttfs "$DATASET" "$TAG" "$tag" "$EXPT" \
        >> "$OUT_ROOT/MANIFEST.tsv"
}

case "$EXPT" in
  baseline)
    run baseline \
        --variants ttfs_ann "ttfs_group${NGROUPS}" ttfs_nan \
        --init_logit 3.0
    ;;

  routing)
    # The three routings at identical G and identical parameter count, so the
    # only difference is how the partition was chosen. init_logit 0.5 so the
    # gradient routing starts free enough to actually move a neuron.
    run "routing_G${NGROUPS}" \
        --variants "ttfs_group${NGROUPS}" \
                   "ttfs_group${NGROUPS}-coincidence" \
                   "ttfs_group${NGROUPS}-learned" \
        --init_logit "$INIT_LOGIT"
    ;;

  tau)
    # Routing sharpness. Watch route_top1 against the uniform 1/G, not
    # used_groups -- occupancy stays healthy-looking even when the routing is
    # carrying no information at all.
    for TAU in 0.05 0.1 0.25 0.5 1.0; do
      TAU_ROUTE="$TAU" run "tau${TAU}_G${NGROUPS}" \
          --variants "ttfs_group${NGROUPS}-coincidence" \
          --tau_route "$TAU" --init_logit "$INIT_LOGIT"
    done
    ;;

  sigma)
    # Now interpretable: under zscore, sigma sets only the coincidence WINDOW.
    # Under the old centred rule it also set the routing signal strength, so a
    # flat sigma sweep would have been uninformative either way.
    for SIG in 0.5 1 2 4 8; do
      run "sigma${SIG}_G${NGROUPS}" \
          --variants "ttfs_group${NGROUPS}-coincidence" \
          --sigma "$SIG" --init_logit "$INIT_LOGIT"
    done
    ;;

  steps)
    for T in 10 25 50; do
      run "steps${T}_G${NGROUPS}" \
          --variants "ttfs_group${NGROUPS}" "ttfs_group${NGROUPS}-coincidence" \
          --num_steps "$T" --init_logit "$INIT_LOGIT"
    done
    ;;

  fire)
    # How sparse the code is. Detached, sparsity is the thing that rescued the
    # rate-based NAN -- k-WTA at 10% was worth +33 points -- so the spiking
    # analogue is worth a look before reading anything into the routing.
    for TF in 0.5 0.7 0.9; do
      run "fire${TF}_G${NGROUPS}" \
          --variants "ttfs_group${NGROUPS}-coincidence" \
          --target_fire "$TF" --init_logit "$INIT_LOGIT"
    done
    ;;

  *)
    echo "ERROR: EXPT must be baseline|routing|tau|sigma|steps|fire"
    exit 1
    ;;
esac

cp -f "$OUT_ROOT/logs/ttfs_${JOBID}.log" \
      "$OUT_ROOT/logs/ttfs_${EXPT}_${DATASET}_${TAG}_${JOBID}.log" 2>/dev/null || true

echo
echo "finished $(date)"
echo "Everything under: $OUT_ROOT"
echo "=============================================================="