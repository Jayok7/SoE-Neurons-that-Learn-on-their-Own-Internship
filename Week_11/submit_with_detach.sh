#!/bin/bash
# =============================================================================
# submit_with_detach.sh — Weeks 5-10, rerun under the Bull-faithful rule.
#
# ONE script for every week, so a rerun cannot silently differ from the week it
# is replacing. WEEK selects which experiment runs; everything else is shared.
#
# EVERYTHING lands under a single tree:
#
#   $RUN_ROOT/                                     (default: .../Week_11)
#       experiments/<week>/<dataset>/<tag>/...     results JSON, PNGs, tables
#       logs/<week>_<dataset>_<tag>_<jobid>.log    stdout
#       logs/<week>_<dataset>_<tag>_<jobid>.err    stderr
#       MANIFEST.tsv                               one line per completed run
#
# so the whole set can be tarred and sent in one go:
#   tar czf week11.tar.gz -C $PROJECT_DIR Week_11
#
# RUN_ROOT must match the literal path in the SBATCH --output/--error lines.
# Both need editing together when starting a new week.
#
# -----------------------------------------------------------------------------
# WEEK
#   w56   the Week 5-6 foundation: simple_interleaved.py + simple_combined.py
#         at one width, through the original two runners. Cheap, and the only
#         place the two schedules are separate programs.
#         NOTE the schedules are NOT expected to agree under DETACH=1. They
#         touch disjoint parameter sets but deliver different numbers of
#         reconstruction updates, so ANN in particular gains about a point
#         from combined. `detach_head` is recorded in every results JSON.
#   w7    Week 7: the Fashion-MNIST width sweep + collapse analysis
#   w89   Weeks 8-9: the same on whichever DATASET is given
#   w10   Week 10: EXPT selects nodes20 | width | penalty | depth | matched
#   w11   ring routing and the minimally attached regime:
#         EXPT=ring      one overlap sweep at RING_GROUPS
#         EXPT=ringdial  the full groups x overlap grid
#         EXPT=beta      the standard variants at a given BETA
#         EXPT=pensplit  penalty sweep with decoder_within / decoder_between
#
# AXES
#   DETACH=0|1        0 = Weeks 5-10 behaviour (task gradient reaches the
#                     encoder). 1 = Bull's rule, hidden layer trained by
#                     reconstruction alone. AE variants only either way.
#   SCHEDULE          interleaved | combined | "interleaved combined"
#                     Combined equalises the task-update budget against the
#                     MLP, which is the confound flagged in every deck since
#                     Week 7 and never actually removed.
#   INHIB             none | kwta | lateral   (mutual inhibition)
#   PENALTY           none | decoder | activation | spectral
#
# NOTE the two are not independent. Under DETACH=1 the CE and reconstruction
# gradients reach disjoint parameter sets, so the schedules coincide and LAM
# becomes an encoder-decoder learning-rate multiplier rather than a trade-off.
# A combined DETACH=1 run is therefore a check, not a new condition; the
# combined runs that carry information are the DETACH=0 ones.
#
# SUBMIT
#   sbatch --export=ALL,WEEK=w56,DETACH=1,HIDDEN=20 submit_with_detach.sh
#   sbatch --export=ALL,WEEK=w7,DETACH=1,MODE=both submit_with_detach.sh
#   sbatch --export=ALL,WEEK=w89,DATASET=emnist-balanced,SCHEDULE=combined submit_with_detach.sh
#   sbatch --export=ALL,WEEK=w10,EXPT=penalty,DETACH=1,INHIB=lateral submit_with_detach.sh
#
# PREREQUISITE
#   Code and results live together under the week directory:
#
#     Week_11/
#         models.py  simple_core.py  datasets_ext.py  collapse_metrics.py
#         sweep_hidden.py  collapse_analysis.py  group_ae.py  dendritic_ae.py
#         inhibition.py
#         simple_interleaved.py  simple_combined.py     (WEEK=w56 only)
#         submit_with_detach.sh
#         ttfs/          submit_ttfs_detach.sh and the spiking scripts
#         experiments/   logs/   MANIFEST.tsv
#
#   If the week directory has no .py files the script falls back to
#   With_detach/ and says so. Set WORK_DIR to override either way.
# =============================================================================

#SBATCH --job-name=detach-rerun
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jamalidrissou2@gmail.com
# SBATCH directives cannot expand shell variables, so these paths are literal.
# EDIT BOTH when moving to a new week's directory, and create it first:
#     mkdir -p .../Week_11/logs
# Slurm writes here before the script runs, so a missing directory kills the
# job before any of it executes.
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship/Week_11/logs/run_%j.log
#SBATCH --error=/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship/Week_11/logs/run_%j.err

# NOT `set -u`: the anaconda module's activate.d scripts reference unbound
# variables (ADDR2LINE), so -u makes `conda activate` fail before Python runs.
set -eo pipefail

PROJECT_DIR="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship"

# RUN_ROOT is where logs/ and experiments/ are written. It must match the
# literal path in the SBATCH directives above.
RUN_ROOT="${RUN_ROOT:-$PROJECT_DIR/Week_11}"
OUT_ROOT="$RUN_ROOT"
LOG_DIR="$OUT_ROOT/logs"

# The code can live either alongside the outputs or stay in a shared directory.
# RUN_ROOT is preferred so a week's results and the code that produced them sit
# together; With_detach is the fallback so an existing checkout keeps working.
if [ -n "$WORK_DIR" ]; then
    :
elif [ -e "$RUN_ROOT/sweep_hidden.py" ]; then
    WORK_DIR="$RUN_ROOT"
elif [ -e "$PROJECT_DIR/With_detach/sweep_hidden.py" ]; then
    WORK_DIR="$PROJECT_DIR/With_detach"
    echo "NOTE: no code in $RUN_ROOT; using $WORK_DIR"
else
    echo "ERROR: sweep_hidden.py not found in $RUN_ROOT or $PROJECT_DIR/With_detach"
    exit 1
fi

mkdir -p "$LOG_DIR" "$OUT_ROOT/experiments"
cd "$WORK_DIR"

module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2
eval "$(conda shell.bash hook)"
conda activate torch_gpu

# ---- Overridable parameters -------------------------------------------------
WEEK="${WEEK:-w10}"                  # w56 | w7 | w89 | w10
DATASET="${DATASET:-emnist-balanced}"
EXPT="${EXPT:-width}"                # w10 only
MODE="${MODE:-sweep}"                # sweep | collapse | both
DETACH="${DETACH:-1}"                # 1 = Bull's rule
SCHEDULE="${SCHEDULE:-interleaved}"  # interleaved | combined | both
LAM="${LAM:-2.0}"
AE_PROB="${AE_PROB:-0.5}"
INHIB="${INHIB:-none}"               # none | kwta | lateral
INHIB_K_FRAC="${INHIB_K_FRAC:-0.1}"
INHIB_K_MIN="${INHIB_K_MIN:-2}"
INHIB_BOOST="${INHIB_BOOST:-1.0}"
INHIB_STRENGTH="${INHIB_STRENGTH:-1.0}"
INHIB_STEPS="${INHIB_STEPS:-3}"
INHIB_ETA="${INHIB_ETA:-0.01}"
INHIB_THETA_PLUS="${INHIB_THETA_PLUS:-0.05}"
INHIB_THETA_DECAY="${INHIB_THETA_DECAY:-0.9}"
PENALTY="${PENALTY:-none}"           # none | decoder | activation | spectral
PENALTY_WEIGHTS="${PENALTY_WEIGHTS:-0 0.01 0.1 1 10}"
# The group variant must divide PENALTY_WIDTH: group8 for 128, group5 for 20.
PENALTY_VARIANTS="${PENALTY_VARIANTS:-nan group8}"
HIDDEN="${HIDDEN:-128}"
# Must come after HIDDEN: defaulting to $HIDDEN before it exists silently
# yields an empty width.
PENALTY_WIDTH="${PENALTY_WIDTH:-$HIDDEN}"
BETA="${BETA:-1.0}"                  # task gradient reaching the hidden layer
RING_OVERLAPS="${RING_OVERLAPS:-1 2 4 8 16 32 64 128}"
RING_GROUPS="${RING_GROUPS:-128}"
HIDDEN_SIZES="${HIDDEN_SIZES:-2 4 8 16 32 64 128 256 512}"
SEEDS="${SEEDS:-5}"
EPOCHS="${EPOCHS:-25}"
BATCH="${BATCH:-128}"
LR="${LR:-0.001}"
RES="${RES:-28}"
CLASSES="${CLASSES:-}"
ACT="${ACT:-sigmoid}"
COLLAPSE_SAMPLES="${COLLAPSE_SAMPLES:-2000}"

JOBID="${SLURM_JOB_ID:-manual}"
export NAN_DATA_DIR="$PROJECT_DIR/data"
mkdir -p "$NAN_DATA_DIR"

# ---- Tag: encodes every axis, so no two conditions can collide --------------
TAG="sch-$(echo "$SCHEDULE" | tr ' ' '+')"
[ "$DETACH" = "1" ] && TAG="det_${TAG}" || TAG="att_${TAG}"
if [ "$INHIB" != "none" ]; then
    TAG="${TAG}_inh-${INHIB}"
    # k and boost MUST be in the tag: without them every k-WTA job at a given
    # week and dataset writes to the same folder and the sweep overwrites
    # itself. This is how the first k-fraction sweep was lost.
    [ "$INHIB" = "kwta" ] && TAG="${TAG}_kf${INHIB_K_FRAC}_km${INHIB_K_MIN}_b${INHIB_BOOST}"
    [ "$INHIB" = "lateral" ] && TAG="${TAG}_s${INHIB_STRENGTH}_e${INHIB_ETA}_tp${INHIB_THETA_PLUS}"
fi
[ "$PENALTY" != "none" ] && TAG="${TAG}_pen-${PENALTY}"
# HIDDEN belongs in the tag: w56 pins one width per run, so two widths under
# the same tag would overwrite each other.
[ "$WEEK" = "w56" ] && TAG="${TAG}_h${HIDDEN}"
[ "$DETACH" != "1" ] && [ "$BETA" != "1.0" ] && TAG="${TAG}_b${BETA}"

FLAGS=""
[ "$DETACH" = "1" ] && FLAGS="$FLAGS --detach_head"
[ "$DETACH" != "1" ] && [ "$BETA" != "1.0" ] && FLAGS="$FLAGS --beta $BETA"
[ "$INHIB" != "none" ] && FLAGS="$FLAGS --inhibition $INHIB \
    --inhib_k_frac $INHIB_K_FRAC --inhib_k_min $INHIB_K_MIN \
    --inhib_boost $INHIB_BOOST --inhib_strength $INHIB_STRENGTH \
    --inhib_steps $INHIB_STEPS --inhib_eta $INHIB_ETA \
    --inhib_theta_plus $INHIB_THETA_PLUS --inhib_theta_decay $INHIB_THETA_DECAY"

COMMON="--seeds $SEEDS --epochs $EPOCHS --batch_size $BATCH --lr $LR \
        --res $RES --dataset $DATASET --activation $ACT --ae_prob $AE_PROB \
        --schedule $SCHEDULE --lam $LAM --collapse_samples $COLLAPSE_SAMPLES"
[ -n "$CLASSES" ] && COMMON="$COMMON --classes $CLASSES"

echo "=============================================================="
echo "Job $JOBID on $(hostname) | $(date)"
echo "week=$WEEK dataset=$DATASET expt=$EXPT mode=$MODE"
echo "detach=$DETACH schedule=$SCHEDULE inhib=$INHIB penalty=$PENALTY"
echo "seeds=$SEEDS epochs=$EPOCHS H=$HIDDEN tag=$TAG"
[ "$EXPT" = "penalty" ] && echo "penalty: width=$PENALTY_WIDTH variants=$PENALTY_VARIANTS"
echo "code    -> $WORK_DIR"
echo "root    -> $RUN_ROOT"
echo "outputs -> $OUT_ROOT/experiments/$WEEK/$DATASET/$TAG/"
echo "=============================================================="

if [ "$DETACH" = "1" ] && [ "$SCHEDULE" = "combined" ]; then
    echo "[note] DETACH=1 with SCHEDULE=combined: parameter sets are disjoint,"
    echo "       so this should reproduce the interleaved detached numbers."
    echo "       Treat it as a consistency check, not a separate condition."
fi

# ---- Fail fast on missing files ---------------------------------------------
NEEDED="models.py simple_core.py datasets_ext.py collapse_metrics.py \
        sweep_hidden.py collapse_analysis.py group_ae.py dendritic_ae.py \
        inhibition.py"
[ "$WEEK" = "w56" ] && NEEDED="$NEEDED simple_interleaved.py simple_combined.py"
for f in $NEEDED; do
    [ -e "$f" ] || { echo "ERROR: $f missing from $WORK_DIR"; exit 1; }
done

# A pre-detach copy of sweep_hidden.py would accept every flag except the one
# that matters and run the whole sweep attached. Check for the flag itself
# rather than trusting the directory.
for f in sweep_hidden.py collapse_analysis.py; do
    grep -q -- "--detach_head" "$f" || {
        echo "ERROR: $WORK_DIR/$f has no --detach_head; this is a pre-detach"
        echo "       copy and would run the sweep ATTACHED. Copy the patched"
        echo "       files into $WORK_DIR."
        exit 1; }
done
grep -q -- "--inhib_boost" sweep_hidden.py || \
    echo "NOTE: sweep_hidden.py predates the k-WTA boost knob; INHIB_BOOST ignored."

# ---- Refuse to run on the synthetic stand-in --------------------------------
python - "$DATASET" <<'PYCHECK'
import sys
from datasets_ext import get_loaders, default_classes
name = sys.argv[1]
tr, te, in_dim = get_loaders(dataset=name, resolution=28,
                             num_classes=default_classes(name),
                             batch_size=256, verbose=True)
if len(tr.dataset) <= 2000 and len(te.dataset) <= 1000:
    print(f"ERROR: '{name}' fell back to the synthetic stand-in.")
    sys.exit(1)
print(f"OK: {name} -> {len(tr.dataset)} train / {len(te.dataset)} test")
PYCHECK

run_one () {
    local sub="$1" variants="$2" widths="$3" extra="${4:-}"
    # An empty width silently becomes `--hidden_sizes` with no argument, which
    # argparse rejects after the job has already queued and started.
    [ -n "$widths" ]   || { echo "ERROR: [$sub] no widths set"; exit 1; }
    [ -n "$variants" ] || { echo "ERROR: [$sub] no variants set"; exit 1; }
    export NAN_OUT_DIR="$OUT_ROOT/experiments/$WEEK/$DATASET/$TAG/$sub"
    mkdir -p "$NAN_OUT_DIR"
    echo "--------------------------------------------------------------"
    echo "[$sub] variants: $variants"
    echo "[$sub] widths  : $widths"
    echo "[$sub] output  : $NAN_OUT_DIR"
    echo "--------------------------------------------------------------"
    if [ "$MODE" = "sweep" ] || [ "$MODE" = "both" ]; then
        python sweep_hidden.py $COMMON $FLAGS --variants $variants \
               --hidden_sizes $widths $extra
    fi
    if [ "$MODE" = "collapse" ] || [ "$MODE" = "both" ]; then
        python collapse_analysis.py $COMMON $FLAGS --variants $variants \
               --hidden "$HIDDEN"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$JOBID" "$WEEK" "$DATASET" "$TAG" "$sub" "$SCHEDULE" \
        "$DETACH" "$INHIB" "$PENALTY" >> "$OUT_ROOT/MANIFEST.tsv"
}

ALL_VARIANTS="linear frozen mlp nan ann"
NODES20_VARIANTS="nan group10 group5 group4 group2 ann"
WIDTH_VARIANTS="nan group64 group32 group16 group8 group4 group2 ann"
DEPTH_VARIANTS="nan dend2 dend4 dend8"
MATCHED_WIDTHS="128 214 385 727"

case "$WEEK" in
  w56)
    # The foundation, at ONE width, through the original two runners.
    # The two schedules are NOT expected to match under DETACH=1 -- see header.
    export NAN_OUT_DIR="$OUT_ROOT/experiments/w56/$DATASET/$TAG"
    mkdir -p "$NAN_OUT_DIR"
    W56="--seeds $SEEDS --epochs $EPOCHS --batch_size $BATCH --lr $LR \
         --hidden $HIDDEN --res $RES --classes ${CLASSES:-10} --activation $ACT \
         --variants $ALL_VARIANTS"
    W56_FLAGS=""
    [ "$DETACH" = "1" ] && W56_FLAGS="--detach_head"
    python simple_interleaved.py $W56 $W56_FLAGS --ae_prob "$AE_PROB"
    python simple_combined.py    $W56 $W56_FLAGS --lam "$LAM"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$JOBID" w56 "$DATASET" "$TAG" both "$SCHEDULE" "$DETACH" \
        "$INHIB" "$PENALTY" >> "$OUT_ROOT/MANIFEST.tsv"
    ;;

  w7|w89)
    # Same code path; w7 is just DATASET=fashion. Kept as separate WEEK values
    # so the output tree matches how the reports are organised.
    run_one sweep "$ALL_VARIANTS" "$HIDDEN_SIZES"
    ;;

  w10)
    PEN_ARGS="--penalty $PENALTY --penalty_sweep $PENALTY_WEIGHTS"
    case "$EXPT" in
      nodes20) run_one nodes20 "$NODES20_VARIANTS" "20" ;;
      width)   run_one width   "$WIDTH_VARIANTS"   "$HIDDEN" ;;
      penalty) run_one penalty "$PENALTY_VARIANTS" "$PENALTY_WIDTH" "$PEN_ARGS" ;;
      depth)   run_one depth   "$DEPTH_VARIANTS"   "$HIDDEN" ;;
      matched) run_one matched "nan"               "$MATCHED_WIDTHS" ;;
      all)     run_one nodes20 "$NODES20_VARIANTS" "20"
               run_one width   "$WIDTH_VARIANTS"   "$HIDDEN"
               run_one depth   "$DEPTH_VARIANTS"   "$HIDDEN"
               run_one matched "nan"               "$MATCHED_WIDTHS" ;;
      *) echo "ERROR: EXPT must be nodes20|width|penalty|depth|matched|all"; exit 1 ;;
    esac
    ;;

  w11)
    # Ring routing and the minimally attached regime. Same output tree as the
    # rest, under experiments/w11/.
    RING_VARIANTS=""
    for O in $RING_OVERLAPS; do RING_VARIANTS="$RING_VARIANTS ring${RING_GROUPS}o${O}"; done
    case "$EXPT" in
      ring)      run_one "ring_G${RING_GROUPS}" "ann$RING_VARIANTS nan" "$HIDDEN" ;;
      ringdial)  run_one "ringdial" \
                     "ann ring2o1 ring2o2 ring4o1 ring4o2 ring4o4 \
                      ring8o1 ring8o2 ring8o4 ring8o8 \
                      ring32o1 ring32o2 ring32o8 ring32o32 \
                      ring128o1 ring128o2 ring128o8 ring128o32 ring128o128 nan" \
                     "$HIDDEN" ;;
      beta)      run_one "beta${BETA}" "$ALL_VARIANTS" "$HIDDEN" ;;
      pensplit)  run_one "pensplit" "$PENALTY_VARIANTS" "$PENALTY_WIDTH" \
                     "--penalty $PENALTY --penalty_sweep $PENALTY_WEIGHTS" ;;
      *) echo "ERROR: for WEEK=w11, EXPT must be ring|ringdial|beta|pensplit"; exit 1 ;;
    esac
    ;;

  *) echo "ERROR: WEEK must be w56 | w7 | w89 | w10 | w11"; exit 1 ;;
esac

# ---- Tag the logs so they are identifiable inside the single log dir ---------
cp -f "$LOG_DIR/run_${JOBID}.log" \
      "$LOG_DIR/${WEEK}_${DATASET}_${TAG}_${JOBID}.log" 2>/dev/null || true
cp -f "$LOG_DIR/run_${JOBID}.err" \
      "$LOG_DIR/${WEEK}_${DATASET}_${TAG}_${JOBID}.err" 2>/dev/null || true

echo
echo "Finished: $(date)"
echo "Everything under: $OUT_ROOT"
echo "Package with:  tar czf $(basename "$RUN_ROOT").tar.gz -C $PROJECT_DIR $(basename "$RUN_ROOT")"
echo "=============================================================="
