#!/bin/bash
# =============================================================================
# submit_with_detach.sh — Weeks 5-10, rerun under the Bull-faithful rule.
#
# ONE script for every week, so a rerun cannot silently differ from the week it
# is replacing. WEEK selects which experiment runs; everything else is shared.
#
# EVERYTHING lands under a single tree:
#
#   $ROOT/With_detach/
#       experiments/<week>/<dataset>/<tag>/...     results JSON, PNGs, tables
#       logs/<week>_<dataset>_<tag>_<jobid>.log    stdout
#       logs/<week>_<dataset>_<tag>_<jobid>.err    stderr
#       MANIFEST.tsv                               one line per completed run
#
# so the whole set can be tarred and sent in one go:
#   tar czf with_detach.tar.gz -C $ROOT With_detach
#
# -----------------------------------------------------------------------------
# WEEK
#   w56   the Week 5-6 foundation: simple_interleaved.py + simple_combined.py
#         at one width. THE CORRECTNESS CHECK — under DETACH=1 the two scripts
#         have disjoint parameter sets and must agree to within seed noise.
#         Run this first and read it before submitting anything else.
#   w7    Week 7: the Fashion-MNIST width sweep + collapse analysis
#   w89   Weeks 8-9: the same on whichever DATASET is given
#   w10   Week 10: EXPT selects nodes20 | width | penalty | depth | matched
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
#   $WORK_DIR must contain: models.py simple_core.py datasets_ext.py
#   collapse_metrics.py sweep_hidden.py collapse_analysis.py group_ae.py
#   dendritic_ae.py inhibition.py  — and for WEEK=w56 also
#   simple_interleaved.py simple_combined.py
# =============================================================================

#SBATCH --job-name=detach-rerun
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship/With_detach/logs/run_%j.log
#SBATCH --error=/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship/With_detach/logs/run_%j.err

# NOT `set -u`: the anaconda module's activate.d scripts reference unbound
# variables (ADDR2LINE), so -u makes `conda activate` fail before Python runs.
set -eo pipefail

PROJECT_DIR="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship"
OUT_ROOT="$PROJECT_DIR/With_detach"
LOG_DIR="$OUT_ROOT/logs"
WORK_DIR="${WORK_DIR:-$PROJECT_DIR/With_detach}"  # self-contained: all deps live here
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
INHIB_STRENGTH="${INHIB_STRENGTH:-1.0}"
INHIB_STEPS="${INHIB_STEPS:-3}"
INHIB_ETA="${INHIB_ETA:-0.01}"
INHIB_THETA_PLUS="${INHIB_THETA_PLUS:-0.05}"
INHIB_THETA_DECAY="${INHIB_THETA_DECAY:-0.9}"
PENALTY="${PENALTY:-none}"           # none | decoder | activation | spectral
PENALTY_WEIGHTS="${PENALTY_WEIGHTS:-0 0.01 0.1 1 10}"
HIDDEN="${HIDDEN:-128}"
HIDDEN_SIZES="${HIDDEN_SIZES:-2 4 8 16 32 64 128 256 512}"
SEEDS="${SEEDS:-5}"
EPOCHS="${EPOCHS:-15}"
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
[ "$INHIB" != "none" ] && TAG="${TAG}_inh-${INHIB}"
[ "$PENALTY" != "none" ] && TAG="${TAG}_pen-${PENALTY}"

FLAGS=""
[ "$DETACH" = "1" ] && FLAGS="$FLAGS --detach_head"
[ "$INHIB" != "none" ] && FLAGS="$FLAGS --inhibition $INHIB \
    --inhib_k_frac $INHIB_K_FRAC --inhib_strength $INHIB_STRENGTH \
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
    # The foundation, at ONE width, through the original two runners. This is
    # the only place the interleaved/combined equivalence check can be made
    # cleanly, because the two scripts are separate programs.
    export NAN_OUT_DIR="$OUT_ROOT/experiments/w56/$DATASET/$TAG"
    mkdir -p "$NAN_OUT_DIR"
    W56="--seeds $SEEDS --epochs $EPOCHS --batch_size $BATCH --lr $LR \
         --hidden $HIDDEN --res $RES --classes ${CLASSES:-10} --activation $ACT \
         --variants $ALL_VARIANTS"
    W56_FLAGS=""
    [ "$DETACH" = "1" ] && W56_FLAGS="--detach_head"
    python simple_interleaved.py $W56 $W56_FLAGS --ae_prob "$AE_PROB"
    python simple_combined.py    $W56 $W56_FLAGS --lam "$LAM"
    if [ "$DETACH" = "1" ]; then
        echo
        echo "=== EQUIVALENCE CHECK (detached: schedules should coincide) ==="
        python - "$NAN_OUT_DIR" <<'PYEQ'
import json, os, sys
d = sys.argv[1]
a = json.load(open(os.path.join(d, "simple_interleaved_det",
                                "results_interleaved.json")))["curves"]
b = json.load(open(os.path.join(d, "simple_combined_det",
                                "results_combined.json")))["curves"]
bad = False
for v in sorted(set(a) & set(b)):
    x, y = a[v]["acc"]["mean"][-1], b[v]["acc"]["mean"][-1]
    s = max(a[v]["acc"]["sem"][-1], b[v]["acc"]["sem"][-1], 1e-9)
    flag = "" if abs(x - y) <= 3 * s + 0.3 else "  <== INVESTIGATE"
    bad |= bool(flag)
    print(f"  {v:12} interleaved {x:6.2f}  combined {y:6.2f}  "
          f"diff {x - y:+.2f} (sem {s:.2f}){flag}")
print("\n  RESULT:", "MISMATCH — task gradient may still be reaching the "
      "encoder. Do not trust downstream runs." if bad else
      "consistent; the detach is clean.")
PYEQ
    fi
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
      penalty) run_one penalty "nan group5"        "20" "$PEN_ARGS" ;;
      depth)   run_one depth   "$DEPTH_VARIANTS"   "$HIDDEN" ;;
      matched) run_one matched "nan"               "$MATCHED_WIDTHS" ;;
      all)     run_one nodes20 "$NODES20_VARIANTS" "20"
               run_one width   "$WIDTH_VARIANTS"   "$HIDDEN"
               run_one depth   "$DEPTH_VARIANTS"   "$HIDDEN"
               run_one matched "nan"               "$MATCHED_WIDTHS" ;;
      *) echo "ERROR: EXPT must be nodes20|width|penalty|depth|matched|all"; exit 1 ;;
    esac
    ;;

  *) echo "ERROR: WEEK must be w56 | w7 | w89 | w10"; exit 1 ;;
esac

# ---- Tag the logs so they are identifiable inside the single log dir ---------
cp -f "$LOG_DIR/run_${JOBID}.log" \
      "$LOG_DIR/${WEEK}_${DATASET}_${TAG}_${JOBID}.log" 2>/dev/null || true
cp -f "$LOG_DIR/run_${JOBID}.err" \
      "$LOG_DIR/${WEEK}_${DATASET}_${TAG}_${JOBID}.err" 2>/dev/null || true

echo
echo "Finished: $(date)"
echo "Everything under: $OUT_ROOT"
echo "Package with:  tar czf with_detach.tar.gz -C $PROJECT_DIR With_detach"
echo "=============================================================="
