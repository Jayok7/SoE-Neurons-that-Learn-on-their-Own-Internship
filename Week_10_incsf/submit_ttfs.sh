#!/bin/bash
# =============================================================================
# submit_ttfs.sh -- time-to-first-spike NAN, with grouping by spike coincidence.
#
# THE EXPERIMENTS
#
#   EXPT=baseline   Does a TTFS layer work at all, and does the fixed group
#                   dial behave the way the rate-based one did? Runs
#                   ttfs_nan, ttfs_group<G>, ttfs_ann at fixed routing.
#                   Run this FIRST -- everything else assumes the spiking
#                   layer trains.
#
#   EXPT=routing    THE EXPERIMENT. Same G, three ways of choosing the
#                   partition: fixed block (control), coincidence (the Hebbian
#                   spike-timing rule), and learned (gradient). Identical
#                   parameter count and identical group count, so any
#                   difference is attributable to HOW the partition was chosen.
#                   Started at --init_logit 0.5 so the two dynamic routings
#                   begin equally uncommitted; at the default 3.0 the gradient
#                   routing cannot flip a neuron and the comparison is rigged.
#
#   EXPT=sigma      How wide a coincidence window to call "together". sigma is
#                   in timesteps; too small and no two neurons ever count as
#                   coincident, too large and everything does. Cheap sweep,
#                   worth doing before reading anything into the routing run.
#
#   EXPT=steps      Time resolution. A TTFS code with T steps can express T
#                   distinct latencies, so T caps how much the layer can say.
#                   Establishes where that stops being the binding constraint.
#
# SUBMIT WITH
#     sbatch --export=ALL,DATASET=emnist-balanced,EXPT=baseline submit_ttfs.sh
#     sbatch --export=ALL,DATASET=emnist-balanced,EXPT=routing  submit_ttfs.sh
#     sbatch --export=ALL,DATASET=emnist-balanced,EXPT=sigma    submit_ttfs.sh
#     sbatch --export=ALL,DATASET=emnist-balanced,EXPT=steps    submit_ttfs.sh
#
# INTERLEAVED ONLY
#   Matching Week 10 and Weeks 5-9. There is no schedule knob: two schedules on
#   one axis would be two different training budgets drawn as if comparable.
#
# COST
#   A TTFS forward pass runs the layer T times, so a T=25 run costs roughly 25
#   forward passes per batch against 1 for the rate-based model. Budget an
#   order of magnitude more time than the equivalent Week 10 sweep, and start
#   with EPOCHS small.
#
# PREREQUISITES
#   Week_10/ next to this directory (datasets_ext.py, collapse_metrics.py).
#   snntorch in the environment:  pip install snntorch
# =============================================================================
#SBATCH --job-name=ttfs
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ttfs_%j.log

# NOT `set -u`: the anaconda module's activate.d scripts reference unbound
# variables (ADDR2LINE), so -u makes `conda activate` fail before Python runs.
set -eo pipefail
mkdir -p logs

module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2
eval "$(conda shell.bash hook)"
conda activate torch_gpu

DATASET="${DATASET:-emnist-balanced}"
EXPT="${EXPT:-baseline}"
HIDDEN="${HIDDEN:-128}"
NGROUPS="${NGROUPS:-16}"
STEPS="${STEPS:-25}"
EPOCHS="${EPOCHS:-15}"
SEEDS="${SEEDS:-3}"
BATCH="${BATCH:-128}"
LR="${LR:-0.001}"
BETA="${BETA:-0.9}"
SLOPE="${SLOPE:-25}"
HEAD="${HEAD:-code}"
LATENCY="${LATENCY:-linear}"
TARGET_FIRE="${TARGET_FIRE:-0.9}"
HEBB_ETA="${HEBB_ETA:-0.02}"
TAU_ROUTE="${TAU_ROUTE:-0.5}"
INIT_LOGIT="${INIT_LOGIT:-3.0}"
RANK_SAMPLES="${RANK_SAMPLES:-2000}"

cd /mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship/Week_10

echo "============================================================"
echo "job     : ${SLURM_JOB_ID:-manual}   $(date)"
echo "expt    : $EXPT"
echo "dataset : $DATASET"
echo "H=$HIDDEN  G=$NGROUPS  T=$STEPS  epochs=$EPOCHS  seeds=$SEEDS"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "
import sys, site
print('Python:', sys.version)
print('Executable:', sys.executable)
print('ENABLE_USER_SITE:', site.ENABLE_USER_SITE)
print('USER_SITE:', site.getusersitepackages())
print('sys.path:')
for p in sys.path: print('  ', p)
import snntorch
print('snntorch', snntorch.__version__)
"
echo

# Fail loudly on a silent layer or a broken gradient path before spending a
# day of GPU time on it.
echo "--- model self-test ---"
python ttfs_nan.py
echo

run () {
    local tag="$1"; shift
    echo "------------------------------------------------------------"
    echo "[$tag] $*"
    echo "------------------------------------------------------------"
    python train_ttfs.py \
        --dataset "$DATASET" --hidden "$HIDDEN" --num_steps "$STEPS" \
        --epochs "$EPOCHS" --seeds "$SEEDS" --batch "$BATCH" --lr "$LR" \
        --beta "$BETA" --slope "$SLOPE" --head "$HEAD" --latency "$LATENCY" \
        --target_fire "$TARGET_FIRE" --hebb_eta "$HEBB_ETA" \
        --tau_route "$TAU_ROUTE" --rank_samples "$RANK_SAMPLES" \
        --outdir "results/${DATASET}_${tag}" "$@"
}

case "$EXPT" in
  baseline)
    run "baseline_H${HIDDEN}_T${STEPS}" \
        --variants ttfs_ann "ttfs_group${NGROUPS}" ttfs_nan \
        --init_logit "$INIT_LOGIT"
    ;;

  routing)
    run "routing_G${NGROUPS}_T${STEPS}" \
        --variants "ttfs_group${NGROUPS}" \
                   "ttfs_group${NGROUPS}-coincidence" \
                   "ttfs_group${NGROUPS}-learned" \
        --init_logit 0.5
    ;;

  sigma)
    for SIG in 0.5 1 2 4 8; do
      run "sigma${SIG}_G${NGROUPS}_T${STEPS}" \
          --variants "ttfs_group${NGROUPS}-coincidence" \
          --sigma "$SIG" --init_logit 0.5
    done
    ;;

  steps)
    for T in 10 25 50; do
      # --num_steps is what the run actually uses; the tag records it too.
      run "steps${T}_G${NGROUPS}" \
          --variants "ttfs_group${NGROUPS}" "ttfs_group${NGROUPS}-coincidence" \
          --num_steps "$T" --init_logit 0.5
    done
    ;;

  *)
    echo "ERROR: EXPT must be baseline | routing | sigma | steps"
    exit 1
    ;;
esac

echo
echo "finished $(date)"

