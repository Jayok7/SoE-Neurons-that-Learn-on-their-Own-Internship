#!/bin/bash
# =============================================================================
# submit_extra_sweeps.sh  (Week 4 / Fashion-MNIST)  -  submit with:
#     sbatch submit_extra_sweeps.sh                                    # penalty_recon, static, h=128
#     sbatch --export=ALL,MODE=ae_prob submit_extra_sweeps.sh          # AE-probability sweep
#     sbatch --export=ALL,MODE=all,ARCH=spiking submit_extra_sweeps.sh # everything, spiking
#     sbatch --export=ALL,HIDDEN=20 submit_extra_sweeps.sh             # 20-neuron run
#     sbatch --export=ALL,MODE=combined,CONFIRM_COMBINED=1 submit_extra_sweeps.sh
#     sbatch --export=ALL,MODE=combined,CONFIRM_COMBINED=1,ARCH=ann,HIDDEN=20,OPTIM=sgd,LR=0.1 submit_extra_sweeps.sh   # replicate Isaac's Recon Scaler
# -----------------------------------------------------------------------------
# Runs extra_sweeps.py - the follow-up sweeps from the sweep analysis:
#   penalty_recon : div_weight sweep for BOTH penalties (decoder + activation)
#                   vs final recon_acc / test acc / effective rank (overlaid)
#   ae_prob       : AE-cycle probability sweep (the real schedule-pressure knob)
#   combined      : CE + lambda*recon single-loss sweep (needs CONFIRM_COMBINED=1)
#   all           : all of the above
#
# Outputs land in:
#   experiments_fashion_h<HIDDEN>/extra_sweeps/penalty_recon_sweep/
#   experiments_fashion_h<HIDDEN>/extra_sweeps/ae_prob_sweep/
#   experiments_fashion_h<HIDDEN>/extra_sweeps/combined_sweep/
# so results are grouped in one place and never mix across hidden widths.
#
# Prerequisite: `bash setup_env_week4.sh` has been run ONCE on the login node
# (torch_gpu env + Fashion-MNIST cache). No extra deps beyond the harness.
# =============================================================================

#SBATCH --job-name=fashion-extra-sweeps
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/extra_sweeps_%j.log
#SBATCH --error=logs/extra_sweeps_%j.err

set -e
PROJECT_DIR="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship"
cd "$PROJECT_DIR/Week_4"
mkdir -p logs

module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2
eval "$(conda shell.bash hook)"
conda activate torch_gpu

# Fashion-MNIST location (loader reads FashionMNIST/raw from here).
export NAN_DATA_DIR="$PROJECT_DIR/data"

# ---- Overridable parameters -------------------------------------------------
MODE="${MODE:-penalty_recon}"        # penalty_recon | ae_prob | combined | all
ARCH="${ARCH:-static}"               # static | spiking | ann | nn
HIDDEN="${HIDDEN:-128}"
SEEDS="${SEEDS:-5}"
EPOCHS="${EPOCHS:-9}"                # sweep runs: ~60% of a full 15-epoch run
RES="${RES:-14}"
CLASSES="${CLASSES:-10}"
CONFIRM_COMBINED="${CONFIRM_COMBINED:-0}"   # set to 1 to allow the combined sweep
OPTIM="${OPTIM:-adam}"               # adam (harness default) | sgd (Isaac's W2-3 setup)
LR="${LR:-0.001}"                    # Isaac's sgd runs used LR=0.1

# ---- Output separation ------------------------------------------------------
# One dedicated extra_sweeps/ folder per hidden width. extra_sweeps.py then
# creates its own per-experiment subdirs (penalty_recon_sweep/, ae_prob_sweep/,
# combined_sweep/) inside it, and tags every file with the arch, so runs for
# different widths/archs/modes never collide.
export NAN_OUT_DIR="$PROJECT_DIR/experiments_fashion_h${HIDDEN}/extra_sweeps"
mkdir -p "$NAN_OUT_DIR"

# The combined-loss sweep is opt-in (mirrors the script's --confirm_combined gate).
EXTRA_FLAGS=""
if [ "$CONFIRM_COMBINED" = "1" ]; then
    EXTRA_FLAGS="--confirm_combined"
fi

echo "=========================================="
echo "Job: $SLURM_JOB_NAME | ID: $SLURM_JOB_ID"
echo "Node: $(hostname) | Started: $(date)"
echo "mode=$MODE arch=$ARCH hidden=$HIDDEN optim=$OPTIM lr=$LR seeds=$SEEDS epochs=$EPOCHS res=$RES"
echo "output -> $NAN_OUT_DIR"
echo

python extra_sweeps.py \
    --mode "$MODE" \
    --arch "$ARCH" \
    --seeds "$SEEDS" \
    --epochs "$EPOCHS" \
    --res "$RES" \
    --classes "$CLASSES" \
    --hidden "$HIDDEN" \
    --optim "$OPTIM" \
    --lr "$LR" \
    $EXTRA_FLAGS

echo
echo "Finished: $(date)"
echo "=========================================="
