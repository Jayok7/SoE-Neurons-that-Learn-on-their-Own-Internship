#!/bin/bash
# =============================================================================
# submit_experiments.sh  —  submit with:  sbatch submit_experiments.sh
# -----------------------------------------------------------------------------
# Runs the full set of neuron-autoencoder experiments on one L40S GPU and writes
# all curves/JSON/plots to ./experiments/. Designed to be the single job you
# launch to get every figure for the write-up in one shot.
#
# Prerequisite: run `bash setup_env.sh` ONCE first (creates the env, caches MNIST).
# =============================================================================

#SBATCH --job-name=nan-sweeps
#SBATCH --partition=gpuL                 # L40S, usually available (see guide)
#SBATCH --gres=gpu:1                     # one GPU is plenty for these MLPs
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00                  # generous; spiking sweeps are the slow part
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/nan_%j.log
#SBATCH --error=logs/nan_%j.err

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
set -e
PROJECT_DIR="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-intership"
cd "$PROJECT_DIR"
mkdir -p logs experiments

module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2
eval "$(conda shell.bash hook)"
conda activate torch_gpu

# Tell the harness where to read MNIST and where to write results (these env
# vars are read by stage7_experiment_harness.py; no need to edit the .py file).
export NAN_DATA_DIR="$PROJECT_DIR/data"
export NAN_OUT_DIR="$PROJECT_DIR/experiments"

echo "=========================================="
echo "Job: $SLURM_JOB_NAME | ID: $SLURM_JOB_ID"
echo "Node: $(hostname) | Started: $(date)"
echo "=========================================="
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

SCRIPT=stage7_experiment_harness.py
SEEDS=10            # 10 seeds for publishable error bars (paper uses 20; bump if time allows)
EPOCHS=15           # full-resolution MNIST converges well within this

# -----------------------------------------------------------------------------
# EXPERIMENT 1 — the headline collapse story: per-epoch curves for all four
# architectures at full resolution. Produces curve_acc_*.png and curve_rank_*.png,
# the effective-rank-vs-epoch figure is your strongest result.
# -----------------------------------------------------------------------------
echo ">>> [1/6] curves: static, spiking, ann, nn"
for ARCH in static spiking ann nn; do
    echo "    curves arch=$ARCH"
    python "$SCRIPT" --mode curves --arch "$ARCH" \
        --seeds "$SEEDS" --epochs "$EPOCHS" --res 28 --classes 10 --hidden 128
done

# -----------------------------------------------------------------------------
# EXPERIMENT 2 — diversity-penalty sweep (the anti-collapse knob). Accuracy and
# effective rank vs div_weight, for both NAN variants. This is where you find
# whether there is a sweet spot or just a plateau.
# -----------------------------------------------------------------------------
echo ">>> [2/6] diversity-weight sweep: static, spiking"
for ARCH in static spiking; do
    echo "    div_sweep arch=$ARCH"
    python "$SCRIPT" --mode div_sweep --arch "$ARCH" \
        --seeds "$SEEDS" --epochs "$EPOCHS" --res 28 --classes 10 --hidden 128
done

# -----------------------------------------------------------------------------
# EXPERIMENT 3 — local-weight sweep (task vs reconstruction trade-off). The
# variable most likely to reveal any benefit of neuron-AE if one exists.
# -----------------------------------------------------------------------------
echo ">>> [3/6] local-weight sweep: static, spiking"
for ARCH in static spiking; do
    echo "    local_sweep arch=$ARCH"
    python "$SCRIPT" --mode local_sweep --arch "$ARCH" \
        --seeds "$SEEDS" --epochs "$EPOCHS" --res 28 --classes 10 --hidden 128
done

# -----------------------------------------------------------------------------
# EXPERIMENT 4 — sequential schedule 2x2 (freeze/finetune x penalty/no-penalty),
# for both NAN variants. Answers whether fine-tuning and/or the penalty rescue
# the collapse seen under the frozen-sequential schedule.
# -----------------------------------------------------------------------------
echo ">>> [4/6] sequential variants: static, spiking"
for ARCH in static spiking; do
    echo "    seq_variants arch=$ARCH"
    python "$SCRIPT" --mode seq_variants --arch "$ARCH" \
        --seeds "$SEEDS" --epochs "$EPOCHS" --res 28 --classes 10 --hidden 128
done

# -----------------------------------------------------------------------------
# EXPERIMENT 5 — the headline architecture comparison table: NN vs ANN vs
# NAN(static) vs NAN(spiking), one row each (the Bull-style comparison + spiking).
# -----------------------------------------------------------------------------
echo ">>> [5/6] four-architecture comparison table"
python "$SCRIPT" --mode compare_all4 \
    --seeds "$SEEDS" --epochs "$EPOCHS" --res 28 --classes 10 --hidden 128

# -----------------------------------------------------------------------------
# EXPERIMENT 6 — neuron-level visualizations: decoder weight fields (what each
# neuron reconstructs) and activation maps (what inputs excite each neuron).
# Produces a grid of panels per architecture — the visual proof that neurons
# are (or aren't) doing independent work.
# -----------------------------------------------------------------------------
echo ">>> [6/6] neuron visualizations (all architectures)"
python "$SCRIPT" --mode visualize_all \
    --epochs "$EPOCHS" --res 28 --classes 10 --hidden 128

echo "=========================================="
echo "All experiments done — $(date)"
echo "Results in: $NAN_OUT_DIR"
echo "=========================================="
