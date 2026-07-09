#!/bin/bash
# =============================================================================
# submit_speed_compare.sh  (Week 4 / Fashion-MNIST)  -  submit with:
#     sbatch submit_speed_compare.sh                          # HIDDEN=128 (default)
#     sbatch --export=ALL,HIDDEN=20 submit_speed_compare.sh   # 20-neuron run
#     sbatch --export=ALL,HIDDEN=128,ARCHS="nn ann static spiking" submit_speed_compare.sh
# -----------------------------------------------------------------------------
# Runs train_speed_compare.py - the ANN-vs-NAN per-epoch training-speed
# comparison - on one L40S GPU. Writes per-arch history JSON/PNG plus the
# combined ann_vs_nan_speed.{png,json} into experiments_fashion_h<H>/speed_compare/.
#
# Prerequisite: run `bash setup_env_week4.sh` ONCE first
# (creates the env, caches Fashion-MNIST). scikit-learn is needed for the
# macro-F1 metric - the env check below installs it if missing.
# =============================================================================

#SBATCH --job-name=fashion-speed-cmp
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jamalidrissou2@gmail.com
#SBATCH --output=logs/fashion_speed_%j.log
#SBATCH --error=logs/fashion_speed_%j.err

set -e
PROJECT_DIR="/mnt/iusers01/fse-ugpgt01/eee01/m84149ji/autoencoder-internship"
cd "$PROJECT_DIR/Week_4"
mkdir -p logs

module load apps/binapps/anaconda3/2021.11
module load cuda/12.6.2
eval "$(conda shell.bash hook)"
conda activate torch_gpu

# scikit-learn is the one extra dep this script needs (macro-F1).
python -c "import sklearn" 2>/dev/null || pip install scikit-learn

# Fashion-MNIST location (loader reads FashionMNIST/raw from here).
export NAN_DATA_DIR="$PROJECT_DIR/data"

# Overridable parameters
HIDDEN="${HIDDEN:-128}"
ARCHS="${ARCHS:-nn ann static}"     # nn = no-AE baseline, ann = shared decoder, static = NAN
EPOCHS="${EPOCHS:-15}"

# Dedicated speed_compare folder under the per-hidden results dir so these
# files never mix with the harness's penalised_compare / curves output.
export NAN_OUT_DIR="$PROJECT_DIR/experiments_fashion_h${HIDDEN}/speed_compare"
mkdir -p "$NAN_OUT_DIR"

echo "=========================================="
echo "Job: $SLURM_JOB_NAME | ID: $SLURM_JOB_ID"
echo "Node: $(hostname) | Started: $(date)"
echo

python train_speed_compare.py \
    --archs $ARCHS \
    --hidden "$HIDDEN" \
    --epochs "$EPOCHS"

echo
echo "Finished: $(date)"
echo "=========================================="
