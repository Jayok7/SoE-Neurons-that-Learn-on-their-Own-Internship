# Week 4 — Fashion-MNIST

A Fashion-MNIST run of the neuron-autoencoder harness, carried over from the Week 2–3
MNIST setup. All the main structural elements are preserved; only the dataset was swapped.

## What changed vs the MNIST version

Fashion-MNIST is a deliberate drop-in replacement for MNIST — same 60k/10k
split, same 28×28 single-channel images, same 10 classes — so the swap touches
exactly one functional line and nothing about the model, training, metrics, or
plotting had to change.

| File | Change |
|------|--------|
| `models.py` | `datasets.MNIST` → `datasets.FashionMNIST` in `get_loaders`; added the `FASHION_CLASSES` label table; updated docstrings/comments. |
| `run_experiments.py` | Two `--help` strings reworded (digit→garment). Logic identical. |
| `experiments.py` | **Unchanged** (byte-identical). Every function is generic over `resolution` / `num_classes` / `in_dim`, so all four architectures (nn/ann/static/spiking), all schedules, all diagnostics, and both visualisers work as-is. |
| `submit_penalised_compare.sh` | Points at `Week_4/`, writes to `experiments_fashion_h{HIDDEN}/`, runs at `--res 28 --classes 10`. |
| `setup_env_week4.sh` | Downloads Fashion-MNIST (different mirror, `FashionMNIST/raw/` folder) instead of MNIST. |

Why nothing else needed changing: the encoder input dimension is
`resolution*resolution`, the classifier head width is `num_classes`, and every
reshape in the decoder/activation visualisers uses `resolution` — all of which
are identical between the two datasets. The only thing that differs downstream
is the *meaning* of each class (garments instead of digits) and the pixels
themselves.

## New: Training Speed Comparison (ANN vs NAN)

A new standalone script (`train_speed_compare.py`) was added specifically to
measure the per-epoch training cost of each architecture:

| File | Purpose |
|------|---------|
| `train_speed_compare.py` | Trains each architecture end-to-end, records wall-clock time per epoch, accuracy, top-2, recon MSE, and macro-F1. Produces per-arch training curves and a combined speed summary. |
| `submit_speed_compare.sh` | Slurm job script — submits the speed comparison on one L40S GPU. |

### Outputs (in `experiments_fashion_h<H>/speed_compare/`)

- `history_<arch>.json` — full per-epoch history including `epoch_time`
- `history_<arch>.png` — training curves (loss | accuracy+top2 | recon MSE)
- `predictions_<arch>.png` — sample prediction grid (green=correct, red=wrong)
- `ann_vs_nan_speed.json` — summary table (mean epoch time, final val_acc, top-2, macro-F1)
- `ann_vs_nan_speed.png` — bar chart of mean seconds/epoch per architecture

## Files in Week_4

| File | Description |
|------|-------------|
| `models.py` | Architectures, Fashion-MNIST loader, diversity penalties |
| `experiments.py` | Schedules, metrics, plotting (unchanged from Week 2–3) |
| `run_experiments.py` | CLI entrypoint for the experiment harness |
| `train_speed_compare.py` | ANN vs NAN training-speed comparison script |
| `submit_speed_compare.sh` | Slurm job for speed comparison |
| `submit_penalised_compare.sh` | Slurm job for penalised schedule comparison |
| `submit_experiments.sh` | Slurm job for full experiment sweep |
| `setup_env_week4.sh` | One-time env + Fashion-MNIST download setup |

## Running

```bash
# once, on the CSF login node:
cd ~/autoencoder-internship/Week_4
bash setup_env_week4.sh

# submit jobs:
sbatch submit_speed_compare.sh                          # speed comparison (default: h=128)
sbatch submit_penalised_compare.sh                      # penalised schedule comparison

# speed comparison with overrides:
sbatch --export=ALL,HIDDEN=20 submit_speed_compare.sh   # 20-neuron run
sbatch --export=ALL,ARCHS="nn ann static spiking" submit_speed_compare.sh  # include spiking

# single interactive run:
export NAN_DATA_DIR=$HOME/autoencoder-internship/data
export NAN_OUT_DIR=$HOME/autoencoder-internship/experiments_fashion_h128
python run_experiments.py --mode curves --arch spiking --res 28 --classes 10
```

All CLI flags are unchanged: `--mode`, `--arch`, `--seeds`, `--epochs`,
`--res` (7/14/28), `--classes` (2..10), `--hidden`, `--penalty`, `--div_weight`.
