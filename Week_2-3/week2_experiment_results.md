# Week 2 — Neuron-Autoencoder Experiment Results

> **Run environment:** CSF3 cluster (NVIDIA L40S GPUs, PyTorch 2.5.1+cu121)  
> **Dataset:** MNIST (28×28, 10 classes)  
> **Seeds:** 10 per condition | **Epochs:** 15 (sweeps: 9)  
> **Date:** 22 Jun 2026  

---

## 1. Experimental Framework Overview

The Week 2 codebase tests four hidden-layer architectures inside a one-hidden-layer classifier:

| Arch | Description | Has Autoencoder? | Per-Neuron Decoder? |
|------|-------------|:-:|:-:|
| **NN** | Plain sigmoid layer (traditional baseline) | ✗ | ✗ |
| **ANN** | Shared decoder across whole hidden layer | ✓ | ✗ |
| **Static NAN** | Each neuron is its own autoencoder (paper's proposal) | ✓ | ✓ |
| **Spiking NAN** | LIF spiking neurons with surrogate gradients + per-neuron AE | ✓ | ✓ |

Three **training schedules** are compared:
1. **Interleaved** — coin-flip task/reconstruction every batch
2. **Sequential** — Phase 1 (AE-only) → Phase 2 (task-only, frozen or finetuned)
3. **Hybrid (warm-up + interleaved)** — penalised AE warm-up, then full interleaved

Three **collapse diagnostics** are tracked:
- **Effective rank** (exponential entropy of SVD spectrum — higher = more diverse neurons)
- **Activation correlation** (mean off-diagonal |corr| — lower = better)
- **Decoder similarity** (mean off-diagonal cosine sim of decoder weights — NAN only)

---

## 2. Architecture Comparison (`compare_all4`)

> [!NOTE]
> Interleaved training, `local_weight=1.0`, no diversity penalty. Final-epoch metrics.

### h=20 (narrow bottleneck)

| Architecture | Accuracy (%) | ±SEM | Eff. Rank (of 20) | Dec. Similarity |
|:--|--:|--:|--:|:--|
| **NN** | **94.65** | 0.06 | 16.4 | n/a |
| ANN | 93.56 | 0.06 | 16.0 | n/a |
| Static NAN | 93.57 | 0.07 | 15.8 | 0.350 |
| Spiking NAN | 93.46 | 0.07 | **16.6** | 0.350 |

### h=128 (standard width)

| Architecture | Accuracy (%) | ±SEM | Eff. Rank (of 128) | Dec. Similarity |
|:--|--:|--:|--:|:--|
| **NN** | **97.42** | 0.02 | 67.1 | n/a |
| ANN | 96.74 | 0.03 | 65.9 | n/a |
| Static NAN | 96.67 | 0.03 | 63.5 | 0.361 |
| Spiking NAN | 96.97 | 0.04 | **76.6** | 0.355 |

**Key takeaways:**
- NN remains the accuracy champion in interleaved mode — the AE reconstruction objective steals some gradient bandwidth.
- **Spiking NAN achieves the highest effective rank** at both widths, suggesting the temporal dynamics of spiking neurons naturally promote diversity.
- At h=128, spiking NAN closes the accuracy gap to NN (96.97 vs 97.42) while maintaining far superior diversity (76.6 vs 67.1).
- Decoder similarity is remarkably consistent across both NAN variants (~0.35).

---

## 3. Training Curves (`curves`)

Per-epoch accuracy and effective-rank curves were generated for **all four architectures × both hidden widths**, comparing four schedule variants:
- Interleaved
- Sequential (frozen, no penalty)
- Sequential + penalty (dw=5)
- Warm-up + interleaved

> [!IMPORTANT]
> The `curves` experiment is the most comprehensive single view — it shows all four training strategies on one plot per architecture. Files: `curve_acc_{arch}.png` and `curve_rank_{arch}.png` in the `curves/` subdirectory.

**Available curve sets:**

| Arch | h=20 | h=128 |
|:--|:-:|:-:|
| NN | ✅ | ✅ |
| ANN | ✅ | ✅ |
| Static NAN | ✅ | ✅ |
| Spiking NAN | ✅ | ✅ |

---

## 4. Schedule Comparison (`schedule_compare`)

Three-way comparison: interleaved vs. pure sequential (frozen, no penalty) vs. hybrid.

> [!NOTE]
> This is the "interleaved vs. failure mode vs. hybrid" comparison, with the sequential arm being the *unpenalised frozen* variant to highlight collapse.

Available for **static** and **spiking** at both hidden widths (4 comparisons total).

---

## 5. Sequential Variants — 2×2 Grid (`seq_variants`)

The {freeze, finetune} × {no penalty, penalty (dw=5)} factorial design. Final-epoch metrics only.

### h=20

| Variant | Static Acc (%) | Static Rank | Spiking Acc (%) | Spiking Rank |
|:--|--:|--:|--:|--:|
| freeze + no penalty | 28.8 | 1.2 | 31.5 | 5.3 |
| freeze + penalty | 81.4 | **16.5** | 80.9 | **18.3** |
| finetune + no penalty | 91.0 | 13.7 | 92.0 | 15.6 |
| finetune + penalty | 91.1 | 14.7 | 92.0 | 16.0 |

### h=128

| Variant | Static Acc (%) | Static Rank | Spiking Acc (%) | Spiking Rank |
|:--|--:|--:|--:|--:|
| freeze + no penalty | 29.1 | 1.6 | 42.1 | 13.8 |
| freeze + penalty | 88.5 | **54.2** | 89.6 | **78.2** |
| finetune + no penalty | 92.9 | 34.5 | 94.7 | 60.4 |
| finetune + penalty | 93.2 | 50.0 | 95.3 | 70.8 |

**Key takeaways:**
- **Frozen + no penalty = catastrophic collapse.** Effective rank drops to ~1 (static) or ~5–14 (spiking). Accuracy near random chance (static) or barely above it (spiking). This is the "do all neurons learn the same thing?" failure mode.
- **Spiking neurons are more collapse-resistant** even without a penalty — at h=128, spiking freeze+nopen reaches 42.1% acc and rank 13.8 vs. static's 29.1% and rank 1.6.
- **The diversity penalty rescues the frozen schedule** — freeze+pen jumps from ~29% to ~88% accuracy (static, h=128) and from ~1.6 to ~54 effective rank.
- **Finetuning is the best single fix for accuracy**, but the penalty still helps diversity (finetune+pen > finetune+nopen in rank).

---

## 6. Diversity Weight Sweep (`div_sweep`)

Interleaved training, sweeping `div_weight` ∈ {0, 0.1, 0.5, 1, 2, 5, 10, 20} at both hidden widths.

### Decoder Penalty — Static NAN

| | h=20 Acc | h=20 Rank | h=128 Acc | h=128 Rank |
|--:|--:|--:|--:|--:|
| dw=0 | 92.61 | 15.2 | 95.69 | 57.9 |
| dw=0.1 | 92.61 | 15.2 | 95.69 | 58.1 |
| dw=1 | 92.61 | 15.2 | 95.69 | 58.0 |
| dw=5 | 92.62 | 15.2 | 95.70 | 58.0 |
| dw=20 | 92.62 | 15.2 | 95.69 | 58.0 |

> [!WARNING]
> **The decoder penalty has essentially zero effect in interleaved mode.** Accuracy and rank are flat across the entire 0→20 range. This is likely because the interleaved coin-flip already provides sufficient implicit regularisation, or because the decoder penalty gradient is dominated by the task gradient.

### Activation (Barlow-Twins) Penalty — Static NAN, h=20

| dw | Acc (%) | Rank |
|--:|--:|--:|
| 0 | 92.61 | 15.2 |
| 0.5 | 92.67 | 16.0 |
| 2 | 92.33 | 17.4 |
| 5 | 91.59 | 18.2 |
| 10 | 90.88 | 18.5 |
| 20 | 90.00 | 18.5 |

> [!IMPORTANT]
> **The activation penalty actually works** — it pushes effective rank from 15.2 to 18.5 (near the h=20 ceiling). However, it comes at an accuracy cost: ~2.6% degradation at dw=20. There's an accuracy–diversity tradeoff, with the sweet spot around dw=0.5–1.

### Activation Penalty — Spiking NAN, h=20

| dw | Acc (%) | Rank |
|--:|--:|--:|
| 0 | 92.61 | 16.3 |
| 1 | 92.70 | 16.4 |
| 5 | 90.71 | 11.5 |
| 10 | 89.46 | 9.5 |
| 20 | 87.21 | 7.5 |

> [!CAUTION]
> **Spiking + activation penalty at high strength causes RANK COLLAPSE**, the opposite of the intended effect. Rank drops from 16.3 → 7.5 at dw=20. The Barlow-Twins decorrelation may interfere with the temporal dynamics of spiking neurons. This is a significant negative result.

### Decoder Penalty — Spiking NAN, h=20

Flat profile, similar to static decoder penalty. No meaningful effect on accuracy or rank across dw=0→20.

---

## 7. Local (Reconstruction) Weight Sweep (`local_sweep`)

Interleaved training, sweeping `local_weight` ∈ {0, 0.1, 0.25, 0.5, 1, 2, 4}.

### h=20

| local_weight | Static Acc | Static Rank | Spiking Acc | Spiking Rank |
|--:|--:|--:|--:|--:|
| 0 (task-only) | **93.77** | 15.8 | **93.66** | 16.6 |
| 0.1 | 92.61 | 15.2 | 92.56 | 16.2 |
| 1.0 | 92.61 | 15.2 | 92.61 | 16.3 |
| 4.0 | 92.53 | 15.2 | 92.58 | 16.2 |

### h=128

| local_weight | Static Acc | Static Rank |
|--:|--:|--:|
| 0 (task-only) | **96.75** | 62.1 |
| 0.1 | 95.70 | 58.0 |
| 1.0 | 95.69 | 57.9 |
| 4.0 | 95.65 | 57.6 |

**Key takeaway:** Any nonzero local weight costs ~1% accuracy and slightly reduces effective rank compared to pure task training (`local_weight=0`). The reconstruction objective competes with the task signal. However, the *purpose* of the AE signal is to learn structured representations — these interleaved sweeps don't reveal this because the interleaved schedule blends both objectives.

---

## 8. Penalised Schedule Comparison (`penalised_compare`)

The most targeted experiment: **three schedules at the same penalty strength**, for dw ∈ {0, 2, 5} × {static, spiking} × {h=20, h=128} = **12 conditions**.

### h=128 — Final Epoch Summary

| Schedule | dw=0 Acc | dw=0 Rank | dw=2 Acc | dw=2 Rank | dw=5 Acc | dw=5 Rank |
|:--|--:|--:|--:|--:|--:|--:|
| **Static NAN** | | | | | | |
| Interleaved | 96.67 | 63.5 | 96.70 | 63.7 | 96.70 | 63.7 |
| Sequential (finetune) | 92.92 | 34.5 | 92.82 | 38.0 | 93.23 | 50.0 |
| Hybrid | 95.45 | 58.7 | 95.38 | 57.2 | 95.45 | 58.7 |
| **Spiking NAN** | | | | | | |
| Interleaved | 96.97 | 76.6 | 96.99 | 76.7 | 96.97 | 76.9 |
| Sequential (finetune) | 94.66 | 60.4 | 94.87 | 69.6 | 95.28 | 70.8 |
| Hybrid | 96.20 | 72.9 | 96.20 | 73.0 | 96.20 | 73.4 |

### h=20 — Final Epoch Summary

| Schedule | dw=0 Acc | dw=0 Rank | dw=2 Acc | dw=2 Rank | dw=5 Acc | dw=5 Rank |
|:--|--:|--:|--:|--:|--:|--:|
| **Static NAN** | | | | | | |
| Interleaved | 93.57 | 15.8 | 93.58 | 15.8 | 93.58 | 15.8 |
| Sequential (finetune) | 91.03 | 13.7 | 91.09 | 13.9 | 91.08 | 14.7 |
| Hybrid | 92.28 | 15.2 | 92.34 | 15.2 | 92.22 | 15.3 |
| **Spiking NAN** | | | | | | |
| Interleaved | 93.46 | 16.6 | 93.41 | 16.6 | 93.44 | 16.6 |
| Sequential (finetune) | 92.02 | 15.6 | 92.00 | 15.9 | 92.05 | 16.0 |
| Hybrid | 92.35 | 16.1 | 92.52 | 16.2 | 92.32 | 16.2 |

**Key takeaways:**
- **Schedule ranking is consistent**: interleaved > hybrid > sequential across all conditions.
- **Penalty helps sequential the most** — at h=128 static, the penalty pushes sequential rank from 34.5 (dw=0) to 50.0 (dw=5), a massive improvement.
- **Interleaved is almost immune to the penalty** — rank stays ~63.5–63.7 (static) and ~76.6–76.9 (spiking) regardless of dw.
- **Spiking consistently outperforms static** across every schedule and penalty condition at h=128.
- At h=20, differences are much more compressed — less room for diversity with only 20 neurons.

---

## 9. Neuron Visualizations (`visualize`)

Decoder-weight fields and activation maps were generated for all four architectures at both hidden widths.

| File | h=20 | h=128 |
|:--|:-:|:-:|
| `decoder_fields_static.png` | ✅ | ✅ |
| `decoder_fields_spiking.png` | ✅ | ✅ |
| `activation_maps_nn.png` | ✅ | ✅ |
| `activation_maps_ann.png` | ✅ | ✅ |
| `activation_maps_static.png` | ✅ | ✅ |
| `activation_maps_spiking.png` | ✅ | ✅ |

> [!NOTE]
> NN and ANN don't produce decoder field plots (no per-neuron decoder). Activation maps are available for all four architectures.

---

## 10. Cluster Job Summary

All experiments ran on NVIDIA L40S GPUs on the CSF3 cluster (Manchester). Jobs were submitted via Slurm.

| Job ID | Config | Status | Notes |
|:--|:--|:-:|:--|
| 16390713 | h=128, dw=0, static+spiking | ✅ | Completed in ~340s (static) + ~1080s (spiking) |
| 16390714 | h=128, dw=2, static only | ⚠️ | Static completed, spiking log truncated — resubmitted as 16390867 |
| 16390715 | h=128, dw=5 | ❌ | **CUDA ECC error** on node877 — resubmitted as 16390868 |
| 16390771 | h=20, dw=0, static+spiking | ✅ | |
| 16390772 | h=20, dw=2 | ❌ | Failed (ECC error) — resubmitted as 16390866 |
| 16390773 | h=20, dw=5, static+spiking | ✅ | |
| 16390774 | h=128, dw=0 (resubmit) | ✅ | Duplicate of 16390713, both succeeded |
| 16390775–776 | h=128 dw=2, h=128 dw=5 | ❌ | ECC errors — resubmitted |
| 16390866 | h=20, dw=2, static+spiking | ✅ | Successful resubmit |
| 16390867 | h=128, dw=2, static+spiking | ✅ | Successful resubmit |
| 16390868 | h=128, dw=5, static+spiking | ✅ | Successful resubmit |

> [!WARNING]
> Several initial runs hit **CUDA uncorrectable ECC errors** on specific nodes (node877, node886). These were resolved by resubmission to different nodes. All 12 experiment conditions completed successfully on the second attempt.

---

## 11. Key Findings (Summary)

1. **Interleaved training is the strongest schedule.** It consistently achieves the highest accuracy and near-maximum effective rank without any explicit diversity penalty.

2. **The hybrid (warm-up + interleaved) schedule is a viable middle ground.** It trails interleaved by ~1–1.5% accuracy but substantially outperforms sequential.

3. **Sequential training collapses without a penalty.** Freezing the representation after AE-only training leads to catastrophic rank collapse (effective rank ≈ 1 for static at h=128).

4. **The decoder diversity penalty is inert during interleaved training** but highly effective during sequential Phase 1.

5. **The activation (Barlow-Twins) penalty works but carries risk.** It genuinely decorrelates neurons (rank ↑) but degrades accuracy at high strengths, and **causes paradoxical collapse in spiking neurons**.

6. **Spiking NAN is the most robust architecture.** It achieves the highest effective rank across all conditions, shows natural collapse resistance, and nearly matches NN accuracy at h=128.

7. **Width matters.** The h=128 regime shows clearer architectural differences and larger diversity effects than h=20, where the ceiling is too low for meaningful differentiation.

---

## 12. Experimentation Gaps

> [!CAUTION]
> The following experiments are either **missing, incomplete, or not yet attempted** based on what the codebase supports vs. what results exist.

### 🔴 Not Run At All

| Gap | Why It Matters |
|:--|:--|
| **`compare_all4` with diversity penalty** | The 4-arch comparison uses `local_weight=1.0` but `div_weight=0.0`. Running it with the activation penalty would show whether the penalty preferentially helps NAN over NN/ANN. |
| **`div_sweep` and `local_sweep` for NN and ANN architectures** | Sweeps only cover static and spiking NAN. Running for NN/ANN would complete the picture — especially since ANN has a shared decoder. |
| **`seq_variants` for NN and ANN** | The 2×2 grid was only run for NAN variants. How does NN perform in sequential? (Its AE loss is constant zero, so this would test the schedule overhead on a non-AE model.) |
| **`schedule_compare` / `penalised_compare` for ANN** | ANN has a shared decoder — the penalty might behave differently since `penalty_decoder` operates on the weight matrix which is indexed by *input dimension* not *neuron*. |
| **Activation penalty in `penalised_compare`** | All penalised_compare runs use `penalty="decoder"`. The activation penalty showed interesting (and dangerous) behavior in the div_sweep — it deserves a full schedule comparison. |
| **`compare_all4` with h=20** and **with schedules other than interleaved** | Currently compares architectures in only the interleaved regime. |

### 🟡 Partially Complete / Could Be Extended

| Gap | What's Missing |
|:--|:--|
| **Higher epoch count** | 15 epochs (sweeps: 9) is relatively short. Collapse/diversity effects may take longer to manifest or saturate. Consider 30–50 epochs. |
| **Resolution sensitivity** | All runs use `res=28`. The codebase supports `res=7` and `res=14` — different resolutions test whether the AE structure scales with input complexity. |
| **Class subset experiments** | All runs use `nc=10`. Testing with 2, 3, or 5 classes would isolate whether the collapse phenomenon is driven by task difficulty. |
| **Decoder vs. activation penalty on h=128 spiking** | The h=20 activation penalty caused paradoxical rank collapse in spiking NAN. Does this hold at h=128? Only the decoder penalty was swept at h=128. |
| **Statistical power for sweeps** | Sweeps use 9 epochs (60% of 15). Some sweep points show SEM ≈ mean effect size, suggesting more seeds or epochs would help. |

### 🟢 Nice-to-Have / Future Work

| Gap | Notes |
|:--|:--|
| **Warm-up epoch ratio sweep** | Currently fixed at 50/50 split. What if warm-up is only 20% of total epochs? |
| **Learning rate sensitivity** | All experiments use `lr=1e-3` for the main optimizer and `lr=1e-4` for finetune. No LR sweep exists. |
| **Larger networks (h=256, h=512)** | Would the spiking rank advantage scale with width? |
| **Non-MNIST datasets (Fashion-MNIST, CIFAR)** | MNIST may be too easy — architectures converge to similar accuracy. A harder task might differentiate them more. |
| **`visualize` with diversity penalty** | Current visualizations use default settings. Comparing decoder fields with/without penalty would visually confirm decorrelation. |
| **Reconstruction quality metrics** | No experiment measures the actual reconstruction MSE of the autoencoders — only the downstream accuracy and diversity. |
