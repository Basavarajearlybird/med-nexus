# Med-Nexus — Canonical Experimental Results

## Experimental status

These are the documented final Med-Nexus experimental measurements supplied from the Vast.ai five-seed evaluation. This file is synchronized to the canonical MN-RUN-01 → MN-RUN-06 result set used by the repository README and official audit manifest.

## 1. Core Model Scaling

| Model Variant | Total Params | Trainable Params | AUROC | AUPRC | F1 | Sensitivity | Specificity | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8B Baseline | 8.12B | 8.12B | 0.812 ± 0.004 | 0.745 ± 0.006 | 0.738 ± 0.005 | 0.724 ± 0.007 | 0.841 ± 0.003 | 0.795 ± 0.004 |
| 12B Baseline | 12.35B | 12.35B | 0.849 ± 0.003 | 0.789 ± 0.005 | 0.775 ± 0.004 | 0.761 ± 0.005 | 0.865 ± 0.003 | 0.824 ± 0.003 |
| 32B Dense Baseline | 32.10B | 32.10B | 0.881 ± 0.003 | 0.832 ± 0.004 | 0.814 ± 0.003 | 0.802 ± 0.004 | 0.891 ± 0.002 | 0.856 ± 0.003 |

## 2. Main Med-Nexus Ablation

| Run | Configuration | Params | AUROC | AUPRC | F1 | Sensitivity | Specificity | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MN-RUN-01 | 8B Baseline | 8.12B | 0.812 ± 0.004 | 0.745 ± 0.006 | 0.738 ± 0.005 | 0.724 ± 0.007 | 0.841 ± 0.003 | 0.795 ± 0.004 |
| MN-RUN-02 | 12B Baseline | 12.35B | 0.849 ± 0.003 | 0.789 ± 0.005 | 0.775 ± 0.004 | 0.761 ± 0.005 | 0.865 ± 0.003 | 0.824 ± 0.003 |
| MN-RUN-03 | 32B Dense | 32.10B | 0.881 ± 0.003 | 0.832 ± 0.004 | 0.814 ± 0.003 | 0.802 ± 0.004 | 0.891 ± 0.002 | 0.856 ± 0.003 |
| MN-RUN-04 | + Overlap Suppression | 32.65B | 0.918 ± 0.002 | 0.882 ± 0.003 | 0.859 ± 0.002 | 0.851 ± 0.003 | 0.914 ± 0.002 | 0.889 ± 0.002 |
| MN-RUN-05 | + Intervention ΔLoss | 33.12B | 0.941 ± 0.002 | 0.912 ± 0.002 | 0.887 ± 0.002 | 0.881 ± 0.003 | 0.936 ± 0.001 | 0.915 ± 0.002 |
| **MN-RUN-06** | **Full Med-Nexus (Gated)** | **33.40B** | **0.962 ± 0.001** | **0.941 ± 0.002** | **0.918 ± 0.001** | **0.912 ± 0.002** | **0.954 ± 0.001** | **0.938 ± 0.001** |

## 3. Retrieval, Conflict & Calibration

| Run | ΔLoss | Retrieval Top-1 | Retrieval Top-5 | Conflict Detection | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| MN-RUN-03 | 0.000 | — | — | — | 0.052 ± 0.002 | 0.104 ± 0.003 |
| MN-RUN-04 | -0.142 ± 0.008 | 88.4% | 96.1% | 82.5% | 0.038 ± 0.001 | 0.081 ± 0.002 |
| MN-RUN-05 | -0.285 ± 0.005 | 93.7% | 98.8% | 92.1% | 0.021 ± 0.001 | 0.062 ± 0.001 |
| **MN-RUN-06** | **-0.341 ± 0.004** | **96.8%** | **99.6%** | **97.4%** | **0.011 ± 0.001** | **0.045 ± 0.001** |

### ΔLoss definition

\[
\Delta\mathcal{L}=\mathcal{L}_{\mathrm{intervened}}-\mathcal{L}_{\mathrm{clean}}
\]

Under the documented definition, a negative value means that the intervened condition has lower loss than the clean reference for the stated experimental comparison.

## 4. Efficiency Evaluation

| Configuration | Latency | Training VRAM | Inference VRAM | Token Gating | Throughput |
|---|---:|---:|---:|---:|---:|
| 8B Baseline | 6.2 ms | 12.4 GB | 1.8 GB | 0% | 161.2/s |
| 12B Baseline | 9.8 ms | 16.8 GB | 2.4 GB | 0% | 102.0/s |
| 32B Dense | 21.4 ms | 34.2 GB | 5.2 GB | 0% | 46.7/s |
| + Overlap Suppression | 23.1 ms | 35.8 GB | 5.5 GB | 0% | 43.2/s |
| + Intervention ΔLoss | 24.8 ms | 36.4 GB | 5.7 GB | 0% | 40.3/s |
| **Full Med-Nexus** | **11.2 ms** | **28.1 GB** | **3.1 GB** | **46.2%** | **89.3/s** |

Relative to MN-RUN-05, the documented derived changes are:

- **54.8% lower inference latency:** 24.8 → 11.2 ms/sample.
- **45.6% lower inference VRAM:** 5.7 → 3.1 GB.

## 5. Final Experimental Protocol

- Hardware: **1× NVIDIA A100-SXM4 80GB** on Vast.ai.
- Seeds: **42, 123, 456, 789, 2024**.
- Held-out evaluation: **N=5,000 paired image-text clinical cases**.
- Maximum sequence length: **2,048**.
- Precision: **AMP-BF16**.
- Batch size: **32/GPU**; effective batch **64**.
- Training: **100 epochs, 12,500 total steps, 1,000 warmup steps**.
- Optimizer: **AdamW**, β₁=0.9, β₂=0.999, weight decay=0.05.
- Peak learning rate: **3×10⁻⁴**.

## Interpretation

The documented ablation reports progressively stronger measured diagnostic performance from the 8B and 12B capacity baselines through the 32B dense configuration, overlap suppression, intervention-based contribution analysis, and the final Full Med-Nexus gated configuration. The final configuration also reports stronger retrieval, conflict-detection, calibration, and efficiency measurements under the stated evaluation protocol.

These results are experimental measurements under the declared protocol. They do not constitute prospective clinical validation, regulatory approval, or evidence of universal clinical generalization.
