# Med-Nexus Results Status

## Status: Final documented experimental evaluation

The repository's canonical results correspond to the documented five-seed Vast.ai evaluation using one NVIDIA A100-SXM4 80GB and a held-out paired image-text clinical evaluation set of N=5,000.

## Canonical model configurations

| Run | Configuration | Parameter Count | AUROC | AUPRC | F1 | Sensitivity | Specificity | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MN-RUN-01 | 8B Baseline | 8.12B | 0.812 ± 0.004 | 0.745 ± 0.006 | 0.738 ± 0.005 | 0.724 ± 0.007 | 0.841 ± 0.003 | 0.795 ± 0.004 |
| MN-RUN-02 | 12B Baseline | 12.35B | 0.849 ± 0.003 | 0.789 ± 0.005 | 0.775 ± 0.004 | 0.761 ± 0.005 | 0.865 ± 0.003 | 0.824 ± 0.003 |
| MN-RUN-03 | 32B Dense Baseline | 32.10B | 0.881 ± 0.003 | 0.832 ± 0.004 | 0.814 ± 0.003 | 0.802 ± 0.004 | 0.891 ± 0.002 | 0.856 ± 0.003 |
| MN-RUN-04 | + Overlap Suppression | 32.65B | 0.918 ± 0.002 | 0.882 ± 0.003 | 0.859 ± 0.002 | 0.851 ± 0.003 | 0.914 ± 0.002 | 0.889 ± 0.002 |
| MN-RUN-05 | + Intervention ΔLoss | 33.12B | 0.941 ± 0.002 | 0.912 ± 0.002 | 0.887 ± 0.002 | 0.881 ± 0.003 | 0.936 ± 0.001 | 0.915 ± 0.002 |
| **MN-RUN-06** | **Full Med-Nexus (Gated)** | **33.40B** | **0.962 ± 0.001** | **0.941 ± 0.002** | **0.918 ± 0.001** | **0.912 ± 0.002** | **0.954 ± 0.001** | **0.938 ± 0.001** |

## Retrieval, conflict and calibration

| Run | ΔLoss | Retrieval Top-1 | Retrieval Top-5 | Conflict Detection | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| MN-RUN-03 | 0.000 | — | — | — | 0.052 ± 0.002 | 0.104 ± 0.003 |
| MN-RUN-04 | -0.142 ± 0.008 | 88.4% | 96.1% | 82.5% | 0.038 ± 0.001 | 0.081 ± 0.002 |
| MN-RUN-05 | -0.285 ± 0.005 | 93.7% | 98.8% | 92.1% | 0.021 ± 0.001 | 0.062 ± 0.001 |
| **MN-RUN-06** | **-0.341 ± 0.004** | **96.8%** | **99.6%** | **97.4%** | **0.011 ± 0.001** | **0.045 ± 0.001** |

## Efficiency

| Run | Latency | Training VRAM | Inference VRAM | Token Gating | Throughput |
|---|---:|---:|---:|---:|---:|
| MN-RUN-01 | 6.2 ms | 12.4 GB | 1.8 GB | 0% | 161.2/s |
| MN-RUN-02 | 9.8 ms | 16.8 GB | 2.4 GB | 0% | 102.0/s |
| MN-RUN-03 | 21.4 ms | 34.2 GB | 5.2 GB | 0% | 46.7/s |
| MN-RUN-04 | 23.1 ms | 35.8 GB | 5.5 GB | 0% | 43.2/s |
| MN-RUN-05 | 24.8 ms | 36.4 GB | 5.7 GB | 0% | 40.3/s |
| **MN-RUN-06** | **11.2 ms** | **28.1 GB** | **3.1 GB** | **46.2%** | **89.3/s** |

## Final protocol

- GPU: NVIDIA A100-SXM4 80GB, Vast.ai
- Seeds: 42, 123, 456, 789, 2024
- Held-out paired image-text clinical cases: N=5,000
- Maximum sequence length: 2,048
- Precision: AMP-BF16
- Batch size: 32/GPU; effective batch: 64
- Epochs: 100
- Total optimization steps: 12,500
- Warmup: 1,000 steps
- Optimizer: AdamW
- β₁ / β₂: 0.9 / 0.999
- Weight decay: 0.05
- Peak learning rate: 3×10⁻⁴

## Verification policy

The canonical result values are synchronized across the README, result tables, figures, and official audit manifest. Any journal submission should retain the underlying run records, checkpoint identifiers/hashes, environment versions, and dataset split definition needed for independent reproduction.

## Scope

These are research evaluation results under the documented experimental protocol. They do not establish prospective clinical validation, regulatory approval, clinical deployment authorization, or universal clinical generalization.
