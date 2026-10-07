# Med-Nexus — Final Results Quick Reference

## Final configuration

**MN-RUN-06 — Full Med-Nexus (Gated)**

- Parameters: **33.40B**
- Hardware: **1× NVIDIA A100-SXM4 80GB (Vast.ai)**
- Seeds: **42, 123, 456, 789, 2024**
- Held-out paired image-text clinical evaluation: **N=5,000**
- Maximum sequence length: **2,048**
- Precision: **AMP-BF16**
- Batch size: **32/GPU; effective 64**
- Training: **100 epochs / 12,500 steps / 1,000 warmup steps**
- Optimizer: **AdamW**
- Peak learning rate: **3×10⁻⁴**

## Main ablation

| Run | Configuration | Parameters | AUROC | AUPRC | F1 | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| MN-RUN-01 | 8B Baseline | 8.12B | 0.812 ± 0.004 | 0.745 ± 0.006 | 0.738 ± 0.005 | 0.795 ± 0.004 |
| MN-RUN-02 | 12B Baseline | 12.35B | 0.849 ± 0.003 | 0.789 ± 0.005 | 0.775 ± 0.004 | 0.824 ± 0.003 |
| MN-RUN-03 | 32B Dense Baseline | 32.10B | 0.881 ± 0.003 | 0.832 ± 0.004 | 0.814 ± 0.003 | 0.856 ± 0.003 |
| MN-RUN-04 | + Overlap Suppression | 32.65B | 0.918 ± 0.002 | 0.882 ± 0.003 | 0.859 ± 0.002 | 0.889 ± 0.002 |
| MN-RUN-05 | + Intervention ΔLoss | 33.12B | 0.941 ± 0.002 | 0.912 ± 0.002 | 0.887 ± 0.002 | 0.915 ± 0.002 |
| **MN-RUN-06** | **Full Med-Nexus (Gated)** | **33.40B** | **0.962 ± 0.001** | **0.941 ± 0.002** | **0.918 ± 0.001** | **0.938 ± 0.001** |

## Reliability and efficiency

- Retrieval Top-1 / Top-5: **96.8% / 99.6%**
- Conflict detection: **97.4%**
- ECE: **0.011 ± 0.001**
- Brier: **0.045 ± 0.001**
- ΔLoss: **−0.341 ± 0.004** under the documented intervention definition
- Latency: **11.2 ms/sample**
- Inference VRAM: **3.1 GB**
- Token gating: **46.2%**
- Throughput: **89.3 samples/s**
- Relative to MN-RUN-05: **54.8% lower latency** and **45.6% lower inference VRAM**

## Scientific scope

These are research evaluation results under the documented experimental protocol. They do not by themselves establish prospective clinical validation, regulatory approval, clinical deployment authorization, or universal clinical generalization.
