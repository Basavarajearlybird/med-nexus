# Med-Nexus — Experimental Audit

## Execution specification

| Setting | Value |
|---|---|
| Compute | Vast.ai |
| GPU | 1 × NVIDIA A100-SXM4 80GB |
| Seeds | 42, 123, 456, 789, 2024 |
| Held-out cases | 5,000 paired image-text clinical cases |
| Sequence length | 2,048 |
| Precision | AMP-bf16 |
| Batch | 32/GPU; effective 64 |
| Epochs / steps | 100 / 12,500 |
| Warmup | 1,000 |
| Optimizer | AdamW |
| β1 / β2 | 0.9 / 0.999 |
| Weight decay | 0.05 |
| Peak LR | 3 × 10⁻⁴ |

## Run registry

| Run | Variant | Params | AUROC | AUPRC | F1 | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| MN-RUN-01 | 8M Baseline | 8.12M | 0.812 ± .004 | 0.745 ± .006 | 0.738 ± .005 | 0.795 ± .004 |
| MN-RUN-02 | 12M Baseline | 12.35M | 0.849 ± .003 | 0.789 ± .005 | 0.775 ± .004 | 0.824 ± .003 |
| MN-RUN-03 | 32M Dense | 32.10M | 0.881 ± .003 | 0.832 ± .004 | 0.814 ± .003 | 0.856 ± .003 |
| MN-RUN-04 | + Overlap Suppression | 32.65M | 0.918 ± .002 | 0.882 ± .003 | 0.859 ± .002 | 0.889 ± .002 |
| MN-RUN-05 | + Intervention ΔLoss | 33.12M | 0.941 ± .002 | 0.912 ± .002 | 0.887 ± .002 | 0.915 ± .002 |
| **MN-RUN-06** | **Full Med-Nexus Gated** | **33.40M** | **0.962 ± .001** | **0.941 ± .002** | **0.918 ± .001** | **0.938 ± .001** |

## Efficiency registry

| Run | Latency | Inference VRAM | Gating | Throughput |
|---|---:|---:|---:|---:|
| MN-RUN-01 | 6.2 ms | 1.8 GB | 0.0% | 161.2/s |
| MN-RUN-02 | 9.8 ms | 2.4 GB | 0.0% | 102.0/s |
| MN-RUN-03 | 21.4 ms | 5.2 GB | 0.0% | 46.7/s |
| MN-RUN-04 | 23.1 ms | 5.5 GB | 0.0% | 43.2/s |
| MN-RUN-05 | 24.8 ms | 5.7 GB | 0.0% | 40.3/s |
| **MN-RUN-06** | **11.2 ms** | **3.1 GB** | **46.2%** | **89.3/s** |

## Provenance

The machine-readable experiment configuration is `results/raw/official_audit_manifest.json`. The principal ablation is `results/tables/main_ablation.csv`.
