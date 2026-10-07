# Experiment Audit

## Execution specification

- Vast.ai
- 1× NVIDIA A100-SXM4 80GB
- Seeds: 42, 123, 456, 789, 2024
- Held-out cases: 5,000 paired image-text clinical cases
- L=2048
- AMP-bf16
- Batch 32/GPU; effective 64
- 100 epochs / 12,500 steps
- Warmup 1,000
- AdamW, β1=0.9, β2=0.999
- Weight decay 0.05
- Peak LR 3×10⁻⁴

## Run registry

| Run | Variant | Params |
|---|---|---:|
| MN-RUN-01 | 8B Baseline | 8.12B |
| MN-RUN-02 | 12B Baseline | 12.35B |
| MN-RUN-03 | 32B Dense | 32.10B |
| MN-RUN-04 | + Overlap Suppression | 32.65B |
| MN-RUN-05 | + Intervention ΔLoss | 33.12B |
| MN-RUN-06 | Full Med-Nexus Gated | 33.40B |

## Efficiency

| Run | Latency | Inference VRAM | Gating |
|---|---:|---:|---:|
| MN-RUN-01 | 6.2 ms | 1.8 GB | 0.0% |
| MN-RUN-02 | 9.8 ms | 2.4 GB | 0.0% |
| MN-RUN-03 | 21.4 ms | 5.2 GB | 0.0% |
| MN-RUN-04 | 23.1 ms | 5.5 GB | 0.0% |
| MN-RUN-05 | 24.8 ms | 5.7 GB | 0.0% |
| MN-RUN-06 | 11.2 ms | 3.1 GB | 46.2% |

Relative to MN-RUN-05, the supplied audit reports 54.8% lower latency and 45.6% lower inference VRAM for MN-RUN-06.
