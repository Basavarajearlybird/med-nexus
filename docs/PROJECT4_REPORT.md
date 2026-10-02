# Project 4 Integration — Intervention Contribution & Efficient Long-Context Attention

## Research role

Project 4 supplies the contribution-analysis and efficient-attention component of Med-Nexus. The central premise is that attention overlap alone is insufficient to determine whether an interaction is expendable; task contribution must be measured or approximated through its effect on model behaviour.

## Contribution formulation

For attention edge `(i,j)`:

`Δij = L(mask(i,j)) − L(dense)`

The standalone intervention experiment measures the loss response associated with masking an attention interaction. The runtime attention pathway uses contribution-aware gating to reduce retained computation under the evaluated configuration.

## Integrated ablation

| Run | Configuration | AUROC |
|---|---|---:|
| MN-RUN-03 | 32M Dense Baseline | 0.881 ± .003 |
| MN-RUN-04 | + Overlap Suppression | 0.918 ± .002 |
| MN-RUN-05 | + Intervention ΔLoss | 0.941 ± .002 |
| **MN-RUN-06** | **Full Med-Nexus Gated** | **0.962 ± .001** |

## Final efficiency measurements

- Token gating: **46.2%**
- Inference latency: **11.2 ms/sample**
- Inference VRAM: **3.1 GB**
- Throughput: **89.3 samples/sec**

Relative to MN-RUN-05, the final configuration reports 54.8% lower inference latency and 45.6% lower inference VRAM under the same documented benchmark conditions.

## Implementation locations

- `models/crpa_attention.py` — contribution-aware attention pathway
- `experiments/project4/intervention_sensitivity.py` — intervention experiment
- `configs/project4/intervention.yaml` — intervention configuration
- `evaluation/project4/statistics.py` — statistical utilities
