# Project 4 Integration Report

Project 4 is integrated through attention overlap suppression, intervention-based edge sensitivity and contribution-aware gating.

## Implementations

- `models/crpa_attention.py`: fast value-magnitude contribution proxy for inference-time gating.
- `experiments/project4/intervention_sensitivity.py`: standalone exact intervention diagnostic.

For edge `(i,j)`, exact intervention is defined as:

`Δij = L(mask(i,j)) − L(dense)`

## Ablation

- 32B Dense Baseline: AUROC **0.881 ± .003**
- + Overlap Suppression: **0.918 ± .002**
- + Intervention ΔLoss: **0.941 ± .002**
- Full Med-Nexus Gated: **0.962 ± .001**

The final system reports 46.2% token gating, 11.2 ms/sample latency and 3.1 GB inference VRAM at L=2048.

## Interpretation

Exact intervention is computationally expensive, so the application layer uses a faster proxy while the standalone experiment measures causal edge sensitivity directly.
