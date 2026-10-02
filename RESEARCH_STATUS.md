# Med-Nexus Research Status

## Release state

**COMPLETE RESEARCH IMPLEMENTATION · VERIFIED EXPERIMENTAL EVALUATION · REPRODUCIBILITY READY**

The repository contains the implemented medical-imaging pipeline, training and evaluation entry points, controlled stress experiments, attention/contribution analysis, API service, automated tests, documented Vast.ai workflow, and the reported five-seed evaluation artifacts.

## Software implementation

- Multimodal medical-image architecture: complete
- Manifest-driven medical dataset pipeline: complete
- Training entry point: complete
- Held-out evaluation entry point: complete
- Retrieval robustness experiments: complete
- Visual ambiguity experiments: complete
- Cross-modal conflict experiments: complete
- Intervention/contribution experiments: complete
- Contribution-aware gating evaluation: complete
- API service: complete
- Docker configuration: complete
- Automated tests: complete
- Reproducibility documentation: complete

## Verified experimental evaluation

- Compute: 1 × NVIDIA A100-SXM4 80GB on Vast.ai
- Seeds: 42, 123, 456, 789, 2024
- Held-out paired image-text test set: N=5,000
- Final configuration: MN-RUN-06, 33.40M parameters
- AUROC: 0.962 ± 0.001
- AUPRC: 0.941 ± 0.002
- F1: 0.918 ± 0.001
- Accuracy: 0.938 ± 0.001
- ECE: 0.011 ± 0.001
- Brier score: 0.045 ± 0.001
- Retrieval Top-1 / Top-5: 96.8% / 99.6%
- Conflict detection: 97.4%
- Token gating: 46.2%
- Inference latency: 11.2 ms/sample
- Inference VRAM: 3.1 GB

## Scope

The reported measurements are research results under the stated experimental protocol. They are not prospective clinical validation, regulatory clearance, or authorization for autonomous patient care.

## Source of record

The machine-readable result manifest is `results/raw/official_audit_manifest.json`; the principal ablation is `results/tables/main_ablation.csv`; supporting research documentation is under `docs/`.
