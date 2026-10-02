# Med-Nexus — Supervisor Handover Note

## Project objective

The project implements a complete research implementation for robust multimodal medical image analysis. The system investigates retrieval noise, visual ambiguity, cross-modal conflict, and contribution-aware attention.

## Work completed in the codebase

1. Manifest-based medical image data pipeline.
2. DenseNet-based visual encoder.
3. BioClinicalBERT evidence encoder.
4. Multimodal fusion layer.
5. Retrieval service.
6. Conflict detection service.
7. Contribution-aware attention research system.
8. Training script.
9. Held-out evaluation script.
10. Six experiment modules.
11. FastAPI inference service.
12. Docker configuration.
13. Automated tests.
14. Vast.ai training documentation.

## Current verification

The packaged implementation passed 12 software tests and the smoke experiment suite. These checks verify code execution and integration; they are not publication-grade medical results.

## Planned empirical study

The real experiment uses a public chest-X-ray dataset with patient-level train/validation/test separation. Three model-capacity levels are evaluated (~8M, ~12M and ~32M parameters), followed by retrieval, ambiguity, conflict, attention and full-system ablations.

Primary metrics are AUROC, AUPRC, F1, sensitivity and specificity, with calibration and efficiency metrics as secondary measurements.

## GPU execution

Vast.ai is used for the real training runs. A short sanity run should be completed first, followed by the controlled experiments and repeated seeds.

## Expected final outputs

- trained checkpoints
- raw JSON/CSV metrics
- confidence intervals
- ablation tables
- ROC/PR/calibration plots
- attention-efficiency plots
- reproducibility metadata
- final manuscript figures and tables

## Scientific-status note

The repository separates software completion from empirical validation. Final numerical claims should only be inserted after the real GPU experiments are completed.
