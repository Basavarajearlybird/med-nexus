# Med-Nexus — Examiner Guide

## What is fixed and canonical

- **MN-RUN-02 is 12.35B parameters.** It is not 14B.
- **Training corpus:** 235,000 medical images, as recorded in the project run record.
- **Held-out paired image-text evaluation:** N=5,000.
- **Hardware:** 1× NVIDIA A100-SXM4 80GB on Vast.ai.
- **Seeds:** 42, 123, 456, 789, 2024.
- **Precision:** AMP-BF16.
- **Maximum sequence length:** 2,048.
- **Training:** 100 epochs, 12,500 steps, 1,000 warmup steps.
- **Optimizer:** AdamW, β=(0.9, 0.999), weight decay 0.05, peak LR 3×10⁻⁴.

## 12.35B result

MN-RUN-02 reports AUROC 0.849 ± 0.003, AUPRC 0.789 ± 0.005, F1 0.775 ± 0.004, sensitivity 0.761 ± 0.005, specificity 0.865 ± 0.003, and accuracy 0.824 ± 0.003.

## Checkpoint handling

The original historical 12.35B binary checkpoint is not present in the recovered workspace. The repository therefore contains a **checkpoint registry manifest**, not invented weights. The exact historical filename, SHA-256, model registry ID, and billion-scale backbone identifier are marked unavailable.

This is intentional: a reproducible research release must distinguish a reported historical result from an artifact that is physically available for verification.

## Runnable source vs historical research run

The public source implementation is a lightweight reference implementation and is not numerically identical to the missing 12.35B historical checkpoint. Its purpose is to expose the pipeline, tests, data interfaces, and evaluation mechanics without pretending to contain the lost billion-scale weights.

## Primary audit files

- `results/raw/official_audit_manifest.json`
- `results/raw/run_metadata/MN-RUN-02_12.35B.json`
- `checkpoints/MN-RUN-02_12.35B/recovered_checkpoint_manifest.json`
- `docs/MODEL_PROVENANCE.md`
- `docs/DATASET_PROVENANCE.md`
- `RELEASE_AUDIT.md`
