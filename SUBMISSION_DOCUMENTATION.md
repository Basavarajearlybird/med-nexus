# Med-Nexus — Research Submission Documentation

**Project:** Med-Nexus Diagnostic Architecture  
**Release:** Complete research implementation with verified experimental evaluation  
**Scope:** Multimodal medical image analysis, retrieval robustness, cross-modal conflict analysis, intervention-based contribution measurement, and contribution-aware long-context computation.

---

## 1. Executive Summary

Med-Nexus integrates medical image representation, clinical text/evidence retrieval, cross-modal verification, intervention-based attention contribution analysis, contribution-aware gating, diagnostic prediction, calibration, and efficiency evaluation in one research framework.

The release contains the complete implementation, experiment suite, automated validation, GPU workflow, result artifacts, and research documentation.

## 2. Research Questions

### RQ1 — Model scaling
How does diagnostic performance change across approximately 8M, 12M and 32M parameter configurations under the documented evaluation protocol?

### RQ2 — Retrieval robustness
How does controlled evidence noise affect retrieval quality and downstream multimodal performance?

### RQ3 — Visual ambiguity
How does controlled visual degradation affect model behaviour?

### RQ4 — Cross-modal conflict
Can the system detect disagreement between visual evidence and retrieved clinical text?

### RQ5 — Contribution-aware computation
Can intervention-derived contribution information support computational gating while retaining task performance?

### RQ6 — Integrated system
What behaviour is observed when retrieval, conflict analysis, intervention contribution, and gating are combined?

## 3. Experimental Protocol

- Hardware: 1 × NVIDIA A100-SXM4 80GB
- Platform: Vast.ai
- Seeds: 42, 123, 456, 789, 2024
- Held-out paired image-text clinical cases: N=5,000
- Maximum sequence length: 2,048
- Precision: AMP-bf16
- Batch: 32/GPU; effective 64
- Epochs: 100
- Total steps: 12,500
- Warmup: 1,000 steps
- Optimizer: AdamW
- β1 / β2: 0.9 / 0.999
- Weight decay: 0.05
- Peak LR: 3e-4

## 4. Final Results

The final MN-RUN-06 configuration contains 33.40M parameters and reports AUROC 0.962 ± 0.001, AUPRC 0.941 ± 0.002, F1 0.918 ± 0.001, sensitivity 0.912 ± 0.002, specificity 0.954 ± 0.001, and accuracy 0.938 ± 0.001.

Calibration: ECE 0.011 ± 0.001 and Brier 0.045 ± 0.001. Retrieval Top-1/Top-5: 96.8%/99.6%. Conflict detection: 97.4%. Token gating: 46.2%. Inference latency: 11.2 ms/sample. Inference VRAM: 3.1 GB.

## 5. Ablation Logic

The six reported configurations progressively isolate model capacity, overlap suppression, intervention-based contribution analysis, and the final contribution-gated configuration. The complete table is maintained in `results/tables/main_ablation.csv`.

## 6. Result Provenance

The machine-readable run manifest is `results/raw/official_audit_manifest.json`. Aggregate figures are under `results/figures/`; supporting reports are under `docs/`.

## 7. Reproducibility

Use `VAST_AI_TRAINING.md`, `VAST_AI_MED_NEXUS.ipynb`, and `docs/REPRODUCIBILITY.md` for the documented compute and execution workflow.

## 8. Clinical and Scientific Scope

The reported results are research measurements under a defined experimental protocol. They are not prospective clinical validation, regulatory clearance, or authorization for autonomous patient care. Any clinical deployment would require separate validation, governance, privacy/security review, and regulatory assessment.

## 9. Release Statement

**Med-Nexus is released as a complete research implementation with verified experimental evaluation and reproducibility artifacts.**
