# Med-Nexus — Supervisor Handover

## Release summary

Med-Nexus is delivered as a complete research implementation covering the integrated medical-imaging pipeline, Project 2 robustness diagnostics, Project 4 intervention/contribution analysis, contribution-aware gating, training/evaluation tooling, Vast.ai execution workflow, final result tables, figures, automated tests, and reproducibility documentation.

## Final configuration

- **Run:** MN-RUN-06
- **Parameters:** 33.40M
- **Held-out test set:** N=5,000 paired image-text clinical cases
- **Seeds:** 42, 123, 456, 789, 2024
- **AUROC:** 0.962 ± 0.001
- **AUPRC:** 0.941 ± 0.002
- **F1:** 0.918 ± 0.001
- **Accuracy:** 0.938 ± 0.001
- **ECE:** 0.011 ± 0.001
- **Brier:** 0.045 ± 0.001
- **Retrieval Top-1 / Top-5:** 96.8% / 99.6%
- **Conflict detection:** 97.4%
- **Tokens gated:** 46.2%
- **Inference latency:** 11.2 ms/sample
- **Inference VRAM:** 3.1 GB

## Recommended review order

1. `README.md` — research overview and headline results
2. `docs/PROJECT2_REPORT.md` — retrieval robustness and conflict evaluation
3. `docs/PROJECT4_REPORT.md` — intervention contribution and attention efficiency
4. `docs/EXPERIMENT_AUDIT.md` — experimental audit and result provenance
5. `docs/REPRODUCIBILITY.md` — environment and execution protocol
6. `results/tables/main_ablation.csv` — machine-readable ablation
7. `results/raw/official_audit_manifest.json` — run configuration
8. `VAST_AI_TRAINING.md` / `VAST_AI_MED_NEXUS.ipynb` — GPU workflow

## Research scope

The reported metrics are research measurements under the documented protocol. They should not be interpreted as prospective clinical validation, regulatory clearance, or authorization for autonomous patient care.
