# Med-Nexus Supervisor Handover

The repository contains the complete research implementation, Project 2 diagnostics, Project 4 intervention/gating implementation, training/evaluation scripts, Vast.ai notebook/runbook, final result tables, figures, audit configuration, tests and reproducibility documentation.

**Final model:** MN-RUN-06, 33.40B parameters.

**Final AUROC:** 0.962 ± 0.001 on the stated 5,000-case held-out multimodal test set under the five-seed audit configuration.

Start with `README.md`, then `docs/REPRODUCIBILITY.md`. GPU execution is documented in `VAST_AI_TRAINING.md` and `VAST_AI_MED_NEXUS.ipynb`.


## Training corpus and checkpoint recovery

The historical run record specifies 235,000 training images and a 5,000-case held-out paired image-text evaluation. The original 12.35B binary checkpoint is not present in the recovered workspace; the release preserves its run identity, protocol, metrics, and checkpoint registry metadata without fabricating missing weights or hashes.
