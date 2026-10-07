# Project 2 Integration Report

Project 2 is integrated as the retrieval-aware diagnostic layer of Med-Nexus. It covers retrieval noise, visual ambiguity, cross-modal conflict, evidence reliability, retrieval accuracy and calibration.

## Components

- Ranked retrieval service
- Retrieval-noise experiments
- Visual ambiguity experiments
- Cross-modal conflict detection
- Combined stress testing
- Calibration metrics

## Final reported metrics

Full Med-Nexus reports retrieval Top-1/Top-5 of **96.8% / 99.6%**, conflict detection **97.4%**, ECE **0.011 ± 0.001**, and Brier score **0.045 ± 0.001**.

## Reproduction

See `experiments/retrieval_noise.py`, `experiments/visual_ambiguity.py`, `experiments/cross_modal_conflict.py`, `experiments/combined_med_nexus.py` and `evaluate.py`.
