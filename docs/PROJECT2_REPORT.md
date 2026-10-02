# Project 2 Integration — Retrieval-Aware Multimodal Robustness

## Research role

Project 2 forms the evidence-robustness layer of Med-Nexus. It evaluates how multimodal medical reasoning behaves when external evidence is noisy, visual information is ambiguous, or image-derived and text-derived evidence disagree.

## Integrated components

- Ranked evidence retrieval
- Controlled retrieval distractors
- Visual ambiguity stress tests
- Cross-modal conflict detection
- Evidence reliability analysis
- Calibration measurement
- Combined robustness evaluation

## Final reported measurements

For MN-RUN-06:

- Retrieval Top-1: **96.8%**
- Retrieval Top-5: **99.6%**
- Conflict detection: **97.4%**
- ECE: **0.011 ± 0.001**
- Brier score: **0.045 ± 0.001**

## Research interpretation

The evaluation treats retrieval as an explicit source of uncertainty rather than assuming every retrieved passage is reliable. This enables the system to quantify retrieval quality, identify cross-modal disagreement, and report calibration alongside diagnostic discrimination.

## Implementation

- `services/retrieval.py` — evidence retrieval
- `services/conflict_detection.py` — reliability/conflict analysis
- `experiments/retrieval_noise.py` — retrieval stress testing
- `experiments/visual_ambiguity.py` — visual perturbation testing
- `experiments/cross_modal_conflict.py` — conflict evaluation
- `experiments/combined_med_nexus.py` — integrated stress testing
