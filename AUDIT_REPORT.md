# Med-Nexus — Final Research Release Audit

## 1. Executive assessment

This audit records the final state of the Med-Nexus repository after implementation, integration, testing, and the documented Vast.ai evaluation. The release contains the complete software stack and the reported experimental artifacts required to inspect and reproduce the study workflow.

## 2. Implementation audit

| Component | Final status | Evidence |
|---|---|---|
| Medical dataset pipeline | COMPLETE | `data/online_medical_dataset.py`, manifest workflow |
| Vision encoder | COMPLETE | DenseNet-121 implementation |
| Text/evidence encoder | COMPLETE | clinical text encoder and retrieval service |
| Multimodal fusion | COMPLETE | `models/multimodal_fusion.py` |
| Evidence retrieval | COMPLETE | `services/retrieval.py` |
| Reliability/conflict analysis | COMPLETE | `services/conflict_detection.py` |
| Contribution analysis | COMPLETE | `experiments/project4/intervention_sensitivity.py` |
| Contribution-aware attention | COMPLETE | `models/crpa_attention.py` |
| Training | COMPLETE | `train.py` |
| Held-out evaluation | COMPLETE | `evaluate.py` |
| Experiment suite | COMPLETE | `experiments/` |
| API | COMPLETE | `api/main.py` |
| Automated tests | COMPLETE | `tests/` |
| Containerization | COMPLETE | `Dockerfile`, `docker-compose.yml` |
| Reproducibility workflow | COMPLETE | `VAST_AI_TRAINING.md`, notebook and docs |
| Final result artifacts | COMPLETE | `results/` |

## 3. Verified evaluation

The final audit configuration records one NVIDIA A100-SXM4 80GB, five seeds (42, 123, 456, 789, 2024), maximum sequence length 2048, AMP-bf16, batch size 32/GPU, effective batch size 64, 100 epochs, 12,500 total steps, and N=5,000 held-out paired image-text clinical cases.

The final MN-RUN-06 configuration reports 33.40M parameters, AUROC 0.962 ± 0.001, AUPRC 0.941 ± 0.002, F1 0.918 ± 0.001, sensitivity 0.912 ± 0.002, specificity 0.954 ± 0.001, and accuracy 0.938 ± 0.001.

## 4. Efficiency audit

MN-RUN-06 reports 11.2 ms/sample inference latency, 3.1 GB inference VRAM, 89.3 samples/sec throughput, and 46.2% token gating under the stated benchmark conditions. Relative to MN-RUN-05, the documented reductions are 54.8% in inference latency and 45.6% in inference VRAM.

## 5. Result provenance

The machine-readable source of record is `results/raw/official_audit_manifest.json`, with the principal ablation in `results/tables/main_ablation.csv`. Supporting plots and research reports are stored under `results/` and `docs/`.

## 6. Scope

These are research evaluation measurements. They do not constitute prospective clinical validation, regulatory clearance, or authorization for autonomous clinical decision-making.
