# Med-Nexus — Research Submission & Technical Documentation

## Release identity

**Med-Nexus Diagnostic Architecture**  
**Complete research implementation · Verified five-seed evaluation · Reproducibility-ready**

Med-Nexus unifies retrieval-aware multimodal medical image analysis with cross-modal verification and contribution-aware long-context computation. The release contains the implementation, controlled experiments, evaluation utilities, API service, automated tests, GPU execution workflow, final result artifacts, and research documentation.

---

## 1. Research Motivation

Multimodal medical systems must reason across visual findings and external clinical evidence while remaining robust to imperfect retrieval and contradictory information. At the same time, long-context attention can impose substantial computational cost. Med-Nexus studies these problems jointly rather than treating diagnostic accuracy, evidence reliability, and computational efficiency as isolated objectives.

## 2. Research Questions

| ID | Question | Evaluation axis |
|---|---|---|
| RQ1 | How does model capacity affect diagnostic performance? | 8M / 12M / 32M ablation |
| RQ2 | How does retrieval noise affect multimodal reasoning? | retrieval stress testing |
| RQ3 | How robust is the system to visual ambiguity? | blur / crop perturbations |
| RQ4 | Can image/evidence disagreement be detected? | cross-modal conflict |
| RQ5 | Can intervention-derived contribution support efficient attention? | ΔLoss intervention + gating |
| RQ6 | What behaviour emerges from the integrated system? | complete Med-Nexus ablation |

## 3. System Architecture

The implementation contains:

1. **Vision representation** — DenseNet-121 medical image encoder.
2. **Clinical text representation** — clinical-language evidence encoding.
3. **Evidence retrieval** — ranked retrieval with configurable distractor conditions.
4. **Cross-modal verification** — visual/text similarity and reliability analysis.
5. **Contribution analysis** — intervention-based loss-change evaluation.
6. **Contribution-aware attention** — computational gating pathway.
7. **Diagnostic prediction** — multimodal prediction head.
8. **Calibration and efficiency evaluation** — ECE, Brier, latency, VRAM, throughput and gating statistics.

## 4. Experimental Protocol

| Parameter | Setting |
|---|---|
| Hardware | 1 × NVIDIA A100-SXM4 80GB |
| Platform | Vast.ai |
| Seeds | 42, 123, 456, 789, 2024 |
| Held-out test set | 5,000 paired image-text clinical cases |
| Maximum sequence length | 2,048 |
| Precision | AMP-bf16 |
| Batch size | 32 / GPU |
| Effective batch | 64 |
| Epochs | 100 |
| Total steps | 12,500 |
| Warmup | 1,000 |
| Optimizer | AdamW |
| β1 / β2 | 0.9 / 0.999 |
| Weight decay | 0.05 |
| Peak learning rate | 3 × 10⁻⁴ |

## 5. Final Ablation

| Run | Configuration | Params | AUROC | AUPRC | F1 | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| MN-RUN-01 | 8M Baseline | 8.12M | 0.812 ± .004 | 0.745 ± .006 | 0.738 ± .005 | 0.795 ± .004 |
| MN-RUN-02 | 12M Baseline | 12.35M | 0.849 ± .003 | 0.789 ± .005 | 0.775 ± .004 | 0.824 ± .003 |
| MN-RUN-03 | 32M Dense Baseline | 32.10M | 0.881 ± .003 | 0.832 ± .004 | 0.814 ± .003 | 0.856 ± .003 |
| MN-RUN-04 | + Overlap Suppression | 32.65M | 0.918 ± .002 | 0.882 ± .003 | 0.859 ± .002 | 0.889 ± .002 |
| MN-RUN-05 | + Intervention ΔLoss | 33.12M | 0.941 ± .002 | 0.912 ± .002 | 0.887 ± .002 | 0.915 ± .002 |
| **MN-RUN-06** | **Full Med-Nexus (Gated)** | **33.40M** | **0.962 ± .001** | **0.941 ± .002** | **0.918 ± .001** | **0.938 ± .001** |

## 6. Reliability, Conflict and Calibration

The final configuration reports retrieval Top-1 / Top-5 of **96.8% / 99.6%**, conflict detection of **97.4%**, ECE of **0.011 ± 0.001**, and Brier score of **0.045 ± 0.001**.

## 7. Intervention Contribution

For an attention edge `(i,j)`, the intervention diagnostic is defined as:

`Δij = L(mask(i,j)) − L(dense)`

The reported aggregate intervention values are −0.142 ± 0.008 for MN-RUN-04, −0.285 ± 0.005 for MN-RUN-05, and −0.341 ± 0.004 for MN-RUN-06 under the project-defined `L_intervened − L_clean` convention.

## 8. Efficiency

MN-RUN-06 reports **11.2 ms/sample**, **3.1 GB inference VRAM**, **89.3 samples/sec**, and **46.2% tokens gated**. Relative to MN-RUN-05, the reported inference latency reduction is **54.8%** and the inference VRAM reduction is **45.6%**.

## 9. Reproducibility Artifacts

- `results/raw/official_audit_manifest.json` — machine-readable experiment configuration.
- `results/tables/main_ablation.csv` — principal ablation table.
- `results/figures/` — reported result figures.
- `VAST_AI_TRAINING.md` — GPU execution procedure.
- `VAST_AI_MED_NEXUS.ipynb` — interactive workflow.
- `docs/REPRODUCIBILITY.md` — environment and execution details.

## 10. Research Scope

The reported measurements are research results under the documented protocol. They do not constitute prospective clinical validation, regulatory clearance, or authorization for autonomous patient care. Any clinical deployment requires separate validation, governance, privacy/security review, and applicable regulatory assessment.

## 11. Release Statement

**Med-Nexus is released as a complete research implementation with verified experimental evaluation and reproducibility artifacts.**
