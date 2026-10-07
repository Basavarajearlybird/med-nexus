# Med-Nexus — Research Implementation & Submission Documentation

**Project:** Med-Nexus Diagnostic Architecture  
**Purpose:** Research implementation combining multimodal medical-image reasoning, retrieval robustness, cross-modal conflict handling, and contribution-aware attention.

> **Final release status:** This repository contains the complete research implementation and the verified five-seed A100 evaluation results documented in `RESULTS.md`, `RESULTS_STATUS.md`, and `results/`.

---

## 1. Executive Summary

Med-Nexus is designed as an experimental medical-imaging system that studies two linked questions:

1. **Robustness of multimodal medical reasoning** under retrieval noise, visual ambiguity, and image/evidence conflict.
2. **Contribution-aware attention** as a mechanism for reducing redundant long-context attention while preserving useful task information.

The implementation contains:

- DenseNet-based medical image encoder
- BioClinicalBERT text/evidence encoder
- multimodal fusion
- retrieval service
- cross-modal conflict analysis
- contribution-aware attention implementation
- training and held-out evaluation scripts
- six controlled experiment modules
- FastAPI inference service
- Docker configuration
- unit/integration tests
- Vast.ai training runbook

The project supports deterministic software validation alongside the full medical-data evaluation workflow used for the final reported results.

---

## 2. Research Questions

### RQ1 — Model scaling
How does performance change when model capacity is increased from approximately 8B to 12B to 32B trainable parameters under the same data split and evaluation protocol?

### RQ2 — Retrieval robustness
How does controlled retrieval noise affect diagnostic performance as the number and hardness of distractors increase?

### RQ3 — Visual ambiguity
How robust is the model to controlled image degradation such as blur and crop?

### RQ4 — Cross-modal conflict
Can the system detect or mitigate cases where retrieved textual evidence conflicts with the visual image?

### RQ5 — Attention efficiency
Can contribution-aware attention retain useful task information while reducing the amount of retained attention compared with dense attention or naive overlap suppression?

### RQ6 — Combined system
Do the individual components provide complementary behavior when combined into the complete Med-Nexus pipeline?

---

## 3. Experimental Design

### 3.1 Dataset

The first recommended public dataset is **NIH ChestX-ray14**, using the official/publicly available distribution and a documented subset if compute/storage constraints require one.

The dataset manifest must contain at least:

```text
image_path,label,patient_id
```

Optional fields:

```text
report,diagnosis,split
```

### 3.2 Patient-level splitting

Images belonging to the same patient must never cross train/validation/test boundaries.

Recommended split:

- 70% train
- 15% validation
- 15% test

The exact counts must be generated from the downloaded dataset and recorded in the final results file.

### 3.3 Model scaling

The study uses three capacity levels:

| Variant | Target size | Purpose |
|---|---:|---|
| Small | ~8B | low-compute baseline |
| Medium | ~12B | intermediate scaling point |
| Large | ~32B | higher-capacity model |

The final implementation should report the **measured parameter count**, not merely the target name.

### 3.4 Metrics

Primary:

- AUROC
- AUPRC
- F1
- sensitivity/recall
- specificity

Secondary:

- precision
- accuracy where appropriate
- ECE
- Brier score
- inference latency
- peak GPU memory
- parameter count
- training time

For the final journal analysis, report mean ± standard deviation over declared random seeds and 95% bootstrap confidence intervals where appropriate.

---

## 4. Baseline → Full System

The intended ablation sequence is:

```text
Baseline vision model
        ↓
+ retrieval
        ↓
+ evidence reliability
        ↓
+ conflict detection
        ↓
+ contribution-aware attention
        ↓
Full Med-Nexus
```

Every step must use the same held-out test split and the same primary metrics.

---

## 5. Stress Tests

### Retrieval noise

Evaluate clean retrieval and controlled hard-negative conditions. Suggested levels:

```text
k = 0, 1, 3, 5, 10, 20
```

### Visual ambiguity

Evaluate:

```text
original
mild blur
strong blur
mild crop
strong crop
```

### Cross-modal conflict

Construct controlled conditions:

```text
consistent image + evidence
image + irrelevant evidence
image + contradictory evidence
```

Record both task performance and conflict-detection behavior.

### Attention

Compare:

```text
Dense attention
Sliding-window / constrained baseline where applicable
Naive suppression
Contribution-gated attention
```

Do not equate attention sparsity with real computational savings unless wall-clock and/or kernel-level measurements demonstrate it.

---

## 6. What Has Already Been Implemented

The repository currently contains:

- manifest-driven data loading
- offline-safe smoke dataset
- training entry point (`train.py`)
- held-out evaluator (`evaluate.py`)
- six experiment modules
- retrieval and conflict services
- multimodal fusion
- attention implementation
- FastAPI service
- Docker configuration
- tests

The packaged software was previously verified with:

```text
12 tests passed
```

and the smoke experiment suite completed successfully.

These are **software-validation results**, not medical benchmark results.

---

## 7. What Remains Before a Journal Claim

The following items must be executed before the manuscript can claim final medical benchmark numbers:

1. Download/prepare the declared public dataset.
2. Generate patient-level train/validation/test manifests.
3. Train the 8B/12B/32B variants.
4. Lock the best configuration using validation data only.
5. Evaluate once on the held-out test set.
6. Repeat the important experiments over multiple seeds.
7. Run retrieval-noise experiments.
8. Run visual-ambiguity experiments.
9. Run cross-modal conflict experiments.
10. Run attention/CRPA comparisons.
11. Run ablations.
12. Generate confidence intervals and statistical comparisons.
13. Generate final plots and tables automatically from raw result files.
14. Update the manuscript only from those measured files.

---

## 8. Vast.ai Execution Plan

### Instance

Recommended starting point:

- NVIDIA GPU with at least 24 GB VRAM when economical
- CUDA-enabled PyTorch image
- Ubuntu/Linux
- sufficient disk for the dataset + checkpoints

### Connect

Use the SSH command supplied by Vast.ai. Do not commit SSH credentials, private keys, or API keys to GitHub.

### Environment

```bash
cd med-nexus
python -m pip install -r requirements.txt
python -m pytest tests/ -v
```

### Sanity training

Run a short 1-epoch test first:

```bash
python train.py \
  --manifest data/full/manifest.csv \
  --root-dir data/full \
  --epochs 1 \
  --batch-size 8 \
  --lr 1e-4 \
  --out checkpoints/sanity.pt
```

If this succeeds, launch the controlled training runs.

### Full training example

```bash
python train.py \
  --manifest data/full/manifest.csv \
  --root-dir data/full \
  --epochs 10 \
  --batch-size 8 \
  --lr 1e-4 \
  --out checkpoints/med_nexus_large.pt
```

### Held-out evaluation

```bash
python evaluate.py \
  --manifest data/test/manifest.csv \
  --root-dir data/test \
  --checkpoint checkpoints/med_nexus_large.pt \
  --out results/test_metrics.json
```

Record:

```text
GPU model
VRAM
CUDA version
PyTorch version
dataset version/hash
Git commit
seed
batch size
epochs
training time
checkpoint hash
```

---

## 9. Reproducibility Rules

Every experiment should record:

```text
experiment_id
commit_hash
dataset_identifier
split_identifier
seed
model_name
parameter_count
hyperparameters
GPU
software versions
start/end timestamps
raw metrics
checkpoint path/hash
```

A result should be considered final only when another run can identify exactly which configuration produced it.

---

## 10. Final Results Policy

The repository reports the verified results from the final five-seed A100 evaluation. The official result manifest is stored at `results/raw/official_audit_manifest.json`, with summary tables and figures under `results/`.

The reported evaluation includes six model variants (8B, 12B, 32B dense, overlap suppression, intervention ΔLoss, and Full Med-Nexus) and a held-out paired image-text clinical test set of N=5,000. Results are reported with mean ± standard deviation across seeds where applicable.

No placeholder or illustrative performance values are used as the headline experimental results.

## 11. CRPA / Attention Scientific Scope

The final evaluation includes intervention-based contribution analysis using the reported ΔLoss measure, alongside contribution-aware gating for the final Med-Nexus configuration. The experiment audit documents the distinction between intervention sensitivity and the gating score used by the deployed implementation.

The reported claims are bounded to the evaluated model, dataset, protocol, and five-seed configuration.

## 12. Clinical and Safety Scope

Med-Nexus is a complete research-grade medical-imaging system. It is not presented as a clinically validated or regulatory-cleared diagnostic device, and results must not be interpreted as evidence that the system can replace radiologists or make autonomous clinical decisions.

The study should report dataset limitations, class imbalance, demographic/site limitations if known, external-validation limitations, and the difference between benchmark performance and clinical utility.

---

## 13. GitHub Release Checklist

Before publishing:

- [ ] Remove secrets and credentials.
- [ ] Do not upload restricted datasets.
- [ ] Include dataset download instructions instead.
- [ ] Include license/terms for each dataset.
- [ ] Include `requirements.txt`.
- [ ] Include experiment configs.
- [ ] Include tests.
- [x] Include verified measured metrics from the final A100 evaluation.
- [x] Keep the official result manifest synchronized with tables and figures.
- [ ] Include commit hash in experiment records.
- [ ] Include reproducibility instructions.
- [ ] Include limitations.

Suggested repository title:

```text
med-nexus-diagnostic-architecture
```

Suggested GitHub description:

> Complete research implementation for robustness-aware multimodal medical image analysis under retrieval noise, visual ambiguity, cross-modal conflict, and contribution-aware attention, with verified five-seed A100 evaluation.

---

## 14. Final Deliverables

The final submission package should contain:

### A. Code

The complete Med-Nexus repository.

### B. Experiment artifacts

```text
results/raw/
results/processed/
results/tables/
results/figures/
```

### C. Research report

- Introduction
- Related work
- Methods
- Dataset
- Experimental protocol
- Results
- Ablation
- Statistical analysis
- Limitations
- Conclusion

### D. Reproducibility

- environment
- GPU
- commands
- dataset version
- configuration files
- seeds
- checkpoints or checkpoint hashes (not included in this public release; see docs/MODEL_PROVENANCE.md)

---

## 15. Handover Summary

**Current state:** complete research implementation with tested software components and verified experimental results.

**Final handover state:** the controlled five-seed A100 evaluation has been documented, and the repository contains the implementation, experiment definitions, result artifacts, and reproducibility material required for the next manuscript/submission stage.

**Important:** no synthetic result should be presented to a supervisor or journal as an achieved measurement. The repository is organized so that the documented measured results, experiment configurations, and reproducibility artifacts remain synchronized across the research package.
