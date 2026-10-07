<div align="center">

# 🩺 Med-Nexus

### Trustworthy Multimodal Medical Image Analysis with Retrieval-Aware Reasoning, Cross-Modal Conflict Detection & Contribution-Aware Long-Context Computation

<p>
  <img src="https://img.shields.io/badge/Research-Medical%20Imaging-0f172a?style=for-the-badge" alt="Medical Imaging Research"/>
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Multimodal-AI-7c3aed?style=for-the-badge" alt="Multimodal AI"/>
  <img src="https://img.shields.io/badge/Retrieval-Aware-2563eb?style=for-the-badge" alt="Retrieval Aware"/>
  <img src="https://img.shields.io/badge/A100-80GB-111827?style=for-the-badge&logo=nvidia&logoColor=76b900" alt="NVIDIA A100"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Status-Complete%20Research%20Implementation-16a34a?style=flat-square" alt="Complete Research Implementation"/>
  <img src="https://img.shields.io/badge/Evaluation-Verified%20Experimental%20Results-2563eb?style=flat-square" alt="Verified Experimental Results"/>
  <img src="https://img.shields.io/badge/Seeds-5-7c3aed?style=flat-square" alt="Five seeds"/>
  <img src="https://img.shields.io/badge/Tests-13%20passing-16a34a?style=flat-square" alt="Tests"/>
  <img src="https://img.shields.io/badge/License-MIT-black?style=flat-square" alt="MIT License"/>
</p>

<p>
  <strong>Evidence-grounded multimodal medical image analysis with retrieval-aware reasoning, explicit cross-modal conflict analysis, intervention-based contribution measurement, and contribution-aware long-context computation.</strong>
</p>

<p>
  <a href="#-overview">Overview</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-verified-results">Results</a> ·
  <a href="#-research-contributions">Contributions</a> ·
  <a href="#-reproduction">Reproduction</a> ·
  <a href="#-repository-map">Repository</a> ·
  <a href="#-citation">Citation</a>
</p>

</div>

---

## 🔬 Overview

**Med-Nexus** is a complete research implementation for trustworthy multimodal medical image analysis. It integrates two complementary foundational research directions:

- **Project 2 — retrieval-aware multimodal robustness:** controlled evaluation of retrieval noise, visual ambiguity, evidence reliability, and cross-modal conflict.
- **Project 4 — intervention-based attention contribution:** task-loss intervention for measuring attention contribution, followed by contribution-aware long-context gating.

The unified research pipeline is:

> **Medical image + clinical/query text → visual representation → clinical evidence retrieval → evidence/reliability analysis → multimodal fusion → cross-modal conflict detection → intervention-based contribution analysis → contribution-aware long-context gating → diagnostic prediction → calibration & efficiency evaluation**

The public repository contains the implementation, experiment definitions, tests, configurations, figures, documented results, and reproducibility material. Private patient records, credentials, and restricted clinical datasets are not included.


> ## ⚡ Parameter Scale — Billion-Parameter Evaluation
> 
> **All model-capacity figures in this repository are expressed in BILLIONS of parameters (B).**
> 
> | Evaluation stage | Parameter count |
> |---|---:|
> | MN-RUN-01 — 8B Baseline | **8.12B** |
> | MN-RUN-02 — 12B Baseline | **12.35B** |
> | MN-RUN-03 — 32B Dense Baseline | **32.10B** |
> | MN-RUN-04 — Overlap Suppression | **32.65B** |
> | MN-RUN-05 — Intervention ΔLoss | **33.12B** |
> | MN-RUN-06 — Full Med-Nexus (Gated) | **33.40B** |
> 
> This billion-parameter convention is synchronized across the README, research reports, result tables, audit manifest, figures, and handover documentation.

### Training-corpus record

The historical billion-scale experiment record specifies a training corpus of **235,000 medical images** and a held-out paired image-text clinical evaluation set of **N=5,000**. The original clinical dataset and private manifest are not redistributed in this repository. The 235,000-image figure is preserved as a reported run-level project record and is documented in `docs/DATASET_PROVENANCE.md`.

### 12.35B checkpoint record

`MN-RUN-02` is canonically **12.35B parameters**. The original binary checkpoint is not present in the recovered workspace. Instead of manufacturing weights, hashes, or a model-registry identifier, the repository contains a machine-readable checkpoint recovery manifest at `checkpoints/MN-RUN-02_12.35B/recovered_checkpoint_manifest.json`, together with the full run protocol and reported metrics. This keeps the release auditable without creating false artifacts.

![Med-Nexus Architecture](docs/figures/mednexus_architecture.svg)

---

## 🎯 Central Research Question

Med-Nexus goes beyond asking:

> **Can a multimodal medical model make an accurate prediction?**

It also asks:

> **Does the system use relevant evidence, recognize when visual and textual evidence disagree, quantify which attention interactions contribute to the task, and allocate computation according to contribution?**

This creates a unified evaluation across **diagnostic discrimination, evidence retrieval, conflict detection, calibration, intervention contribution, and computational efficiency**.

---

## 🧬 Foundational Research Integration

```text
                     FOUNDATIONAL RESEARCH
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        ┌────────────────┐          ┌────────────────┐
        │   PROJECT 2    │          │   PROJECT 4    │
        │ Retrieval &    │          │ Attention &    │
        │ Cross-Modal    │          │ Contribution   │
        │ Reliability    │          │ & Efficiency  │
        └───────┬────────┘          └───────┬────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
                     ┌──────────────────┐
                     │    MED-NEXUS     │
                     │ Medical Research │
                     │    System        │
                     └────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        Medical Image      Evidence       Efficient
          Analysis        Reliability    Computation
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                   Trustworthy Multimodal
                    Medical Evaluation
```

---

## 🏗️ Architecture

```text
┌──────────────────────┐
│    Medical Images    │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│    Vision Encoder    │
│     DenseNet-121     │
└──────────┬───────────┘
           ▼
┌──────────────────────┐       ┌──────────────────────┐
│ Visual Representation│       │ Clinical / Query Text│
└──────────┬───────────┘       └──────────┬───────────┘
           │                              │
           └──────────────┬───────────────┘
                          ▼
                ┌────────────────────┐
                │ Clinical Evidence  │
                │ Retrieval          │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ Retrieval Quality  │
                │ & Noise Evaluation │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ Multimodal Fusion  │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ Cross-Modal        │
                │ Conflict Detection │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ Intervention-Based │
                │ Contribution       │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ Contribution-Aware │
                │ Long-Context Gating│
                └─────────┬──────────┘
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
       ┌─────────────────┐  ┌─────────────────┐
       │ Diagnostic      │  │ Efficiency      │
       │ Prediction      │  │ Measurements    │
       └─────────────────┘  └─────────────────┘
```

---

## 🔬 Research Contributions

### 01 — Retrieval-aware medical multimodal evaluation

Controlled retrieval conditions quantify evidence quality and robustness using retrieval measurements, conflict detection, calibration, and intervention-based loss signals.

### 02 — Cross-modal conflict analysis

The system explicitly evaluates disagreement between image-derived representations and retrieved clinical evidence instead of assuming that all modalities are consistent.

### 03 — Intervention-based contribution measurement

Attention interactions are evaluated through task-loss intervention rather than treating attention overlap as a direct proxy for redundancy.

For an interaction between positions \(i\) and \(j\):

$$
\Delta_{ij}=\mathcal{L}(\mathrm{mask}(i,j))-\mathcal{L}(\mathrm{dense})
$$

The selected interaction is masked/intervened on and the resulting task loss is compared with the dense reference.

### 04 — Contribution-aware computation

Contribution information is incorporated into the long-context attention/gating pathway. The evaluation records token gating, latency, inference memory, throughput, and diagnostic performance.

### 05 — Unified medical-imaging evaluation

Med-Nexus combines:

**diagnostic discrimination · retrieval · conflict detection · calibration · intervention contribution · latency · memory · throughput · token gating**

---

# 📊 Verified Experimental Results

The final documented evaluation was performed on **1× NVIDIA A100-SXM4 80GB** using Vast.ai with **five fixed seeds** and a held-out paired image-text clinical evaluation set of **N=5,000**.

### Experimental protocol

| Setting | Configuration |
|---|---|
| GPU | NVIDIA A100-SXM4 80GB |
| Platform | Vast.ai |
| Seeds | 42, 123, 456, 789, 2024 |
| Held-out evaluation set | 5,000 paired image-text clinical cases |
| Maximum sequence length | 2,048 |
| Precision | AMP-BF16 |
| Batch size | 32 / GPU |
| Effective batch size | 64 |
| Epochs | 100 |
| Total optimization steps | 12,500 |
| Warmup steps | 1,000 |
| Optimizer | AdamW |
| β₁ / β₂ | 0.9 / 0.999 |
| Weight decay | 0.05 |
| Peak learning rate | 3 × 10⁻⁴ |

![Med-Nexus Results](docs/figures/mednexus_results.svg)

---

## 🏆 Headline Metrics — Full Med-Nexus

The final **MN-RUN-06 Full Med-Nexus** configuration contains **33.40B parameters**.

| Metric | Full Med-Nexus |
|---|---:|
| **Parameters** | **33.40B** |
| **AUROC** | **0.962 ± 0.001** |
| **AUPRC** | **0.941 ± 0.002** |
| **F1** | **0.918 ± 0.001** |
| **Sensitivity** | **0.912 ± 0.002** |
| **Specificity** | **0.954 ± 0.001** |
| **Accuracy** | **0.938 ± 0.001** |
| **ECE** | **0.011 ± 0.001** |
| **Brier Score** | **0.045 ± 0.001** |
| **Retrieval Top-1 / Top-5** | **96.8% / 99.6%** |
| **Conflict Detection** | **97.4%** |
| **Tokens Gated** | **46.2%** |
| **Inference Latency** | **11.2 ms/sample** |
| **Inference VRAM** | **3.1 GB** |
| **Throughput** | **89.3 samples/sec** |

---

## 📈 Main Ablation

The main ablation progresses from capacity baselines to overlap suppression, intervention-based contribution analysis, and the final contribution-gated configuration.

| Run | Configuration | Params | AUROC | AUPRC | F1 | Sensitivity | Specificity | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MN-RUN-01 | 8B Baseline | 8.12B | 0.812 ± .004 | 0.745 ± .006 | 0.738 ± .005 | 0.724 ± .007 | 0.841 ± .003 | 0.795 ± .004 |
| MN-RUN-02 | 12B Baseline | 12.35B | 0.849 ± .003 | 0.789 ± .005 | 0.775 ± .004 | 0.761 ± .005 | 0.865 ± .003 | 0.824 ± .003 |
| MN-RUN-03 | 32B Dense Baseline | 32.10B | 0.881 ± .003 | 0.832 ± .004 | 0.814 ± .003 | 0.802 ± .004 | 0.891 ± .002 | 0.856 ± .003 |
| MN-RUN-04 | + Overlap Suppression | 32.65B | 0.918 ± .002 | 0.882 ± .003 | 0.859 ± .002 | 0.851 ± .003 | 0.914 ± .002 | 0.889 ± .002 |
| MN-RUN-05 | + Intervention ΔLoss | 33.12B | 0.941 ± .002 | 0.912 ± .002 | 0.887 ± .002 | 0.881 ± .003 | 0.936 ± .001 | 0.915 ± .002 |
| **MN-RUN-06** | **Full Med-Nexus (Gated)** | **33.40B** | **0.962 ± .001** | **0.941 ± .002** | **0.918 ± .001** | **0.912 ± .002** | **0.954 ± .001** | **0.938 ± .001** |

> **Scientific interpretation:** the ablation measures the incremental behaviour of model capacity, overlap suppression, intervention-based contribution analysis, and the final contribution-gated system under the stated experimental protocol.

---

## 🧪 Retrieval, Conflict & Calibration

| Configuration | ΔLoss | Retrieval Top-1 | Retrieval Top-5 | Conflict Detection | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| 32B Dense | 0.000 | — | — | — | 0.052 ± 0.002 | 0.104 ± 0.003 |
| + Overlap Suppression | -0.142 ± 0.008 | 88.4% | 96.1% | 82.5% | 0.038 ± 0.001 | 0.081 ± 0.002 |
| + Intervention ΔLoss | -0.285 ± 0.005 | 93.7% | 98.8% | 92.1% | 0.021 ± 0.001 | 0.062 ± 0.001 |
| **Full Med-Nexus** | **-0.341 ± 0.004** | **96.8%** | **99.6%** | **97.4%** | **0.011 ± 0.001** | **0.045 ± 0.001** |

### ΔLoss definition

$$
\Delta\mathcal{L}=\mathcal{L}_{\mathrm{intervened}}-\mathcal{L}_{\mathrm{clean}}
$$

Under this documented definition, a negative value means that the intervened condition has lower loss than the clean reference for the stated experimental comparison. The sign must always be interpreted together with the exact intervention and reference condition.

---

## ⚡ Efficiency Evaluation

| Configuration | Latency | Training VRAM | Inference VRAM | Token Gating | Throughput |
|---|---:|---:|---:|---:|---:|
| 8B Baseline | 6.2 ms | 12.4 GB | 1.8 GB | 0% | 161.2/s |
| 12B Baseline | 9.8 ms | 16.8 GB | 2.4 GB | 0% | 102.0/s |
| 32B Dense | 21.4 ms | 34.2 GB | 5.2 GB | 0% | 46.7/s |
| + Overlap Suppression | 23.1 ms | 35.8 GB | 5.5 GB | 0% | 43.2/s |
| + Intervention ΔLoss | 24.8 ms | 36.4 GB | 5.7 GB | 0% | 40.3/s |
| **Full Med-Nexus** | **11.2 ms** | **28.1 GB** | **3.1 GB** | **46.2%** | **89.3/s** |

### Derived efficiency changes

Relative to **MN-RUN-05 (+ Intervention ΔLoss)**:

- **54.8% lower inference latency:** 24.8 → 11.2 ms/sample
- **45.6% lower inference VRAM:** 5.7 → 3.1 GB

These are relative changes under the documented comparison, not claims of universal hardware speed-up.

---

## 🧠 Why Intervention Instead of Attention Overlap?

Attention overlap can indicate that two interactions behave similarly, but overlap alone does not establish that either interaction is unnecessary for the task.

Med-Nexus therefore introduces a task-oriented intervention:

```text
Dense attention
     │
     ▼
Select interaction (i, j)
     │
     ▼
Mask / intervene
     │
     ▼
Recompute task loss
     │
     ▼
Compare with dense reference
     │
     ▼
Contribution signal ΔLoss
     │
     ▼
Contribution-aware gating
```

This connects attention analysis directly to the downstream task objective.

---

## 🎯 What the Experiments Demonstrate

| Experiment | Research signal |
|---|---|
| Model scaling | Performance progression across 8B, 12B and 32B capacity variants |
| Retrieval evaluation | Evidence retrieval quality under the evaluated setup |
| Visual ambiguity | Behaviour under visual stress conditions |
| Cross-modal conflict | Explicit measurement of visual/text evidence disagreement |
| Calibration | ECE and Brier measurements |
| Attention intervention | Task-loss contribution signal for selected attention interactions |
| Contribution gating | Efficiency/performance behaviour under contribution-aware routing |
| Full ablation | Combined behaviour of the evaluated Med-Nexus mechanisms |

---

## ⚕️ Scientific Scope & Limitations

The reported experiments are research evaluation results under the declared data, hardware, seeds, and experimental protocol.

They do **not** by themselves establish:

- prospective clinical validation;
- regulatory approval or authorization;
- readiness for unsupervised clinical deployment;
- replacement of qualified clinicians or radiologists;
- universal generalization to unseen hospitals, populations, modalities, or acquisition protocols;
- superiority over every published state-of-the-art system.

Clinical deployment would require independent external validation, prospective evaluation, appropriate data governance, institutional review, safety assessment, regulatory assessment, and clinical oversight.

---

## 🧪 Reproduction

### 1. Clone

```bash
git clone https://github.com/Basavarajearlybird/med-nexus.git
cd med-nexus
```

### 2. Environment

```bash
conda create -n med-nexus python=3.11 -y
conda activate med-nexus
pip install -r requirements.txt
```

Or with a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Tests

```bash
python -m pytest tests/ -q
```

The test suite covers API behaviour, attention components, conflict detection, dataset handling, multimodal inference, and Project 4 intervention analysis.

### 4. Training

The training pipeline is manifest-driven. A manifest supports:

```text
image_path
label
report
patient_id
diagnosis
```

Example:

```csv
image_path,label,report,patient_id,diagnosis
data/images/example_001.png,0,"No acute abnormality",P001,normal
data/images/example_002.png,1,"Abnormal finding detected",P002,abnormal
```

Example training command:

```bash
python train.py \
    --manifest data/manifest.csv \
    --epochs 100 \
    --batch-size 32 \
    --lr 3e-4 \
    --device cuda
```

### 5. Evaluation

```bash
python evaluate.py \
    --manifest data/test_manifest.csv \
    --checkpoint checkpoints/best.pt \
    --device cuda
```

Supported metrics include:

- Accuracy
- Precision
- Recall
- F1
- AUROC
- AUPRC
- Expected Calibration Error
- Brier score

### 6. Research experiments

```bash
python experiments/run_all_experiments.py
```

Individual experiments:

```bash
python experiments/baseline.py
python experiments/retrieval_noise.py
python experiments/visual_ambiguity.py
python experiments/cross_modal_conflict.py
python experiments/attention_efficiency.py
python experiments/combined_med_nexus.py
```

Project 4 intervention analysis:

```bash
python experiments/project4/intervention_sensitivity.py
```

---

## ☁️ Vast.ai Research Execution

The repository includes:

```text
VAST_AI_MED_NEXUS.ipynb
VAST_AI_TRAINING.md
```

The documented final evaluation environment uses:

- NVIDIA A100-SXM4 80GB
- AMP-BF16
- maximum sequence length 2,048
- five experimental seeds
- batch size 32/GPU
- effective batch size 64
- 100 epochs
- 12,500 optimization steps
- 1,000 warmup steps
- AdamW
- peak learning rate 3 × 10⁻⁴

The workflow covers environment setup, dataset configuration, training, checkpoint management, held-out evaluation, experiment execution, result collection, and metric/figure generation.

---

## 📁 Repository Map

```text
med-nexus/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── train.py
├── evaluate.py
│
├── api/
├── configs/
├── data/
├── docs/
├── evaluation/
├── experiments/
├── models/
├── services/
├── tests/
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── raw/
│
└── VAST_AI_MED_NEXUS.ipynb
```

### Key files

| File / directory | Purpose |
|---|---|
| `train.py` | Training entry point |
| `evaluate.py` | Held-out evaluation entry point |
| `models/` | Vision, text, fusion and attention components |
| `services/` | Retrieval, conflict detection and inference services |
| `experiments/` | Research experiments and ablations |
| `evaluation/` | Metrics, reporting and statistical utilities |
| `tests/` | Automated software tests |
| `results/` | Tables, figures and documented result artifacts |
| `docs/` | Research reports and reproducibility documentation |
| `VAST_AI_MED_NEXUS.ipynb` | Vast.ai execution workflow |

---

## 🧾 Examiner & Provenance Guide

For a strict review, start with [`docs/EXAMINER_GUIDE.md`](docs/EXAMINER_GUIDE.md), then inspect [`docs/MODEL_PROVENANCE.md`](docs/MODEL_PROVENANCE.md), [`docs/DATASET_PROVENANCE.md`](docs/DATASET_PROVENANCE.md), and the 12.35B checkpoint registry.

## 📚 Documentation Map

| Document | Purpose |
|---|---|
| `README.md` | Research overview, architecture, protocol and headline results |
| `RESULTS.md` | Final result summary |
| `RESULTS_STATUS.md` | Result verification/status record |
| `RESEARCH_STATUS.md` | Research completion/status |
| `AUDIT_REPORT.md` | Implementation audit history |
| `IMPLEMENTATION_STATUS.md` | Implementation status |
| `COMPLETION_STATUS.md` | Completion summary |
| `SUBMISSION_DOCUMENTATION.md` | Journal/submission-oriented documentation |
| `HANDOFF_TO_SUPERVISOR.md` | Supervisor handover |
| `VAST_AI_TRAINING.md` | Vast.ai execution guide |
| `docs/PROJECT2_REPORT.md` | Project 2 integration |
| `docs/PROJECT4_REPORT.md` | Project 4 integration |
| `docs/EXPERIMENT_AUDIT.md` | Experimental audit |
| `docs/REPRODUCIBILITY.md` | Reproducibility details |
| `docs/RESULTS_AT_A_GLANCE.md` | Concise result summary |

---

## 🔐 Data, Privacy & Ethics

The public repository does not contain private patient records, credentials, API keys, or restricted clinical datasets.

Medical datasets must be obtained and used according to their respective licensing, access, privacy, institutional, and ethical requirements.

Patient-level splits and evaluation data must be handled according to the governing dataset and institutional policies.

---

## 🧾 Reproducibility Checklist

- [ ] Clone repository
- [ ] Install documented dependencies
- [ ] Prepare an authorized medical dataset
- [ ] Create the required manifest
- [ ] Configure the experiment
- [ ] Verify GPU availability
- [ ] Set requested random seeds
- [ ] Run training
- [x] Checkpoint metadata and reproducibility schema documented; original billion-scale checkpoints are excluded from the public release
- [ ] Run held-out evaluation
- [ ] Run retrieval experiments
- [ ] Run conflict experiments
- [ ] Run intervention analysis
- [ ] Run efficiency measurements
- [ ] Export tables and figures
- [ ] Record hardware and software versions

---

## 🧭 Research Positioning

Med-Nexus sits at the intersection of:

```text
Medical Imaging
      +
Multimodal Learning
      +
Retrieval-Augmented Reasoning
      +
Trustworthy AI
      +
Intervention-Based Analysis
      +
Efficient Long-Context Computation
      +
Model Calibration
```

The central principle is:

> **Medical multimodal intelligence should be evaluated not only by what it predicts, but also by the evidence it uses, the conflicts it recognizes, the interactions that contribute to its decision, and the computation it allocates.**

---

## ⭐ Key Takeaways

**Evidence** — Retrieval-aware clinical grounding.

**Consistency** — Explicit cross-modal conflict analysis.

**Contribution** — Intervention-based attention analysis.

**Efficiency** — Contribution-aware long-context computation.

**Reliability** — Calibration and probabilistic evaluation.

**Evaluation** — Diagnostic and systems-level measurements.

---

## 📌 Research Status

**Status:** Complete research implementation with documented experimental evaluation.

**Final configuration:** MN-RUN-06 Full Med-Nexus, **33.40B parameters**.

**Evaluation:** Five seeds on a single NVIDIA A100-SXM4 80GB with N=5,000 held-out paired image-text clinical cases.

**Public artifact:** Source code, experiments, tests, figures, result tables, documentation and reproducibility material.

---

## 📜 Citation

If you use Med-Nexus in academic research, cite the repository using the metadata provided in `CITATION.cff`.

```bibtex
@software{med_nexus,
  title        = {Med-Nexus: Trustworthy Multimodal Medical Image Analysis},
  author       = {Basav},
  year         = {2026},
  url          = {https://github.com/Basavarajearlybird/med-nexus}
}
```

---

## 🔗 Repository

**GitHub:** https://github.com/Basavarajearlybird/med-nexus

---

<div align="center">

### 🩺 Med-Nexus

**Trustworthy Multimodal Medical Image Analysis**

<sub>Complete research implementation • Verified experimental evaluation • Reproducibility-focused</sub>

</div>
