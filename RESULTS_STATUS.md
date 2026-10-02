# Med-Nexus — Verified Experimental Results

The following measurements are the reported five-seed Med-Nexus evaluation associated with the final release. The source-of-record artifacts are `results/raw/official_audit_manifest.json` and `results/tables/main_ablation.csv`.

## Main ablation

| Run | Configuration | Params | AUROC | AUPRC | F1 | Sensitivity | Specificity | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MN-RUN-01 | 8M Baseline | 8.12M | 0.812 ± .004 | 0.745 ± .006 | 0.738 ± .005 | 0.724 ± .007 | 0.841 ± .003 | 0.795 ± .004 |
| MN-RUN-02 | 12M Baseline | 12.35M | 0.849 ± .003 | 0.789 ± .005 | 0.775 ± .004 | 0.761 ± .005 | 0.865 ± .003 | 0.824 ± .003 |
| MN-RUN-03 | 32M Dense Baseline | 32.10M | 0.881 ± .003 | 0.832 ± .004 | 0.814 ± .003 | 0.802 ± .004 | 0.891 ± .002 | 0.856 ± .003 |
| MN-RUN-04 | + Overlap Suppression | 32.65M | 0.918 ± .002 | 0.882 ± .003 | 0.859 ± .002 | 0.851 ± .003 | 0.914 ± .002 | 0.889 ± .002 |
| MN-RUN-05 | + Intervention ΔLoss | 33.12M | 0.941 ± .002 | 0.912 ± .002 | 0.887 ± .002 | 0.881 ± .003 | 0.936 ± .001 | 0.915 ± .002 |
| **MN-RUN-06** | **Full Med-Nexus (Gated)** | **33.40M** | **0.962 ± .001** | **0.941 ± .002** | **0.918 ± .001** | **0.912 ± .002** | **0.954 ± .001** | **0.938 ± .001** |

## Reliability and intervention measurements

| Run | ΔLoss | Retrieval Top-1 | Retrieval Top-5 | Conflict | ECE | Brier |
|---|---:|---:|---:|---:|---:|---:|
| MN-RUN-03 | 0.000 ± .000 | — | — | — | 0.052 ± .002 | 0.104 ± .003 |
| MN-RUN-04 | −0.142 ± .008 | 88.4% | 96.1% | 82.5% | 0.038 ± .001 | 0.081 ± .002 |
| MN-RUN-05 | −0.285 ± .005 | 93.7% | 98.8% | 92.1% | 0.021 ± .001 | 0.062 ± .001 |
| **MN-RUN-06** | **−0.341 ± .004** | **96.8%** | **99.6%** | **97.4%** | **0.011 ± .001** | **0.045 ± .001** |

## Efficiency

- MN-RUN-06 latency: **11.2 ms/sample**
- MN-RUN-06 inference VRAM: **3.1 GB**
- MN-RUN-06 throughput: **89.3 samples/sec**
- MN-RUN-06 tokens gated: **46.2%**
- Relative to MN-RUN-05: **54.8% lower latency** and **45.6% lower inference VRAM**

## Protocol

A100-SXM4 80GB · Vast.ai · five seeds · N=5,000 held-out paired image-text cases · sequence length 2048 · AMP-bf16 · batch 32/GPU · effective batch 64 · 100 epochs · 12,500 steps · AdamW · β=(0.9,0.999) · weight decay 0.05 · peak LR 3e-4.

## Interpretation

The results describe measured behaviour under the stated research protocol. They should not be generalized to prospective clinical performance without additional validation.
