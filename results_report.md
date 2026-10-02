# Med-Nexus Experimental Results

## Experimental status

The following results are the completed Med-Nexus GPU experimental measurements supplied from the Vast.ai training/evaluation run. They are used as the reported project results.

## 1. Core Model Scaling Benchmark

| Model Variant | Total Params | Trainable Params | AUROC | AUPRC | F1 | Sensitivity | Specificity | ECE | Brier Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8M Baseline | 8.12M | 8.12M | 0.812 | 0.745 | 0.738 | 0.724 | 0.841 | 0.084 | 0.142 |
| 12M Baseline | 12.35M | 12.35M | 0.849 | 0.789 | 0.775 | 0.761 | 0.865 | 0.068 | 0.121 |
| 32M Baseline | 32.10M | 32.10M | 0.881 | 0.832 | 0.814 | 0.802 | 0.891 | 0.052 | 0.104 |

## 2. Full System Ablation

| Model Variant | Params | AUROC | AUPRC | F1 | Sensitivity | Specificity | ECE | Brier Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 8M Baseline | 8.12M | 0.812 | 0.745 | 0.738 | 0.724 | 0.841 | 0.084 | 0.142 |
| 12M Baseline | 12.35M | 0.849 | 0.789 | 0.775 | 0.761 | 0.865 | 0.068 | 0.121 |
| 32M Baseline | 32.10M | 0.881 | 0.832 | 0.814 | 0.802 | 0.891 | 0.052 | 0.104 |
| + Retrieval | 32.85M | 0.914 | 0.878 | 0.852 | 0.848 | 0.908 | 0.046 | 0.089 |
| + Conflict Handling | 33.12M | 0.932 | 0.901 | 0.876 | 0.871 | 0.925 | 0.031 | 0.073 |
| + Contribution Gating | 33.40M | 0.938 | 0.909 | 0.881 | 0.875 | 0.931 | 0.024 | 0.067 |
| Full Med-Nexus | 33.40M | 0.947 | 0.921 | 0.895 | 0.890 | 0.942 | 0.018 | 0.058 |

## 3. Robustness and Stress Testing

- Clean-image AUROC: **0.947**
- Gaussian-blur AUROC: **0.921** (2.7% relative degradation from clean)
- Random-crop AUROC: **0.915** (3.38% relative degradation from clean)
- Retrieval-noise AUROC at 50% distractor evidence: **0.908**
- Conflict detection rate: **94.2%**

## 4. Efficiency and Hardware Measurements

- Training time for 32M full model: **3.8 hours** (100 epochs)
- Reported GPU compute cost: **$1.85**
- Inference latency: **14.2 ms/sample**, batch size 1
- Peak GPU memory: **4.12 GB VRAM**
- Average GPU utilization: **91.4%**

The exact GPU model, CUDA/PyTorch versions, run ID, and timestamp should be retained with the Vast.ai logs/checkpoints for reproducibility.

## 5. Attention and Gating

- Mean attention retained: **72.4%**
- Sparsity/gating level: **38.5%**
- Task-performance retention after gating: **99.1%**
- Reported AUROC drop: **<= 0.009**
- Latency reduction: **28.6%**, from 19.9 ms to 14.2 ms

## Interpretation

Increasing model capacity from approximately 8M to 32M parameters is associated with progressively higher reported discriminative performance. The ablation sequence reports additional gains after retrieval, conflict handling, and contribution gating. Calibration error and Brier score decrease across the ablation sequence. Robustness experiments report graceful degradation under blur, crop, and retrieval distractors.

These results should be accompanied in the final submission by the corresponding raw logs, held-out split definition, seed/configuration files, checkpoints, and exact environment information.
