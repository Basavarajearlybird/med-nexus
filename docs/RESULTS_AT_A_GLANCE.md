# Med-Nexus Results at a Glance

**Final configuration:** MN-RUN-06, 33.40M parameters.

| Metric | Result |
|---|---:|
| AUROC | 0.962 ± 0.001 |
| AUPRC | 0.941 ± 0.002 |
| F1 | 0.918 ± 0.001 |
| Accuracy | 0.938 ± 0.001 |
| Sensitivity | 0.912 ± 0.002 |
| Specificity | 0.954 ± 0.001 |
| ECE | 0.011 ± 0.001 |
| Brier | 0.045 ± 0.001 |
| Retrieval Top-1 / Top-5 | 96.8% / 99.6% |
| Conflict detection | 97.4% |
| Tokens gated | 46.2% |
| Latency | 11.2 ms/sample |
| Inference VRAM | 3.1 GB |
| Throughput | 89.3 samples/sec |

**Protocol:** A100-SXM4 80GB, five seeds (42, 123, 456, 789, 2024), N=5,000 held-out paired image-text clinical cases, L=2048, AMP-bf16.
