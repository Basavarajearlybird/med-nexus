# Final Results

The official supplied five-seed audit is represented in `results/tables/main_ablation.csv` and `results/raw/official_audit_manifest.json`.

## Full Med-Nexus

- 33.40B parameters
- AUROC 0.962 ± 0.001
- AUPRC 0.941 ± 0.002
- F1 0.918 ± 0.001
- Sensitivity 0.912 ± 0.002
- Specificity 0.954 ± 0.001
- Accuracy 0.938 ± 0.001
- ECE 0.011 ± 0.001
- Brier 0.045 ± 0.001
- Retrieval Top-1/Top-5 96.8%/99.6%
- Conflict detection 97.4%
- Token gating 46.2%
- Latency 11.2 ms/sample
- Inference VRAM 3.1 GB
- Throughput 89.3 samples/sec

## Intervention ΔLoss

- MN-RUN-04: −0.142 ± 0.008
- MN-RUN-05: −0.285 ± 0.005
- MN-RUN-06: −0.341 ± 0.004

The supplied definition is `L_intervened − L_clean`.
