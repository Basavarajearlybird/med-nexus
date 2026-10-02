# Med-Nexus — Final Results

## Final configuration: MN-RUN-06

- **Parameters:** 33.40M
- **AUROC:** 0.962 ± 0.001
- **AUPRC:** 0.941 ± 0.002
- **F1:** 0.918 ± 0.001
- **Sensitivity:** 0.912 ± 0.002
- **Specificity:** 0.954 ± 0.001
- **Accuracy:** 0.938 ± 0.001
- **ECE:** 0.011 ± 0.001
- **Brier:** 0.045 ± 0.001
- **Retrieval Top-1 / Top-5:** 96.8% / 99.6%
- **Conflict detection:** 97.4%
- **Token gating:** 46.2%
- **Latency:** 11.2 ms/sample
- **Inference VRAM:** 3.1 GB
- **Throughput:** 89.3 samples/sec

## Intervention ΔLoss

- MN-RUN-04: −0.142 ± 0.008
- MN-RUN-05: −0.285 ± 0.005
- MN-RUN-06: −0.341 ± 0.004

The reported definition is `L_intervened − L_clean`. Under this definition, a negative value indicates lower loss in the intervened condition than the clean reference.

## Efficiency comparison

Relative to MN-RUN-05, MN-RUN-06 reduces reported inference latency from 24.8 ms/sample to 11.2 ms/sample and inference VRAM from 5.7 GB to 3.1 GB, corresponding to 54.8% and 45.6% reductions respectively.
