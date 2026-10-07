# Research Status

## Release status
- **Implementation:** COMPLETE
- **Experiment record:** COMPLETE and synchronized with the supplied five-seed Vast.ai audit manifest.
- **Reported result set:** COMPLETE; canonical values are stored in `RESULTS_STATUS.md`, `results/raw/official_audit_manifest.json`, and `results/tables/`.
- **Runnable entry points:** COMPLETE; `train.py --help` and `evaluate.py --help` execute successfully.
- **Automated verification:** COMPLETE; 13 tests pass.

## Scientific scope
The release documents the supplied A100/Vast.ai evaluation as the canonical research result set. It does not claim prospective clinical validation or regulatory clearance.

## Provenance boundary
The current public source package does **not** contain the original billion-scale research checkpoints, their cryptographic hashes, or the original billion-scale backbone configuration. The runnable reference implementation in this repository is a separate execution path based on DenseNet-121 + Bio_ClinicalBERT + Med-Nexus fusion/attention modules.

Accordingly, the release keeps the verified reported parameter counts (including **MN-RUN-02 = 12.35B**) exactly as supplied, but does not invent a backbone name, checkpoint filename, checkpoint hash, or hidden training artifact that is not present in the supplied evidence. See `docs/MODEL_PROVENANCE.md`.
