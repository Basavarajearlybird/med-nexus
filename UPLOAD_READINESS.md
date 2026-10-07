# Upload Readiness

This package has been audited before upload.

## Verified
- 13 automated tests pass.
- `train.py --help` works.
- `evaluate.py --help` works.
- Circular package import is fixed.
- Canonical MN-RUN-02 parameter count is **12.35B**.
- All canonical result tables are synchronized.
- Research status is synchronized.
- Model/checkpoint provenance is documented without invented identifiers.

## Important provenance boundary
The supplied research-result artifact does not contain the original billion-scale checkpoint files, hashes, or exact billion-scale backbone identifier. The package therefore does not fabricate them. See `docs/MODEL_PROVENANCE.md`.

The runnable source implementation uses DenseNet-121 + Bio_ClinicalBERT + Med-Nexus fusion/attention. The reported billion-scale runs are retained as the supplied experimental record.
