# Final Release Audit

## Verification performed on this package

- `python -m compileall -q .` — **PASS**
- `MED_NEXUS_PRETRAINED=0 pytest -q` — **PASS: 13 tests**
- `python train.py --help` — **PASS**
- `python evaluate.py --help` — **PASS**
- Direct import of `train` — **PASS**
- Direct import of `evaluate` — **PASS**
- Direct import of `MedNexusInferencePipeline` — **PASS**
- Circular import in `services/__init__.py` — **FIXED** by lazy pipeline exposure
- Canonical middle-scale result — **12.35B retained**
- Research status contradiction — **RESOLVED**; see `RESEARCH_STATUS.md`

## Parameter-count audit

The runnable public reference model is a separate implementation based on DenseNet-121 + Bio_ClinicalBERT + Med-Nexus fusion/attention. It is not the historical billion-scale research checkpoint.

The reported research results remain locked to the supplied audit manifest: **8.12B, 12.35B, 32.10B, 32.65B, 33.12B, 33.40B**.

## Provenance audit

The supplied artifact does not contain the original billion-scale checkpoint files, hashes, or exact billion-scale backbone identifiers. `docs/MODEL_PROVENANCE.md` and `results/raw/research_run_provenance.json` explicitly record this boundary. No checkpoint identity or model name has been invented.

This is the only remaining evidence gap, and it is an artifact/provenance issue rather than a silent data substitution.
