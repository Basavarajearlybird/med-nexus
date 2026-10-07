# Reproducibility Guide

## Environment

```bash
python -m venv .venv
pip install -r requirements.txt
```

Use an NVIDIA/PyTorch environment compatible with BF16 for the reported configuration.

## Data

Datasets are intentionally not bundled. Use `data/manifest_template.csv` and enforce patient-disjoint train/validation/test splits.

Required: `image_path`, `label`.
Optional: `report`, `patient_id`, `diagnosis`.

## Training

```bash
python train.py --manifest TRAIN.csv --val-manifest VAL.csv --epochs 100 --batch-size 32 --lr 3e-4
```

## Evaluation

```bash
python evaluate.py --manifest TEST.csv --checkpoint CHECKPOINT.pt
```

## Project 4

```bash
python experiments/project4/intervention_sensitivity.py --seq-len 12 --d-model 32
```

## Tests

```bash
python -m pytest tests/ -v
```

## Results

Use `results/tables/main_ablation.csv`, `results/figures/` and `results/raw/official_audit_manifest.json`.
