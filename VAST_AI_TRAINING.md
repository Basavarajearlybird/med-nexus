# VAST.ai Training Runbook

This file is an operational guide, not a claim that training has already been completed.

## 1. Prepare a real manifest
Create a CSV with `image_path,label,report,patient_id,diagnosis`. Keep all studies from one patient in a single split. Do not use the smoke URLs for research results.

## 2. Verify locally
```bash
python -m pytest tests/ -v
python train.py --help
python evaluate.py --help
```

## 3. Upload to VAST.ai
Use a CUDA-enabled PyTorch image compatible with the installed torch/torchvision versions. Install `requirements.txt`. Copy the repository and dataset/manifest.

## 4. Train
```bash
python train.py --manifest data/full/manifest.csv --root-dir data/full --epochs 5 --batch-size 8 --lr 1e-4 --out checkpoints/med_nexus.pt
```
Adjust batch size to GPU memory. Do not assume a fixed GPU, price or runtime.

## 5. Evaluate
```bash
python evaluate.py --manifest data/test/manifest.csv --root-dir data/test --checkpoint checkpoints/med_nexus.pt --out results/test_metrics.json
```

## 6. Research stress tests
After the trained checkpoint is verified, run retrieval-noise, visual-ambiguity, conflict, attention and combined experiments. Save raw JSON/CSV, configs, seeds, checkpoint hash, GPU type and runtime.

## 7. Budget safety
The ₹3,000 GPU allocation is a hard budget. Record start/end time and hourly price before launching. Stop the instance after copying checkpoints and results. Never spend GPU time debugging import errors.
