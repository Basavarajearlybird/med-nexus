# GitHub Release Guide

## Repository

Recommended repository name: `med-nexus`

## Upload from the project directory

```bash
git init
git add .
git commit -m "Release complete Med-Nexus research system"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

## Never upload

```text
*.pem
*.key
.env
Vast API keys
SSH private keys
restricted medical datasets
patient-identifying data
large raw dataset archives
large model checkpoints unless distribution is explicitly permitted
```

## Recommended root

```text
med-nexus/
├── api/
├── configs/
├── data/
├── evaluation/
├── experiments/
├── models/
├── services/
├── tests/
├── docs/
├── results/
├── train.py
├── evaluate.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```
