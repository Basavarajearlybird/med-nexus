# GitHub Upload Guide

## 1. Create repository

Create an empty GitHub repository named:

`med-nexus-diagnostic-architecture`

## 2. From the project directory

```bash
git init
git add .
git commit -m "Initial Med-Nexus research implementation"
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

## 3. Never upload

```text
*.pem
*.key
.env
Vast API keys
SSH private keys
restricted medical datasets
patient-identifying data
large raw dataset archives
```

## 4. Recommended release structure

```text
med-nexus-diagnostic-architecture/
├── api/
├── configs/
├── data/
├── evaluation/
├── experiments/
├── models/
├── services/
├── tests/
├── docs/
├── checkpoints/          # optional, only if legally/distribution permitted
├── results/
├── train.py
├── evaluate.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```
