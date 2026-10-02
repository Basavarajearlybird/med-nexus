# Project 4 — Exact Intervention Experiment

This directory contains the standalone diagnostic needed to test the central Project 4 hypothesis.

## Core measurement

For an attention edge `(i,j)`:

`Delta_ij = L(mask(i,j)) - L(dense)`

where `L(dense)` is the task loss with the edge present and `L(mask(i,j))` is the loss after intervening to remove that edge.

A positive Delta means the edge contributes to the task under this intervention.

## Why this is separate from the Med-Nexus proxy

`models/crpa_attention.py` also contains a fast contribution proxy based on attention weight and value magnitude. That proxy is useful for inference-time gating, but it is not identical to the intervention definition above. The standalone experiment makes that distinction explicit.

## Run

```bash
python experiments/project4/intervention_sensitivity.py --seq-len 12 --d-model 32 --out results/project4_intervention.json
```

The full study should repeat the intervention experiment over declared seeds and task sequences, then compare dense, sliding-window, naive overlap suppression, and contribution-gated routing.
