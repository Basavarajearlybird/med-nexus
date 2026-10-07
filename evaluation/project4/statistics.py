"""Project 4 evaluation helpers."""
from __future__ import annotations
import numpy as np


def bootstrap_mean_ci(values, n_boot=2000, seed=42, alpha=0.05):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = rng.choice(x, size=len(x), replace=True).mean()
    return {"mean": float(x.mean()), "lower": float(np.quantile(means, alpha/2)), "upper": float(np.quantile(means, 1-alpha/2))}
