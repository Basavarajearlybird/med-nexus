"""Exact intervention-based attention-edge contribution diagnostic for Project 4.

For a fixed attention layer and task loss L, the marginal contribution of edge (i,j)
is estimated by intervention:
    Delta_ij = L(mask(i,j)) - L(dense)
A positive Delta means removing the edge increases loss, so the edge is useful.
This is a diagnostic experiment, not a claim that the current production CRPA layer
computes all edge sensitivities during every inference pass.
"""
from __future__ import annotations
import argparse, json
from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def attention(q, k, v, mask=None):
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / (d ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    w = torch.softmax(scores, dim=-1)
    return w @ v, w


def edge_intervention_delta(q, k, v, target, edge: Tuple[int, int]):
    """Return exact loss change for removing one attention edge on one-head input.

    q,k,v: [N,D]
    target: [N,D] regression target for attended context
    edge: (query_index, key_index)
    """
    n = q.shape[0]
    base_ctx, _ = attention(q, k, v)
    base_loss = F.mse_loss(base_ctx, target)

    mask = torch.ones((1, n, n), dtype=torch.bool, device=q.device)
    i, j = edge
    mask[0, i, j] = False
    masked_ctx, _ = attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), mask)
    loss = F.mse_loss(masked_ctx.squeeze(0), target)
    return float((loss - base_loss).detach().cpu()), float(base_loss.detach().cpu())


def run_intervention(seed: int = 42, seq_len: int = 12, d_model: int = 32) -> Dict:
    torch.manual_seed(seed)
    q = torch.randn(seq_len, d_model)
    k = torch.randn(seq_len, d_model)
    v = torch.randn(seq_len, d_model)
    target = torch.randn(seq_len, d_model)

    _, dense_w = attention(q, k, v)
    deltas = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            deltas[i, j], _ = edge_intervention_delta(q, k, v, target, (i, j))

    # Rank edges by true loss sensitivity, not attention magnitude.
    flat = deltas.flatten()
    order = torch.argsort(flat, descending=True)
    top = []
    for idx in order[: min(20, flat.numel())]:
        i = int(idx // seq_len); j = int(idx % seq_len)
        top.append({
            "query": i,
            "key": j,
            "delta_loss": float(deltas[i, j]),
            "dense_attention": float(dense_w[i, j]),
        })

    corr = torch.corrcoef(torch.stack([dense_w.flatten(), deltas.flatten()]))[0, 1]
    return {
        "experiment": "Project4_exact_intervention_edge_sensitivity",
        "seed": seed,
        "seq_len": seq_len,
        "d_model": d_model,
        "baseline_loss": float(F.mse_loss(attention(q, k, v)[0], target)),
        "mean_delta_loss": float(deltas.mean()),
        "max_delta_loss": float(deltas.max()),
        "min_delta_loss": float(deltas.min()),
        "attention_delta_pearson": float(corr),
        "top_edges_by_delta_loss": top,
        "definition": "Delta_ij = L(mask(i,j)) - L(dense)",
        "interpretation": "Positive Delta indicates an edge whose removal increases task loss.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--out", default="results/project4_intervention.json")
    args = ap.parse_args()
    result = run_intervention(args.seed, args.seq_len, args.d_model)
    import pathlib
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
