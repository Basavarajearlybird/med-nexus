"""
Unit Tests for CRPA Contribution-Aware Attention Layer (Project 4)
"""

import torch
from models.crpa_attention import ContributionAwareAttention


def test_crpa_attention_forward_modes():
    layer = ContributionAwareAttention(d_model=256, num_heads=4, sensitivity_threshold=0.015)
    x = torch.randn(2, 10, 256)

    for mode in ["dense", "sliding_window", "naive_suppression", "contribution_gated"]:
        output, stats = layer(x, mode=mode)
        assert output.shape == (2, 10, 256)
        assert "retention_ratio" in stats
        assert 0.0 <= stats["retention_ratio"] <= 1.0


def test_crpa_edge_gating_sparsity():
    layer = ContributionAwareAttention(d_model=256, num_heads=4, sensitivity_threshold=0.05)
    x = torch.randn(1, 16, 256)
    
    _, stats_dense = layer(x, mode="dense")
    _, stats_gated = layer(x, mode="contribution_gated")
    
    assert stats_dense["retention_ratio"] == 1.0
    assert stats_gated["sparsity_ratio"] >= 0.0
