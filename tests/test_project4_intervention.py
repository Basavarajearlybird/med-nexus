from experiments.project4.intervention_sensitivity import run_intervention

def test_project4_intervention_schema():
    r = run_intervention(seed=42, seq_len=5, d_model=8)
    assert r["experiment"] == "Project4_exact_intervention_edge_sensitivity"
    assert len(r["top_edges_by_delta_loss"]) == 20 or len(r["top_edges_by_delta_loss"]) == 25
    assert "attention_delta_pearson" in r
    assert "definition" in r
