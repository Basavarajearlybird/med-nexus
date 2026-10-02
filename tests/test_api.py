"""
Unit and Integration Tests for Med-Nexus FastAPI Service
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Healthy"
    assert "version" in data


def test_analyze_diagnostic_endpoint():
    payload = {
        "query": "Assess lung consolidation",
        "patient_history": "Patient presents with cough and fever.",
        "attention_mode": "contribution_gated",
        "k_distractors": 0,
        "visual_perturbation": "clean"
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "conflict_detected" in data
    assert "attention_stats" in data
    assert "performance_metrics" in data
