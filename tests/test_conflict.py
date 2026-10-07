"""
Unit Tests for GIB Conflict Detection Engine (Project 2)
"""

from services.conflict_detection import ConflictDetectionService


def test_normal_agreement():
    service = ConflictDetectionService(conflict_threshold=0.45)
    result = service.calculate_explicit_reliability(cross_modal_sim=0.85, k_distractors=0)
    assert result["conflict_detected"] is False
    assert result["evidence_reliability_score"] > 0.8
    assert result["risk_level"] == "HIGH_RELIABILITY_AGREEMENT"


def test_cross_modal_conflict():
    service = ConflictDetectionService(conflict_threshold=0.45)
    result = service.calculate_explicit_reliability(cross_modal_sim=0.25, k_distractors=0)
    assert result["conflict_detected"] is True
    assert result["risk_level"] == "HIGH_CROSS_MODAL_CONFLICT"


def test_retrieval_distractor_penalty():
    service = ConflictDetectionService(conflict_threshold=0.45)
    res_clean = service.calculate_explicit_reliability(cross_modal_sim=0.80, k_distractors=0)
    res_noisy = service.calculate_explicit_reliability(cross_modal_sim=0.80, k_distractors=5)
    assert res_noisy["evidence_reliability_score"] < res_clean["evidence_reliability_score"]

