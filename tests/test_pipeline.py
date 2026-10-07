"""
Med-Nexus Unit Tests
"""

import pytest
import torch
from models.multimodal_fusion import MedNexusDiagnosticModel
from services.conflict_detection import compute_reliability_tensor
from services.retrieval import EvidenceRetrievalService

def test_model_initialization():
    model = MedNexusDiagnosticModel(embedding_dim=256)
    assert model is not None
    assert isinstance(model, torch.nn.Module)

def test_conflict_detection_formula():
    cos_sim = torch.tensor([0.9])
    k_distractors = 0
    
    # R_ev = sigmoid(alpha * cos_sim + beta * retrieval_relevance - gamma * k_distractors)
    # alpha=3, beta=2, retrieval=0.8 -> z = 2.7 + 1.6 = 4.3 -> sigmoid(4.3)
    r_ev = compute_reliability_tensor(cos_sim, k_distractors=k_distractors)
    
    assert r_ev.item() > 0.95 # Highly reliable

def test_retrieval_service():
    service = EvidenceRetrievalService()
    results = service.retrieve_evidence("pneumonia", top_k=1, k_distractors=2)
    assert len(results) == 3 # 1 real + 2 distractors
