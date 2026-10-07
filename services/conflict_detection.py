"""
Med-Nexus GIB Conflict Detection & Evidence Reliability Engine (Project 2)
Computes mathematically explicit Evidence Reliability Scores (R_ev) and evaluates cross-modal alignment.
"""

import numpy as np
import torch
from typing import Dict, Any


class ConflictDetectionService:
    """
    Rigorously computes Evidence Reliability Score R_ev ∈ [0, 1] based on:
    1. Cross-Modal Agreement (S_cross_modal): Cosine similarity between visual and textual embeddings.
    2. Retrieval Relevance (S_retrieval): Semantic alignment of top-k retrieved evidence.
    3. Retrieval Noise Penalty (P_distractor): Distractor passage penalty based on count k.
    
    Formula:
        R_ev = sigmoid( alpha * S_cross_modal + beta * S_retrieval - gamma * k_distractors )
    """

    def __init__(self, alpha: float = 3.0, beta: float = 2.0, gamma: float = 0.25, conflict_threshold: float = 0.45):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.conflict_threshold = conflict_threshold

    def calculate_explicit_reliability(
        self, 
        cross_modal_sim: float, 
        retrieval_relevance: float = 0.8, 
        k_distractors: int = 0
    ) -> Dict[str, Any]:
        """
        Computes explicit Evidence Reliability Score R_ev.
        """
        # Sigmoid linear combination
        z = self.alpha * cross_modal_sim + self.beta * retrieval_relevance - self.gamma * k_distractors
        r_ev = float(1.0 / (1.0 + np.exp(-z)))

        conflict_detected = (cross_modal_sim < self.conflict_threshold)

        if conflict_detected:
            risk_level = "HIGH_CROSS_MODAL_CONFLICT"
            recommendation = (
                f"Cross-modal conflict detected (Sim={cross_modal_sim:.3f} < {self.conflict_threshold}). "
                f"Evidence Reliability R_ev={r_ev:.3f}. Attenuating textual reliance and relying on visual imaging backbone."
            )
        elif r_ev < 0.4:
            risk_level = "UNRELIABLE_RETRIEVAL"
            recommendation = f"Retrieved text contains significant noise (k={k_distractors}). R_ev={r_ev:.3f}. Calibrating confidence."
        else:
            risk_level = "HIGH_RELIABILITY_AGREEMENT"
            recommendation = f"Strong visual and textual agreement verified. R_ev={r_ev:.3f}."

        return {
            "conflict_detected": conflict_detected,
            "evidence_reliability_score": round(r_ev, 4),
            "cross_modal_similarity": round(cross_modal_sim, 4),
            "retrieval_relevance": round(retrieval_relevance, 4),
            "distractor_count_k": k_distractors,
            "risk_level": risk_level,
            "clinical_recommendation": recommendation
        }


def compute_reliability_tensor(
    cos_sim: torch.Tensor, 
    retrieval_relevance: float = 0.8,
    k_distractors: int = 0, 
    alpha: float = 3.0, 
    beta: float = 2.0, 
    gamma: float = 0.25
) -> torch.Tensor:
    """PyTorch tensor formulation for GPU/batched execution."""
    z = alpha * cos_sim + beta * retrieval_relevance - gamma * k_distractors
    return torch.sigmoid(z)
