"""
Med-Nexus Multimodal Fusion & Calibrated Diagnostic Head
Integrates GIB Conflict-Resilient Fusion (Project 2) and CRPA Attention (Project 4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from models.vision_encoder import MedicalVisionEncoder
from models.text_encoder import ClinicalTextEncoder
from models.crpa_attention import ContributionAwareAttention
from services.conflict_detection import compute_reliability_tensor


class MedNexusDiagnosticModel(nn.Module):
    """
    Master Med-Nexus Multimodal Architecture combining:
    1. Vision Backbone (Chest X-ray feature extractor)
    2. Clinical Text Encoder (RAG evidence encoder)
    3. GIB Conflict Detector & Evidence Reliability Estimator (R_ev)
    4. CRPA Longitudinal Attention Layer (Contribution-Gated temporal tracking)
    5. Calibrated Diagnostic Classification Head
    """

    def __init__(
        self, 
        embedding_dim: int = 256, 
        num_classes: int = 2,
        conflict_threshold: float = 0.45,
        temperature_scaling: float = 1.25,
        sensitivity_threshold: float = 0.015
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.conflict_threshold = conflict_threshold
        self.temperature_scaling = temperature_scaling

        self.vision_encoder = MedicalVisionEncoder(in_channels=3, embedding_dim=embedding_dim)
        self.text_encoder = ClinicalTextEncoder(vocab_size=10000, embedding_dim=embedding_dim)
        self.crpa_attention = ContributionAwareAttention(
            d_model=embedding_dim, 
            num_heads=4, 
            sensitivity_threshold=sensitivity_threshold
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def compute_cross_modal_conflict(
        self, 
        img_feat: torch.Tensor, 
        text_feat: torch.Tensor,
        retrieval_relevance: float = 0.8,
        k_distractors: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes cosine similarity between visual and textual embeddings,
        cross-modal conflict flag, and explicit Evidence Reliability Score R_ev.
        """
        cos_sim = F.cosine_similarity(img_feat, text_feat, dim=-1)
        conflict_flag = (cos_sim < self.conflict_threshold)

        # Explicit R_ev calculation tensor formula
        evidence_reliability = compute_reliability_tensor(
            cos_sim, 
            retrieval_relevance=retrieval_relevance, 
            k_distractors=k_distractors
        )

        return cos_sim, conflict_flag, evidence_reliability

    def forward(
        self, 
        image: torch.Tensor, 
        text_list: List[str], 
        attention_mode: str = "contribution_gated",
        longitudinal_seq: torch.Tensor = None,
        retrieval_relevance: float = 0.8,
        k_distractors: int = 0
    ) -> Dict[str, Any]:
        """
        Forward pass.
        Args:
            image: Image tensor [batch_size, 3, H, W]
            text_list: List of retrieved evidence report strings
            attention_mode: "dense", "sliding_window", "naive_suppression", or "contribution_gated"
            longitudinal_seq: Optional temporal sequence tensor [batch_size, seq_len, embedding_dim]
            retrieval_relevance: Float relevance score from retrieval service
            k_distractors: Number of distractor passages for explicit R_ev calculation
        Returns:
            Dictionary containing prediction logits, probabilities, confidence, conflict flag,
            evidence reliability, and attention efficiency metrics.
        """
        B = image.size(0)
        device = image.device

        # 1. Vision Feature Extraction
        img_feat = self.vision_encoder(image)  # [B, embedding_dim]

        # 2. Text Feature Extraction
        # One text representation is required per image.  For a single image,
        # concatenate all retrieved evidence into one context.  For batched
        # inference, callers must provide exactly one string per image.
        if len(text_list) != B:
            if B == 1:
                text_list = [" \n".join(str(t) for t in text_list)]
            else:
                raise ValueError(f"Expected {B} text contexts, received {len(text_list)}")
        text_feat = self.text_encoder.encode_text(text_list, device=device)

        # 3. GIB Conflict Detection & Explicit Evidence Reliability Estimation (Project 2)
        cos_sim, conflict_flag, evidence_reliability = self.compute_cross_modal_conflict(
            img_feat, text_feat, retrieval_relevance=retrieval_relevance, k_distractors=k_distractors
        )

        # Re-weight text feature based on evidence reliability score R_ev
        weighted_text_feat = text_feat * evidence_reliability.unsqueeze(-1)

        # 4. Longitudinal Temporal Sequence Tracking & CRPA Attention (Project 4)
        if longitudinal_seq is None:
            # Construct sequence: [Vision, Weighted Text]
            seq_tokens = torch.stack([img_feat, weighted_text_feat], dim=1)  # [B, 2, embedding_dim]
        else:
            seq_tokens = longitudinal_seq

        attended_seq, attn_stats = self.crpa_attention(seq_tokens, mode=attention_mode)
        
        # Pooled multimodal context vector
        context_vector = torch.mean(attended_seq, dim=1)  # [B, embedding_dim]
        fused_features = torch.cat([img_feat, context_vector], dim=-1)  # [B, embedding_dim * 2]

        # 5. Diagnostic Classification & Temperature Calibration
        raw_logits = self.classifier(fused_features)
        calibrated_logits = raw_logits / self.temperature_scaling
        probs = F.softmax(calibrated_logits, dim=-1)
        confidence, preds = torch.max(probs, dim=-1)

        return {
            "logits": raw_logits,
            "probabilities": probs,
            "predictions": preds,
            "confidence": confidence,
            "cross_modal_similarity": cos_sim,
            "conflict_detected": conflict_flag,
            "evidence_reliability": evidence_reliability,
            "attention_stats": attn_stats
        }
