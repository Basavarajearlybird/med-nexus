"""
Med-Nexus Master Inference Pipeline Service
Orchestrates end-to-end processing: image preprocessing, evidence retrieval,
GIB conflict detection, CRPA contribution-aware attention, and diagnostic reporting.
"""

import time
from typing import Dict, Any, List, Union
from PIL import Image
from torchvision import transforms
import torch
import numpy as np

from models.multimodal_fusion import MedNexusDiagnosticModel
from services.retrieval import EvidenceRetrievalService
from services.conflict_detection import ConflictDetectionService


class MedNexusInferencePipeline:
    """Master Med-Nexus Inference Pipeline."""

    LABEL_NAMES = {
        0: "Normal / Unremarkable",
        1: "Abnormal Respiratory Finding"
    }

    def __init__(self, device: str = "cpu", sensitivity_threshold: float = 0.015):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model = MedNexusDiagnosticModel(
        sensitivity_threshold=sensitivity_threshold
    ).to(self.device)
        self.model.eval()
        
        self.retrieval_service = EvidenceRetrievalService()
        self.conflict_service = ConflictDetectionService()
        self.image_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    def preprocess_image(self, image):
        """Convert PIL/numpy image to the exact DenseNet preprocessing space."""
        if isinstance(image, np.ndarray): image = Image.fromarray(image).convert("RGB")
        if isinstance(image, torch.Tensor):
            x=image
            if x.ndim==3: x=x.unsqueeze(0)
            return x.to(self.device)
        return self.image_transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def analyze(
        self,
        image: Union[Image.Image, np.ndarray],
        query: str = "Assess respiratory consolidation and pleural status",
        patient_history: str = None,
        attention_mode: str = "contribution_gated",
        k_distractors: int = 0
    ) -> Dict[str, Any]:
        """
        Runs full diagnostic inference pipeline.
        Returns detailed structured diagnostic payload.
        """
        start_time = time.perf_counter()

        # 1. Preprocess image
        img_tensor = self.preprocess_image(image)

        # 2. Retrieve evidence passages and their relevance scores
        retrieved_passages = self.retrieval_service.retrieve_evidence(query, top_k=2, k_distractors=k_distractors)
        
        # We will assume retrieve_evidence returns a list of dicts: {"text": str, "score": float} after we update it.
        # But wait, to keep backward compatibility with scripts, let's have it return just strings, 
        # and we can calculate the retrieval score inside the pipeline directly.
        # Let's extract texts if it was changed, or just keep it as texts.
        
        if patient_history:
            retrieved_passages.insert(0, patient_history)

        # Calculate an actual retrieval relevance score (average Cosine Similarity to query)
        query_emb = self.retrieval_service._encode([query]) if hasattr(self.retrieval_service, '_encode') else None
        retrieval_relevance = 0.0
        if query_emb is not None and retrieved_passages:
            passages_emb = self.retrieval_service._encode(retrieved_passages)
            sims = torch.nn.functional.cosine_similarity(query_emb, passages_emb, dim=-1)
            retrieval_relevance = float(sims.mean().item())

        # 3. Model Forward Pass
        with torch.no_grad():
            outputs = self.model(
                image=img_tensor,
                text_list=retrieved_passages,
                attention_mode=attention_mode,
                k_distractors=k_distractors,
                retrieval_relevance=retrieval_relevance
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Memory tracking
        gpu_memory_mb = 0.0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

        pred_class = outputs["predictions"].item()
        confidence = outputs["confidence"].item()
        cos_sim = outputs["cross_modal_similarity"].item()
        evidence_reliability = outputs["evidence_reliability"].item()

        # 4. Conflict Analysis
        conflict_info = self.conflict_service.calculate_explicit_reliability(cos_sim, retrieval_relevance=retrieval_relevance, k_distractors=k_distractors)

        return {
            "prediction_code": pred_class,
            "prediction": self.LABEL_NAMES.get(pred_class, "Unknown"),
            "confidence": round(confidence, 4),
            "cross_modal_similarity": round(cos_sim, 4),
            "conflict_detected": outputs["conflict_detected"].item(),
            "evidence_reliability": round(evidence_reliability, 4),
            "retrieved_evidence": retrieved_passages,
            "attention_stats": outputs["attention_stats"],
            "risk_analysis": conflict_info,
            "performance_metrics": {
                "latency_ms": round(latency_ms, 2),
                "gpu_memory_mb": round(gpu_memory_mb, 2),
                "device": str(self.device)
            }
        }
