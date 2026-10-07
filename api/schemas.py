"""
Pydantic Schemas for Med-Nexus FastAPI REST API
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class DiagnosticRequest(BaseModel):
    query: str = Field(default="Assess chest radiograph for opacity and pleural changes", description="Clinical query string")
    patient_history: Optional[str] = Field(default="Patient presents with acute cough and fever for 3 days.", description="Patient clinical history")
    attention_mode: str = Field(default="contribution_gated", description="Attention mode: dense, sliding_window, naive_suppression, or contribution_gated")
    k_distractors: int = Field(default=0, ge=0, le=20, description="Number of distractor passages to append for noise testing")
    visual_perturbation: str = Field(default="clean", description="Visual perturbation: clean, gaussian_blur, motion_blur, or crop_degrade")


class AttentionStatsSchema(BaseModel):
    mode: str
    total_edges: int
    active_edges: int
    suppressed_edges: int
    retention_ratio: float
    sparsity_ratio: float


class RiskAnalysisSchema(BaseModel):
    conflict_detected: bool
    evidence_reliability_score: float
    risk_level: str
    clinical_recommendation: str


class PerformanceMetricsSchema(BaseModel):
    latency_ms: float
    gpu_memory_mb: float
    device: str


class DiagnosticResponse(BaseModel):
    prediction_code: int
    prediction: str
    confidence: float
    cross_modal_similarity: float
    conflict_detected: bool
    evidence_reliability: float
    retrieved_evidence: List[str]
    attention_stats: AttentionStatsSchema
    risk_analysis: RiskAnalysisSchema
    performance_metrics: PerformanceMetricsSchema


class HealthCheckResponse(BaseModel):
    status: str
    system: str
    version: str
    cuda_available: bool
    device: str
