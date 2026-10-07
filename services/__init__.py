"""Med-Nexus Service Layer.

Imports are intentionally kept lightweight to avoid package-level circular imports.
Import concrete services directly, e.g. ``from services.retrieval import ...``.
The inference pipeline is exposed lazily through ``__getattr__`` so importing a
model does not recursively import the pipeline.
"""
from services.retrieval import EvidenceRetrievalService
from services.conflict_detection import ConflictDetectionService

__all__ = [
    "EvidenceRetrievalService",
    "ConflictDetectionService",
    "MedNexusInferencePipeline",
]


def __getattr__(name):
    if name == "MedNexusInferencePipeline":
        from services.inference_pipeline import MedNexusInferencePipeline
        return MedNexusInferencePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
