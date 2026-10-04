"""Diagnostics of labeled embeddings without dimensionality reduction."""

from .audit import AuditConfig, evaluate_embeddings
from .report import EmbeddingQualityReport

__all__ = ["AuditConfig", "EmbeddingQualityReport", "evaluate_embeddings"]
__version__ = "0.1.0"
