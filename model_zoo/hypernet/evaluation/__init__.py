"""
Evaluation pipeline for FunctionalHyperNetwork.

Provides comprehensive metrics for:
- Latent space analysis (clustering, separability)
- Reconstruction quality (weight-level and behavioral)
- Editing quality (full NxN pattern matrix)

Usage:
    python cli.py experiment hypernet evaluate --model path/to/model.pt
"""

from .pipeline import run_evaluation, EvaluationResults
from .latent_metrics import compute_latent_metrics
from .reconstruction_metrics import compute_reconstruction_metrics
from .editing_metrics import compute_editing_metrics

__all__ = [
    'run_evaluation',
    'EvaluationResults',
    'compute_latent_metrics',
    'compute_reconstruction_metrics',
    'compute_editing_metrics',
]
