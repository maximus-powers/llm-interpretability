"""
MUAT Functional HyperNetwork

This package implements a Conditional VAE that generates neural network weights
conditioned on behavioral signatures. The key insight is using FUNCTIONAL loss
(does the generated network behave correctly?) rather than just weight reconstruction.

Key components:
- models/: FunctionalHyperNetwork, SubjectNetwork, BehaviorEditor
- train.py: Training script with TensorBoard logging
- evaluate.py: Evaluation and inference utilities
- experiments/: Research experiments validating the approach
- components/: Legacy building blocks
- utils/: Data loading helpers

Usage:
    # Train
    python -m hypernet.train --epochs 150 --use-functional-loss
    
    # Evaluate
    python -m hypernet.evaluate --model model.pt evaluate
    
    # Edit behavior
    python -m hypernet.evaluate --model model.pt edit sorted_descending sorted_ascending

Key findings:
- Full loop VERIFIED: Edit behavioral conditioning → decode weights → behavior changes
- 14 behavior patterns supported with test cases
- Reconstruction cosine ~0.83, editing success rate varies by pattern pair
"""

from hypernet.models import (
    FunctionalHyperNetwork,
    SubjectNetwork,
    BehaviorEditor,
)
from hypernet.models.functional_hypernetwork import HyperNetConfig

__version__ = "2.0.0"

__all__ = [
    'FunctionalHyperNetwork',
    'SubjectNetwork',
    'BehaviorEditor',
    'HyperNetConfig',
]
