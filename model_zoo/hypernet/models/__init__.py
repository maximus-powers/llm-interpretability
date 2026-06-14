"""
MUAT Hypernetwork Models

This module provides models for generating neural network weights
conditioned on behavioral signatures.
"""

from .functional_hypernetwork import (
    FunctionalHyperNetwork,
    SubjectNetwork,
    BehaviorEditor,
)

__all__ = [
    'FunctionalHyperNetwork',
    'SubjectNetwork', 
    'BehaviorEditor',
]
