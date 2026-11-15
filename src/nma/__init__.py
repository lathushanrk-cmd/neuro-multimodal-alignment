"""Neuro Multimodal Alignment: explicit, cross-validated representational comparisons."""

from .data import AlignedDataset
from .rsa import rsa_score, representational_dissimilarity
from .cca import CrossValidatedCCA
from .procrustes import CrossValidatedProcrustes
from .simulation import generate_shared_private

__version__ = "0.5.0"
__all__ = [
    "AlignedDataset",
    "rsa_score",
    "representational_dissimilarity",
    "CrossValidatedCCA",
    "CrossValidatedProcrustes",
    "generate_shared_private",
]
