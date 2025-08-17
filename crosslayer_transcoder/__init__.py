"""
Crosslayer Transcoder Package

An implementation of Anthropic's crosslayer transcoders for neural network interpretability research.
"""

# Import main classes for easy access
from .data import ActivationDataModule
from .metrics import ReplacementModelAccuracy
from .model import CrossLayerTranscoderModule

# Version info
__version__ = "0.1.0"

__all__ = [
    "ActivationDataModule",
    "CrossLayerTranscoderModule",
    "ReplacementModelAccuracy",
]
