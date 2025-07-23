"""
Data Loader Package

Shared memory data loader for streaming neural network activations
with efficient multiprocessing data generation.
"""

import torch
import torch.multiprocessing as mp

from data.activation_sources import ActivationComputer, DiskActivationSource
from data.data_generator import DataGeneratorProcess

# Export main classes
from data.datamodule import ActivationDataModule
from data.generation_loop import DataGenerationLoop
from data.process_monitor import ProcessMonitor
from data.shared_memory import SharedActivationBuffer

# Note: Multiprocessing start method is set conditionally:
# - 'spawn' for shared memory mode (needed for PyTorch tensor sharing)
# - 'fork' for simple buffer mode (needed for h5py compatibility)
# This is handled in the respective setup methods.


__all__ = [
    "ActivationDataModule",
    "SharedActivationBuffer",
    "DataGeneratorProcess",
    "ActivationComputer",
    "DiskActivationSource",
    "DataGenerationLoop",
    "ProcessMonitor",
]
