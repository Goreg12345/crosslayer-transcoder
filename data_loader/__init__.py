"""
Data Loader Package

Shared memory data loader for streaming neural network activations
with efficient multiprocessing data generation.
"""

import torch
import torch.multiprocessing as mp

# Set multiprocessing to use 'spawn' method for better compatibility with PyTorch
# This ensures shared tensors work properly across processes
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

# Export main classes
from data_loader.config import DataLoaderConfig, get_test_config, get_production_config
from data_loader.dataset import SharedMemoryDataLoader, SharedMemoryDataset
from data_loader.shared_memory import SharedActivationBuffer
from data_loader.data_generator import DataGeneratorProcess
from data_loader.factory import (
    actvs_loader_from_config,
    actvs_loader_from_test_config,
    actvs_loader_from_production_config,
)
from data_loader.activation_sources import ActivationComputer, DiskActivationSource
from data_loader.generation_loop import DataGenerationLoop
from data_loader.process_monitor import ProcessMonitor

__all__ = [
    "DataLoaderConfig",
    "get_test_config", 
    "get_production_config",
    "SharedMemoryDataLoader",
    "SharedMemoryDataset",
    "SharedActivationBuffer",
    "DataGeneratorProcess",
    "actvs_loader_from_config",
    "actvs_loader_from_test_config",
    "actvs_loader_from_production_config",
    "ActivationComputer",
    "DiskActivationSource", 
    "DataGenerationLoop",
    "ProcessMonitor",
]
