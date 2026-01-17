"""
Data Loader Package

Shared memory data loader for streaming neural network activations
with efficient multiprocessing data generation.
"""

import torch
import torch.multiprocessing as mp

from .activation_sources import ActivationComputer, DiskActivationSource
from .data_generator import DataGeneratorProcess

# Export main classes
from .datamodule import ActivationDataModule
from .tensor_datamodule import TensorDataModule
from .deployment_policy import (
    BaseDeploymentPolicy,
    CPUOnlyPolicy,
    DeploymentPolicy,
    DynamicPolicy,
    GPUOnlyPolicy,
    create_deployment_policy,
)
from .generation_loop import DataGenerationLoop
from .process_monitor import ProcessMonitor
from .shared_memory import SharedActivationBuffer
from .tensor_datamodule import TensorDataModule

# Note: Multiprocessing start method is set conditionally:
# - 'spawn' for shared memory mode (needed for PyTorch tensor sharing)
# - 'fork' for simple buffer mode (needed for h5py compatibility)
# This is handled in the respective setup methods.


__all__ = [
    "ActivationDataModule",
    "TensorDataModule",
    "SharedActivationBuffer",
    "DataGeneratorProcess",
    "ActivationComputer",
    "DiskActivationSource",
    "DataGenerationLoop",
    "ProcessMonitor",
    "DeploymentPolicy",
    "BaseDeploymentPolicy",
    "CPUOnlyPolicy",
    "GPUOnlyPolicy",
    "DynamicPolicy",
    "create_deployment_policy",
    "TensorDataModule",
]
