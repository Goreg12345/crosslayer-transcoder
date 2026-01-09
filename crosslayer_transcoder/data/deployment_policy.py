"""
Deployment policy configuration for GPT-2 model placement.
Allows users to control where the GPT-2 model runs during activation generation.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class DeploymentPolicy(Enum):
    """
    Policy for deploying GPT-2 during activation generation.

    CPU_ONLY: Model runs only on CPU
    GPU_ONLY: Model runs only on GPU (requires CUDA)
    DYNAMIC: Model switches between CPU/GPU based on buffer load (default behavior)
    """

    CPU_ONLY = "cpu_only"
    GPU_ONLY = "gpu_only"
    DYNAMIC = "dynamic"

    @classmethod
    def from_string(cls, policy_str: str) -> "DeploymentPolicy":
        """Create policy from string, case-insensitive."""
        policy_str = policy_str.lower().strip()
        for policy in cls:
            if policy.value == policy_str:
                return policy
        raise ValueError(f"Unknown deployment policy: {policy_str}. Valid options: {[p.value for p in cls]}")

    def __str__(self) -> str:
        return self.value


class BaseDeploymentPolicy(ABC):
    """
    Abstract base class for deployment policy implementations.
    Each policy handles model instantiation, device management, and selection logic.
    """

    def __init__(self, device_map: str = "auto"):
        """
        Initialize deployment policy.

        Args:
            device_map: Device specification (e.g., "cpu", "auto", "cuda:0", "cuda:0,1,2,3")
        """
        self.device_map = device_map
        self.cpu_model: Optional[Any] = None
        self.gpu_model: Optional[Any] = None
        self.current_model: Optional[Any] = None
        self.current_device: str = "cpu"

    @abstractmethod
    def setup_models(self, model_name: str, model_dtype: torch.dtype, **kwargs) -> None:
        """
        Setup CPU and/or GPU models based on policy.

        Args:
            model_name: HuggingFace model name
            model_dtype: Model data type
            **kwargs: Additional model arguments
        """
        pass

    @abstractmethod
    def select_device(self, buffer_stats: Optional[Dict] = None) -> str:
        """
        Select device based on policy and buffer statistics.

        Args:
            buffer_stats: Current buffer statistics (for dynamic policies)

        Returns:
            Selected device string ("cpu" or "cuda" or specific GPU)
        """
        pass

    @abstractmethod
    def get_current_model(self) -> Any:
        """Get the currently active model."""
        pass

    def get_current_device(self) -> str:
        """Get the current device string."""
        return self.current_device

    def cleanup(self) -> None:
        """Clean up models and free GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _parse_device_map(self) -> Tuple[List[str], bool]:
        """
        Parse device_map string into list of devices and multi-GPU flag.

        Returns:
            Tuple of (device_list, is_multi_gpu)
        """
        if self.device_map == "auto":
            # Use all available GPUs or fallback to CPU
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                return [f"cuda:{i}" for i in range(device_count)], device_count > 1
            else:
                return ["cpu"], False
        elif self.device_map == "cpu":
            return ["cpu"], False
        elif "," in self.device_map:
            # Multi-GPU specification like "cuda:0,1,2,3"
            devices = [d.strip() for d in self.device_map.split(",")]
            # Handle both "cuda:0,1,2" and "cuda:0,cuda:1,cuda:2" formats
            normalized_devices = []
            for device in devices:
                if device.isdigit():
                    normalized_devices.append(f"cuda:{device}")
                else:
                    normalized_devices.append(device)
            return normalized_devices, len(normalized_devices) > 1
        else:
            # Single device specification like "cuda:0"
            return [self.device_map], False


class CPUOnlyPolicy(BaseDeploymentPolicy):
    """Policy that keeps models on CPU only."""

    def setup_models(self, model_name: str, model_dtype: torch.dtype, **kwargs) -> None:
        """Setup CPU model only."""
        import nnsight

        logger.info(f"Loading CPU model: {model_name}")
        self.cpu_model = nnsight.LanguageModel(
            model_name, device_map="cpu", torch_dtype=model_dtype, dispatch=True, **kwargs
        )
        self.cpu_model.requires_grad_(False)

        self.current_model = self.cpu_model
        self.current_device = "cpu"
        logger.info("CPU-only policy: Model loaded on CPU")

    def select_device(self, buffer_stats: Optional[Dict] = None) -> str:
        """Always return CPU."""
        return "cpu"

    def get_current_model(self) -> Any:
        """Return CPU model."""
        return self.cpu_model


class GPUOnlyPolicy(BaseDeploymentPolicy):
    """Policy that keeps models on GPU only."""

    def setup_models(self, model_name: str, model_dtype: torch.dtype, **kwargs) -> None:
        """Setup GPU model only."""
        import nnsight

        if not torch.cuda.is_available():
            raise RuntimeError("GPU-only policy requires CUDA, but CUDA is not available")

        devices, is_multi_gpu = self._parse_device_map()
        gpu_device_map = devices[0] if not is_multi_gpu else self.device_map

        logger.info(f"Loading GPU model: {model_name} on {gpu_device_map}")
        self.gpu_model = nnsight.LanguageModel(
            model_name, device_map=gpu_device_map, torch_dtype=model_dtype, dispatch=True, **kwargs
        )
        self.gpu_model.requires_grad_(False)

        self.current_model = self.gpu_model
        self.current_device = gpu_device_map  # Use the actual device where model was loaded
        logger.info(f"GPU-only policy: Model loaded on {self.current_device}")

    def select_device(self, buffer_stats: Optional[Dict] = None) -> str:
        """Always return GPU device."""
        return self.current_device

    def get_current_model(self) -> Any:
        """Return GPU model."""
        return self.gpu_model


class DynamicPolicy(BaseDeploymentPolicy):
    """Policy that switches between CPU and GPU based on buffer load."""

    def __init__(self, device_map: str = "auto", cpu_threshold: float = 80.0, gpu_threshold: float = 50.0):
        """
        Initialize dynamic policy.

        Args:
            device_map: Device specification
            cpu_threshold: Switch to CPU when buffer fill > this percentage
            gpu_threshold: Switch to GPU when buffer fill < this percentage
        """
        super().__init__(device_map)
        self.cpu_threshold = cpu_threshold
        self.gpu_threshold = gpu_threshold

    def setup_models(self, model_name: str, model_dtype: torch.dtype, **kwargs) -> None:
        """Setup both CPU and GPU models."""
        import nnsight

        # Setup CPU model
        logger.info(f"Loading CPU model: {model_name}")
        self.cpu_model = nnsight.LanguageModel(
            model_name, device_map="cpu", torch_dtype=model_dtype, dispatch=True, **kwargs
        )
        self.cpu_model.requires_grad_(False)

        # Setup GPU model if available
        if torch.cuda.is_available():
            devices, is_multi_gpu = self._parse_device_map()
            gpu_device_map = devices[0] if not is_multi_gpu else self.device_map

            logger.info(f"Loading GPU model: {model_name} on {gpu_device_map}")
            self.gpu_model = nnsight.LanguageModel(
                model_name, device_map=gpu_device_map, torch_dtype=model_dtype, dispatch=True, **kwargs
            )
            self.gpu_model.requires_grad_(False)
            # Store the actual GPU device for later switching
            self.gpu_device = gpu_device_map
        else:
            logger.warning("CUDA not available, dynamic policy will use CPU only")
            self.gpu_model = None
            self.gpu_device = None

        # Start with CPU
        self.current_model = self.cpu_model
        self.current_device = "cpu"
        logger.info("Dynamic policy: Models loaded, starting with CPU")

    def select_device(self, buffer_stats: Optional[Dict] = None) -> str:
        """Select device based on buffer fill level."""
        if buffer_stats is None or self.gpu_model is None:
            return "cpu"

        valid_percentage = buffer_stats.get("valid_percentage", 100.0)

        # Switch to GPU when buffer is low (need faster generation)
        if valid_percentage < self.gpu_threshold and self.current_device == "cpu":
            self.current_device = self.gpu_device
            self.current_model = self.gpu_model
            torch.cuda.empty_cache()
            return self.gpu_device

        # Switch to CPU when buffer is high (save GPU resources)
        elif valid_percentage > self.cpu_threshold and self.current_device != "cpu":
            self.current_device = "cpu"
            self.current_model = self.cpu_model
            torch.cuda.empty_cache()
            return "cpu"

        return self.current_device

    def get_current_model(self) -> Any:
        """Return currently active model."""
        return self.current_model


def create_deployment_policy(
    policy_type: DeploymentPolicy, device_map: str = "auto", **kwargs
) -> BaseDeploymentPolicy:
    """
    Factory function to create deployment policy instances.

    Args:
        policy_type: Type of deployment policy
        device_map: Device specification
        **kwargs: Additional policy-specific arguments

    Returns:
        Deployment policy instance
    """
    if policy_type == DeploymentPolicy.CPU_ONLY:
        return CPUOnlyPolicy(device_map=device_map, **kwargs)
    elif policy_type == DeploymentPolicy.GPU_ONLY:
        return GPUOnlyPolicy(device_map=device_map, **kwargs)
    elif policy_type == DeploymentPolicy.DYNAMIC:
        return DynamicPolicy(device_map=device_map, **kwargs)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
