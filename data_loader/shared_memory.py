"""
Shared memory management for large PyTorch activation tensors.
Handles inter-process communication via queues and shared memory buffers.
"""

import logging
import multiprocessing as mp
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as torch_mp

# No config imports needed in this module

logger = logging.getLogger(__name__)


class SharedActivationBuffer:
    """
    Manages a large shared memory buffer for storing activation tensors.
    Provides thread-safe access for reading/writing activations across processes.
    """

    def __init__(
        self,
        buffer_size: int,
        n_in_out: int,
        n_layers: int,
        activation_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize shared activation buffer.

        Args:
            buffer_size: Number of activation samples to store
            n_in_out: Number of in/out activations (typically 2)
            n_layers: Number of layers in the model
            activation_dim: Dimension of each activation vector
            dtype: Data type for activations
        """
        self.buffer_size = buffer_size
        self.n_in_out = n_in_out
        self.n_layers = n_layers
        self.activation_dim = activation_dim
        self.dtype = dtype

        # Calculate memory requirements for 4D tensor [buffer_size, n_in_out, n_layers, activation_dim]
        self.shape = (buffer_size, n_in_out, n_layers, activation_dim)
        self.element_size = torch.tensor([], dtype=dtype).element_size()
        self.total_size = (
            buffer_size * n_in_out * n_layers * activation_dim * self.element_size
        )

        logger.info(
            f"Creating shared PyTorch buffer: {buffer_size} samples x {n_in_out} in/out x {n_layers} layers x {activation_dim} dims"
        )
        logger.info(f"Total memory: {self.total_size / (1024**3):.2f} GB")

        # Create shared PyTorch tensor directly
        self.buffer_tensor = torch.empty(self.shape, dtype=dtype, requires_grad=False)
        # Make it shared across processes
        self.buffer_tensor.share_memory_()

        # Create shared validity mask tensor
        self.validity_tensor = torch.zeros(
            buffer_size, dtype=torch.bool, requires_grad=False
        )
        self.validity_tensor.share_memory_()

        # Queue for statistics updates (if needed in future)
        self.stats_queue = mp.Queue(maxsize=100)  # Statistics updates

        # Multiprocessing-safe locks
        self.buffer_lock = mp.RLock()
        self.validity_lock = mp.RLock()

        # Statistics
        self.stats = {
            "total_reads": 0,
            "total_writes": 0,
            "last_read_time": None,
            "last_write_time": None,
        }

        logger.info("Shared PyTorch activation buffer initialized successfully")

    def get_activations(self, batch_size: int, timeout: float = 10.0) -> torch.Tensor:
        """
        Get activation samples from the buffer and mark them as invalid.

        Args:
            batch_size: Number of samples to retrieve

        Returns:
            Activations tensor
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        start_time = time.time()

        while True:
            with self.buffer_lock:
                # Find valid indices using PyTorch
                valid_indices = self._get_valid_indices()

                # If we have enough samples, proceed
                if batch_size <= len(valid_indices):
                    # Sample indices randomly
                    perm = torch.randperm(len(valid_indices))[:batch_size]
                    selected_indices = valid_indices[perm]

                    # Get data from buffer (creates a copy)
                    activations = self.buffer_tensor[selected_indices].clone()

                    # Mark selected indices as invalid (they need refresh)
                    self._mark_indices_invalid(selected_indices)

                    # Update stats
                    self.stats["total_reads"] += 1
                    self.stats["last_read_time"] = time.time()

                    return activations

            # Check timeout outside the lock
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout waiting for valid indices")

            # Sleep briefly to allow other processes to fill the buffer
            time.sleep(0.01)

    def set_activations(self, indices: torch.Tensor, activations: torch.Tensor):
        """
        Set activation data at specific indices.

        Args:
            indices: Tensor of indices to update
            activations: Tensor of activation data
        """
        with self.buffer_lock:
            if len(indices) != len(activations):
                raise ValueError("Number of indices must match number of activations")

            # Ensure activations are on CPU and correct dtype
            activations = activations.detach().cpu().to(self.dtype)

            # Update buffer directly
            self.buffer_tensor[indices] = activations

            # Mark as valid
            with self.validity_lock:
                self.validity_tensor[indices] = True

            # Update stats
            self.stats["total_writes"] += 1
            self.stats["last_write_time"] = time.time()

    def _mark_indices_invalid(self, indices: torch.Tensor):
        """
        Mark indices as invalid (needing refresh). Private method.

        Args:
            indices: Tensor of indices that need new data
        """
        with self.validity_lock:
            self.validity_tensor[indices] = False

    def _get_invalid_indices(self) -> torch.Tensor:
        with self.validity_lock:
            invalid_indices = torch.nonzero(
                ~self.validity_tensor, as_tuple=False
            ).squeeze(-1)

        return invalid_indices

    def _get_valid_indices(self) -> torch.Tensor:
        with self.validity_lock:
            valid_indices = torch.nonzero(self.validity_tensor, as_tuple=False).squeeze(
                -1
            )
        return valid_indices

    def force_refresh(self) -> int:
        """
        Force refresh of all invalid indices by marking all as invalid.

        Returns:
            Number of indices marked for refresh
        """
        with self.validity_lock:
            # Mark all as invalid
            self.validity_tensor.fill_(False)

        return self.buffer_size

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self.validity_lock:
            valid_samples = int(torch.sum(self.validity_tensor).item())
        return {
            "buffer_size": self.buffer_size,
            "n_in_out": self.n_in_out,
            "n_layers": self.n_layers,
            "activation_dim": self.activation_dim,
            "buffer_shape": list(self.shape),
            "total_memory_gb": self.total_size / (1024**3),
            "valid_samples": valid_samples,
            "valid_percentage": valid_samples / self.buffer_size * 100,
            "invalid_samples": self.buffer_size - valid_samples,
            **self.stats,
        }

    def cleanup(self):
        """Clean up shared memory resources."""
        try:
            # PyTorch shared tensors are automatically cleaned up by the garbage collector
            # when all references are removed. No manual cleanup needed.
            if hasattr(self, "buffer_tensor"):
                del self.buffer_tensor
            if hasattr(self, "validity_tensor"):
                del self.validity_tensor

            logger.info("Shared PyTorch tensors cleaned up successfully")

        except Exception as e:
            logger.error(f"Error cleaning up shared tensors: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


class SharedMemoryManager:
    """
    Context manager for shared memory resources.
    """

    def __init__(
        self,
        buffer_size: int,
        n_in_out: int,
        n_layers: int,
        activation_dim: int,
        dtype: torch.dtype = torch.float32,
    ):
        self.buffer_size = buffer_size
        self.n_in_out = n_in_out
        self.n_layers = n_layers
        self.activation_dim = activation_dim
        self.dtype = dtype
        self.buffer: Optional[SharedActivationBuffer] = None

    def __enter__(self) -> SharedActivationBuffer:
        self.buffer = SharedActivationBuffer(
            buffer_size=self.buffer_size,
            n_in_out=self.n_in_out,
            n_layers=self.n_layers,
            activation_dim=self.activation_dim,
            dtype=self.dtype,
        )
        return self.buffer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.buffer:
            self.buffer.cleanup()
