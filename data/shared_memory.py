"""
Shared memory management for large PyTorch activation tensors.
Handles inter-process communication via queues and shared memory buffers.
"""

import atexit
import logging
import multiprocessing as mp
import threading
import time
from multiprocessing import shared_memory
from typing import Any, Dict, List, Tuple

import numpy as np
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
        shared_memory_name: str = "activation_buffer",
        timeout_seconds: int = 30,
        generation_batch_size: int = 32,
        max_sequence_length: int = 1024,
        minimum_fill_threshold: float = 0.0,
    ):
        """
        Initialize shared activation buffer.

        Args:
            buffer_size: Number of activation samples to store
            n_in_out: Number of in/out activations (typically 2)
            n_layers: Number of layers in the model
            activation_dim: Dimension of each activation vector
            dtype: Data type for activations
            shared_memory_name: Name for shared memory buffer
            timeout_seconds: Timeout for buffer operations
            generation_batch_size: Batch size for activation generation (for pinned buffer sizing)
            max_sequence_length: Max sequence length (for pinned buffer sizing)
            minimum_fill_threshold: Minimum buffer fill ratio (0.0-1.0) before providing activations
        """
        self.buffer_size = buffer_size
        self.n_in_out = n_in_out
        self.n_layers = n_layers
        self.activation_dim = activation_dim
        self.dtype = dtype
        self.minimum_fill_threshold = minimum_fill_threshold

        # Calculate memory requirements for 4D tensor [buffer_size, n_in_out, n_layers, activation_dim]
        self.shape = (buffer_size, n_in_out, n_layers, activation_dim)
        self.element_size = torch.tensor([], dtype=dtype).element_size()
        self.total_size = buffer_size * n_in_out * n_layers * activation_dim * self.element_size
        self.n_elems = buffer_size * n_in_out * n_layers * activation_dim

        logger.info(
            f"Creating shared PyTorch buffer: {buffer_size} samples x {n_in_out} in/out x {n_layers} layers x {activation_dim} dims"
        )
        logger.info(f"Total memory: {self.total_size / (1024**3):.2f} GB")

        # Create shared PyTorch tensor directly
        # self.buffer_tensor = torch.empty(self.shape, dtype=dtype, requires_grad=False)
        # Make it shared across processes
        # self.buffer_tensor.share_memory_()
        # Faster variant with proper pickling support:
        self.shm = shared_memory.SharedMemory(create=True, size=self.total_size)
        atexit.register(lambda: self.shm.unlink())
        self.shm_name = self.shm.name  # Store name for pickle/unpickle
        self.buffer_tensor = torch.frombuffer(self.shm.buf, dtype=self.dtype, count=self.n_elems).view(
            self.shape
        )

        # Create shared validity mask tensor
        self.validity_tensor = torch.zeros(buffer_size, dtype=torch.bool, requires_grad=False)
        self.validity_tensor.share_memory_()

        # OPTIMIZATION: Create pinned memory buffer for fast GPU->CPU transfers
        # Size it for the maximum batch we'll process: generation_batch_size * max_sequence_length
        max_batch_samples = generation_batch_size * max_sequence_length
        pinned_shape = (max_batch_samples, n_in_out, n_layers, activation_dim)
        pinned_size_gb = (max_batch_samples * n_in_out * n_layers * activation_dim * self.element_size) / (
            1024**3
        )
        logger.info(f"Creating pinned memory buffer: {max_batch_samples} samples ({pinned_size_gb:.3f} GB)")

        self.pinned_buffer = torch.empty(pinned_shape, dtype=dtype, pin_memory=True)
        self.max_batch_samples = max_batch_samples

        logger.info("Pinned memory buffer created successfully")

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

    def __getstate__(self):
        """
        Custom pickling - only send metadata, not the large tensor.
        This makes multiprocessing.Process.start() return immediately.
        """
        state = self.__dict__.copy()
        # Remove the unpicklable objects - these will be recreated in child process
        del state["buffer_tensor"]
        del state["shm"]  # Don't pickle the buffer itself
        del state["pinned_buffer"]  # Don't pickle pinned memory - recreate in child
        # Keep shm_name so child process can reconnect to same shared memory
        return state

    def __setstate__(self, state):
        """
        Custom unpickling - reconnect to the same shared memory block.
        Child process accesses the identical memory created by parent.
        """
        self.__dict__.update(state)
        # Connect to EXISTING shared memory using the stored name
        self.shm = shared_memory.SharedMemory(name=self.shm_name)
        # Recreate tensor view of the SAME physical memory
        self.buffer_tensor = torch.frombuffer(self.shm.buf, dtype=self.dtype, count=self.n_elems).view(
            self.shape
        )

        # Recreate pinned memory buffer in child process with correct size
        pinned_shape = (
            self.max_batch_samples,
            self.n_in_out,
            self.n_layers,
            self.activation_dim,
        )
        self.pinned_buffer = torch.empty(pinned_shape, dtype=self.dtype, pin_memory=True)

        # Now both parent and child processes access identical memory!

    def get_activations(self, batch_size: int, timeout: float = 1000.0) -> torch.Tensor:
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

                # Calculate current buffer fill percentage
                fill_percentage = len(valid_indices) / self.buffer_size

                # Check if buffer meets minimum fill threshold and has enough samples
                if fill_percentage >= self.minimum_fill_threshold and batch_size <= len(valid_indices):
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
        Set activation data at specific indices using optimized pinned memory transfers.

        Args:
            indices: Tensor of indices to update
            activations: Tensor of activation data
        """
        with self.buffer_lock:
            if len(indices) != len(activations):
                raise ValueError("Number of indices must match number of activations")

            # Detach from computation graph and ensure correct dtype
            activations = activations.detach().to(self.dtype)

            # Optimized GPU->CPU transfer using pinned memory
            batch_size = len(indices)
            if activations.is_cuda:
                if batch_size <= self.max_batch_samples:
                    # Use pre-allocated pinned memory buffer for fast transfer
                    pinned_slice = self.pinned_buffer[:batch_size]
                    pinned_slice.copy_(activations, non_blocking=False)
                    cpu_activations = pinned_slice
                else:
                    # Fallback to regular .cpu() for oversized batches
                    logger.warning(
                        f"Batch size {batch_size} exceeds pinned buffer capacity {self.max_batch_samples}, falling back to .cpu()"
                    )
                    cpu_activations = activations.cpu()  # .cpu() includes synchronization
            else:
                cpu_activations = activations

            # Update buffer directly
            self.buffer_tensor[indices] = cpu_activations

            # Mark indices as valid
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
            invalid_indices = torch.nonzero(~self.validity_tensor, as_tuple=False).squeeze(-1)

        return invalid_indices

    def _get_valid_indices(self) -> torch.Tensor:
        with self.validity_lock:
            valid_indices = torch.nonzero(self.validity_tensor, as_tuple=False).squeeze(-1)
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

        fill_percentage = valid_samples / self.buffer_size
        return {
            "buffer_size": self.buffer_size,
            "n_in_out": self.n_in_out,
            "n_layers": self.n_layers,
            "activation_dim": self.activation_dim,
            "buffer_shape": list(self.shape),
            "total_memory_gb": self.total_size / (1024**3),
            "valid_samples": valid_samples,
            "valid_percentage": fill_percentage * 100,
            "invalid_samples": self.buffer_size - valid_samples,
            "minimum_fill_threshold": self.minimum_fill_threshold,
            "minimum_fill_percentage": self.minimum_fill_threshold * 100,
            "above_minimum_threshold": fill_percentage >= self.minimum_fill_threshold,
            **self.stats,
        }

    def cleanup(self):
        """Clean up shared memory resources."""
        try:
            # Clean up tensors
            if hasattr(self, "buffer_tensor"):
                del self.buffer_tensor
            if hasattr(self, "validity_tensor"):
                del self.validity_tensor
            if hasattr(self, "pinned_buffer"):
                del self.pinned_buffer

            # Clean up shared memory properly
            if hasattr(self, "shm"):
                try:
                    # Close our reference to shared memory
                    self.shm.close()
                    # Try to unlink (delete) the shared memory segment
                    # This will only succeed from the creating process
                    try:
                        self.shm.unlink()
                    except FileNotFoundError:
                        # Already unlinked by another process, that's fine
                        pass
                except Exception as shm_error:
                    logger.warning(f"Error cleaning up shared memory: {shm_error}")

            logger.info("Shared memory resources cleaned up successfully")

        except Exception as e:
            logger.error(f"Error cleaning up shared memory: {e}")

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
        shared_memory_name: str = "activation_buffer",
        timeout_seconds: int = 30,
        generation_batch_size: int = 32,
        max_sequence_length: int = 1024,
        minimum_fill_threshold: float = 0.2,
    ):
        self.buffer_size = buffer_size
        self.n_in_out = n_in_out
        self.n_layers = n_layers
        self.activation_dim = activation_dim
        self.dtype = dtype
        self.shared_memory_name = shared_memory_name
        self.timeout_seconds = timeout_seconds
        self.generation_batch_size = generation_batch_size
        self.max_sequence_length = max_sequence_length
        self.minimum_fill_threshold = minimum_fill_threshold
        self.buffer: SharedActivationBuffer | None = None

    def __enter__(self) -> SharedActivationBuffer:
        self.buffer = SharedActivationBuffer(
            buffer_size=self.buffer_size,
            n_in_out=self.n_in_out,
            n_layers=self.n_layers,
            activation_dim=self.activation_dim,
            dtype=self.dtype,
            shared_memory_name=self.shared_memory_name,
            timeout_seconds=self.timeout_seconds,
            generation_batch_size=self.generation_batch_size,
            max_sequence_length=self.max_sequence_length,
            minimum_fill_threshold=self.minimum_fill_threshold,
        )
        return self.buffer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.buffer:
            self.buffer.cleanup()
