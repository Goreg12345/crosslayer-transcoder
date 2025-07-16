"""
PyTorch Dataset and DataLoader for shared memory activation data.
"""

import logging
import multiprocessing as mp
import queue
from typing import Optional

import lightning as L
import torch
from torch.utils.data import Dataset

# DataLoaderConfig no longer needed - using individual parameters
from data.data_generator import DataGeneratorProcess
from data.shared_memory import SharedActivationBuffer

logger = logging.getLogger(__name__)


class SharedMemoryDataset(Dataset):
    """
    PyTorch Dataset that reads activation data from shared memory.
    Works with DataGeneratorProcess to provide continuous data streaming.
    """

    def __init__(self, shared_buffer: SharedActivationBuffer, buffer_size: int):
        self.shared_buffer = shared_buffer
        self.buffer_size = buffer_size

    def __len__(self) -> int:
        """Return the buffer size as dataset length."""
        return self.buffer_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single activation sample. This method is not used directly
        since we override the DataLoader's batch sampling.
        """
        # This shouldn't be called in practice since we use get_batch
        raise NotImplementedError("Use get_batch() instead of individual indexing")

    def get_batch(self, batch_size: int) -> torch.Tensor:
        """
        Get a batch of activation data from shared memory.

        Args:
            batch_size: Number of samples to retrieve

        Returns:
            Tensor of shape [batch_size, n_in_out, n_layers, activation_dim]
        """
        try:
            return self.shared_buffer.get_activations(batch_size)
        except TimeoutError:
            logger.error(f"Timeout waiting for {batch_size} samples")
            raise
        except Exception as e:
            logger.error(f"Error getting batch: {e}")
            raise


def _dataloader_worker(shared_buffer, batch_size, result_queue, stop_event):
    """
    Worker function that runs in a separate process to fetch batches.
    """
    try:
        while not stop_event.is_set():
            try:
                # Get batch from shared memory
                batch = shared_buffer.get_activations(batch_size)

                # Put batch in queue (non-blocking to avoid hanging)
                result_queue.put(batch, timeout=1.0)
            except queue.Full:
                # Queue is full, skip this batch
                continue
            except Exception as e:
                logger.error(f"Error in dataloader worker: {e}")
                break
    except Exception as e:
        logger.error(f"Fatal error in dataloader worker: {e}")


class SharedMemoryDataLoader:
    """
    DataLoader-like interface for streaming activation data from shared memory.
    Uses a pre-started DataGeneratorProcess and provides batches via shared memory.
    Now runs the batch fetching in a separate worker process.
    """

    def __init__(
        self,
        shared_buffer: SharedActivationBuffer,
        dataset: SharedMemoryDataset,
        data_generator: DataGeneratorProcess,
        batch_size: int = 1000,
        use_multiprocessing: bool = True,
        queue_size: int = 2,
        pin_memory: bool = False,
    ):
        self.shared_buffer = shared_buffer
        self.dataset = dataset
        self.generator_process = data_generator
        self.batch_size = batch_size
        self.use_multiprocessing = use_multiprocessing
        self.pin_memory = pin_memory

        # Pre-allocate pinned memory buffer for efficient transfers
        if self.pin_memory:
            # Get the expected shape from shared buffer
            expected_shape = (
                batch_size,
                shared_buffer.n_in_out,
                shared_buffer.n_layers,
                shared_buffer.activation_dim,
            )
            self.pinned_buffer = torch.empty(expected_shape, dtype=shared_buffer.dtype, pin_memory=True)
            logger.info(f"Pre-allocated pinned memory buffer for batch size {batch_size}")
        else:
            self.pinned_buffer = None

        if self.use_multiprocessing:
            # Create multiprocessing components
            ctx = mp.get_context("spawn")  # Use spawn for shared memory compatibility
            self.result_queue = ctx.Queue(maxsize=queue_size)
            self.stop_event = ctx.Event()
            self.worker_process = ctx.Process(
                target=_dataloader_worker,
                args=(
                    self.shared_buffer,
                    self.batch_size,
                    self.result_queue,
                    self.stop_event,
                ),
                daemon=False,
            )
            self.worker_process.start()
            logger.info("Started DataLoader worker process")
        else:
            # Fallback to single-process mode
            self.result_queue = None
            self.stop_event = None
            self.worker_process = None

    def __iter__(self):
        """Iterator interface for PyTorch DataLoader compatibility."""
        return self

    def __next__(self) -> torch.Tensor:
        """Get next batch of data."""
        if self.use_multiprocessing and self.worker_process:
            try:
                # Get batch from worker process queue
                batch = self.result_queue.get(timeout=10.0)

                # If pinned memory requested, copy to pre-allocated pinned buffer (no clone needed)
                if self.pin_memory and batch.device.type == "cpu" and self.pinned_buffer is not None:
                    actual_batch_size = batch.shape[0]
                    if actual_batch_size <= self.batch_size:
                        pinned_slice = self.pinned_buffer[:actual_batch_size]
                        pinned_slice.copy_(batch)
                        return pinned_slice  # Return slice directly, no clone needed

                return batch
            except queue.Empty:
                raise StopIteration("Timeout waiting for batch from worker process")
        else:
            # Fallback to direct access
            batch = self.shared_buffer.get_activations(self.batch_size)
            if self.pin_memory and batch.device.type == "cpu" and self.pinned_buffer is not None:
                # Copy to pre-allocated pinned buffer (no clone needed)
                actual_batch_size = batch.shape[0]
                if actual_batch_size <= self.batch_size:
                    pinned_slice = self.pinned_buffer[:actual_batch_size]
                    pinned_slice.copy_(batch)
                    return pinned_slice  # Return slice directly, no clone needed
            return batch

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return self.shared_buffer.get_stats()

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up SharedMemoryDataLoader...")

        # Stop worker process
        if self.use_multiprocessing and self.worker_process:
            logger.info("Stopping worker process...")
            self.stop_event.set()
            self.worker_process.join(timeout=5.0)

            if self.worker_process.is_alive():
                logger.warning("Force killing worker process...")
                self.worker_process.kill()
                self.worker_process.join()

        if self.generator_process and self.generator_process.is_alive():
            logger.info("Terminating generator process...")
            self.generator_process.terminate()
            self.generator_process.join(timeout=5.0)

            if self.generator_process.is_alive():
                logger.warning("Force killing generator process...")
                self.generator_process.kill()
                self.generator_process.join()

        if self.shared_buffer:
            self.shared_buffer.cleanup()

        # Clean up pinned buffer
        if hasattr(self, "pinned_buffer") and self.pinned_buffer is not None:
            del self.pinned_buffer
            self.pinned_buffer = None

        logger.info("Cleanup complete")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False
