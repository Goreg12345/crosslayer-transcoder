"""
PyTorch Dataset and DataLoader for shared memory activation data.
"""

import logging
import time
from typing import Optional

import torch
from torch.utils.data import Dataset

from data_loader.config import DataLoaderConfig
from data_loader.data_generator import DataGeneratorProcess
from data_loader.shared_memory import SharedActivationBuffer

logger = logging.getLogger(__name__)


class SharedMemoryDataset(Dataset):
    """
    PyTorch Dataset that reads activation data from shared memory.
    Works with DataGeneratorProcess to provide continuous data streaming.
    """

    def __init__(self, shared_buffer: SharedActivationBuffer, config: DataLoaderConfig):
        self.shared_buffer = shared_buffer
        self.config = config
        self.timeout = 30.0  # Timeout for waiting for data

    def __len__(self) -> int:
        """Return the buffer size as dataset length."""
        return self.config.buffer_size

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
            return self.shared_buffer.get_activations(batch_size, timeout=self.timeout)
        except TimeoutError:
            logger.error(f"Timeout waiting for {batch_size} samples")
            raise
        except Exception as e:
            logger.error(f"Error getting batch: {e}")
            raise


class SharedMemoryDataLoader:
    """
    DataLoader-like interface for streaming activation data from shared memory.
    Spawns a DataGeneratorProcess and provides batches via shared memory.
    """

    def __init__(
        self, 
        config: Optional[DataLoaderConfig] = None,
        batch_size: int = 1000,
        timeout: float = 30.0
    ):
        self.config = config or DataLoaderConfig()
        self.batch_size = batch_size
        self.timeout = timeout
        
        # Initialize shared memory buffer
        self.shared_buffer = SharedActivationBuffer(
            buffer_size=self.config.buffer_size,
            n_in_out=self.config.n_in_out,
            n_layers=self.config.n_layers,
            activation_dim=self.config.activation_dim,
            dtype=self.config.dtype
        )
        
        # Create dataset
        self.dataset = SharedMemoryDataset(self.shared_buffer, self.config)
        
        # Generator process (started on first iteration)
        self.generator_process: Optional[DataGeneratorProcess] = None
        self.started = False

    def start_generator(self):
        """Start the data generator process."""
        if self.started:
            return
            
        logger.info("Starting data generator process...")
        self.generator_process = DataGeneratorProcess(self.shared_buffer, self.config)
        self.generator_process.start()
        self.started = True
        
        # Wait for initial data generation
        logger.info("Waiting for initial data generation...")
        self._wait_for_data(min_samples=min(1000, self.batch_size))

    def _wait_for_data(self, min_samples: int = 100):
        """Wait for sufficient data to be available."""
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            stats = self.shared_buffer.get_stats()
            valid_samples = stats["valid_samples"]
            
            if valid_samples >= min_samples:
                logger.info(f"Data ready: {valid_samples} samples available")
                return
                
            logger.info(f"Waiting for data... ({valid_samples}/{min_samples} samples)")
            time.sleep(1.0)
            
        raise TimeoutError(f"Timeout waiting for {min_samples} samples")

    def __iter__(self):
        """Iterator interface for PyTorch DataLoader compatibility."""
        if not self.started:
            self.start_generator()
            
        return self

    def __next__(self) -> torch.Tensor:
        """Get next batch of data."""
        if not self.started:
            self.start_generator()
            
        return self.dataset.get_batch(self.batch_size)

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return self.shared_buffer.get_stats()

    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up SharedMemoryDataLoader...")
        
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