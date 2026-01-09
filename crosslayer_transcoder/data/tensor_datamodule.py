"""
Simple Lightning DataModule for loading pre-computed activations from .pt files.
Designed for toy models and quick experiments without the complexity of shared memory.
"""

import logging

import lightning as L
import torch
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)


class InfiniteTensorDataset(IterableDataset):
    """Iterable dataset that yields batches from a tensor infinitely, with optional shuffling."""

    def __init__(self, tensor: torch.Tensor, batch_size: int, shuffle: bool = True):
        self.tensor = tensor
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        while True:
            # Shuffle indices each epoch
            if self.shuffle:
                indices = torch.randperm(self.tensor.shape[0])
            else:
                indices = torch.arange(self.tensor.shape[0])

            # Yield batches
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                yield self.tensor[batch_indices]


class TensorDataModule(L.LightningDataModule):
    """
    Simple DataModule for loading activations from a .pt tensor file.
    Perfect for toy models and debugging. Repeats infinitely.
    """

    def __init__(
        self,
        tensor_file: str,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        """
        Initialize TensorDataModule.

        Args:
            tensor_file: Path to .pt file containing activation tensor
                         Expected shape: [n_samples, n_in_out, n_layers, activation_dim]
            batch_size: Training batch size
            shuffle: Whether to shuffle data each epoch
            num_workers: Number of DataLoader workers
            pin_memory: Whether to pin memory
        """
        super().__init__()
        self.save_hyperparameters()

        self.tensor_file = tensor_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.dataset = None
        self.train_loader = None

    def setup(self, stage: str = None):
        """Load the tensor and create dataset."""
        if stage == "fit" or stage is None:
            logger.info(f"Loading activations from {self.tensor_file}")
            tensor = torch.load(self.tensor_file, weights_only=True)
            logger.info(f"Loaded tensor with shape: {tensor.shape}")

            # Create infinite dataset that cycles and shuffles
            self.dataset = InfiniteTensorDataset(tensor, self.batch_size, self.shuffle)

            self.train_loader = DataLoader(
                self.dataset,
                batch_size=None,  # Batching handled by dataset
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
            logger.info(
                f"Created infinite DataLoader with batch_size={self.batch_size}"
            )

    def train_dataloader(self):
        """Return training data loader."""
        return self.train_loader

    def val_dataloader(self):
        """Return validation data loader (same as training for toy model)."""
        return self.train_loader

    def teardown(self, stage: str = None):
        """Clean up resources."""
        pass
