"""
Activation data sources for the data generator.
Different sources that can provide batches of neural network activations.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import h5py
import torch


class ActivationSource(ABC):
    """
    Abstract base class for sources of activation data.
    All sources produce batches of activations with shape [batch*seq_len, n_in_out, n_layers, d_model].
    """

    @abstractmethod
    def get_next_batch(self, **kwargs) -> torch.Tensor:
        """
        Get the next batch of activations.

        Returns:
            Tensor of shape [batch*seq_len, n_in_out, n_layers, d_model]
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up any resources (files, etc.)."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this source can provide data."""
        pass


class ActivationComputer(ActivationSource):
    """
    Computes activations by running a forward pass through a language model.
    Pure computation - takes model + tokens, returns activations.
    """

    def __init__(self, config):
        self.config = config

    def get_next_batch(self, model: Any, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute activations by running forward pass.

        Args:
            model: nnsight LanguageModel to run inference on
            tokens: Input tokens of shape [batch, seq_len]

        Returns:
            Activations tensor [batch*seq_len, n_in_out, n_layers, d_model]
        """
        return self._extract_activations(model, tokens)

    @torch.no_grad()
    def _extract_activations(self, model: Any, tokens: torch.Tensor) -> torch.Tensor:
        """
        Extract MLP input/output activations using nnsight tracing.
        EXACT COPY of existing method - no changes to functionality!

        Args:
            model: nnsight LanguageModel to run inference on
            tokens: Tokenized input [batch, seq_len]

        Returns:
            Activations tensor [batch*seq_len, in/out, n_layer, d_model]
        """
        with model.trace(tokens) as tracer:
            mlp_ins = []
            mlp_outs = []

            # Extract from all transformer layers
            for i in range(self.config.n_layers):
                # MLP input (after layer norm)
                mlp_in = model.transformer.h[i].ln_2.input.save()
                mlp_ins.append(mlp_in)

                # MLP output
                mlp_out = model.transformer.h[i].mlp.output.save()
                mlp_outs.append(mlp_out)

        # Stack: [batch, seq_len, n_layer, d_model]
        mlp_ins = torch.stack(mlp_ins, dim=2)
        mlp_outs = torch.stack(mlp_outs, dim=2)

        # Combine input/output: [batch, seq_len, in/out, n_layer, d_model]
        mlp_acts = torch.stack([mlp_ins, mlp_outs], dim=2)

        # Fuse batch and sequence length dimensions: [batch*seq_len, in/out, n_layer, d_model]
        mlp_acts = mlp_acts.reshape(-1, *mlp_acts.shape[2:])

        return mlp_acts

    def is_available(self) -> bool:
        """Computer is always available."""
        return True

    def close(self) -> None:
        """No resources to clean up."""
        pass


class DiskActivationSource(ActivationSource):
    """
    Reads pre-computed activations from an HDF5 file.
    Sequential access through the file.
    """

    def __init__(self, file_path: str, accessor: str = "tensor"):
        self.file_path = file_path
        self.accessor = accessor
        self.position = 0
        self.file_handle: Optional[h5py.File] = None
        self.tensor_handle: Optional[Any] = None

        if self.is_available():
            self._setup_file()

    def _setup_file(self) -> None:
        """Open the HDF5 file and get tensor handle."""
        try:
            self.file_handle = h5py.File(self.file_path, "r")
            self.tensor_handle = self.file_handle[self.accessor]
            self.position = 0
        except Exception as e:
            raise RuntimeError(f"Failed to open activation file {self.file_path}: {e}")

    def get_next_batch(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Read next batch of activations from file.

        Args:
            batch_size: Number of samples to read (uses remaining if None)

        Returns:
            Activations tensor [batch*seq_len, n_in_out, n_layers, d_model]
        """
        if not self.is_available():
            raise RuntimeError("Disk source not available")

        if self.tensor_handle is None:
            raise RuntimeError("File not properly initialized")

        # Determine batch size
        total_samples = self.tensor_handle.shape[0]
        remaining = total_samples - self.position

        if batch_size is None:
            batch_size = remaining
        else:
            batch_size = min(batch_size, remaining)

        if batch_size <= 0:
            raise RuntimeError("No more data available in file")

        # Read batch
        end_pos = self.position + batch_size
        data = self.tensor_handle[self.position : end_pos]
        self.position = end_pos

        # Convert to tensor and return
        return torch.tensor(data, dtype=torch.float32)

    def reset_position(self) -> None:
        """Reset file position to beginning."""
        self.position = 0

    def is_available(self) -> bool:
        """Check if file exists and is readable."""
        return os.path.exists(self.file_path) and os.path.isfile(self.file_path)

    def get_remaining_samples(self) -> int:
        """Get number of samples remaining in file."""
        if not self.is_available() or self.tensor_handle is None:
            return 0
        return max(0, self.tensor_handle.shape[0] - self.position)

    def close(self) -> None:
        """Close the HDF5 file."""
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None
            self.tensor_handle = None
