"""
Activation data sources for the data generator.
Different sources that can provide batches of neural network activations.
"""

import gc
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import einops
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

    def __init__(
        self,
        n_layers: int,
        model_arch: str = "gpt2",
        input_location: str = "pre_norm",
        output_location: str = "post_norm",
        zero_dimensions: Optional[list[int]] = None,
    ):
        self.n_layers = n_layers
        if model_arch not in ("gpt2", "gemma3", "qwen3"):
            raise ValueError(
                f"Unsupported model_arch {model_arch!r}; expected 'gpt2', 'gemma3', or 'qwen3'"
            )
        self.model_arch = model_arch
        if input_location not in ("pre_norm", "post_norm"):
            raise ValueError("input_location must be 'pre_norm' or 'post_norm'")
        if output_location not in ("raw", "post_norm"):
            raise ValueError("output_location must be 'raw' or 'post_norm'")
        if model_arch == "gpt2" and output_location == "post_norm":
            # GPT-2 has no post-MLP norm, so these locations coincide.
            output_location = "raw"
        self.input_location = input_location
        self.output_location = output_location
        self.zero_dimensions = tuple(zero_dimensions or ())
        self._gemma3_multimodal: Optional[bool] = None

    def _detect_gemma3_layout(self, model: Any) -> bool:
        if self._gemma3_multimodal is None:
            underlying = (
                getattr(model, "_model", None)
                or getattr(model, "local_model", None)
                or model
            )
            self._gemma3_multimodal = hasattr(underlying, "language_model")
        return self._gemma3_multimodal

    def _layer_handles(self, model: Any, layer_index: int):
        if self.model_arch == "gpt2":
            layer = model.transformer.h[layer_index]
            mlp_in = layer.ln_2.input if self.input_location == "pre_norm" else layer.ln_2.output
            return mlp_in, layer.mlp.output

        if self.model_arch == "qwen3":
            layer = model.model.layers[layer_index]
            mlp_in = (
                layer.post_attention_layernorm.input
                if self.input_location == "pre_norm"
                else layer.post_attention_layernorm.output
            )
            return mlp_in, layer.mlp.output

        if self._detect_gemma3_layout(model):
            layer = model.model.language_model.layers[layer_index]
        else:
            layer = model.model.layers[layer_index]
        mlp_in = (
            layer.pre_feedforward_layernorm.input
            if self.input_location == "pre_norm"
            else layer.pre_feedforward_layernorm.output
        )
        mlp_out = (
            layer.mlp.output
            if self.output_location == "raw"
            else layer.post_feedforward_layernorm.output
        )
        return mlp_in, mlp_out

    def get_next_batch(self, model: Any, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute activations by running forward pass.

        Args:
            model: nnsight LanguageModel to run inference on
            tokens: Input tokens of shape [batch, seq_len]

        Returns:
            Activations tensor [batch*seq_len, n_in_out, n_layers, d_model]
        """
        gc.collect()
        return self._extract_activations(model, tokens, mask)

    @torch.no_grad()
    def _extract_activations(self, model: Any, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Extract MLP input/output activations using nnsight tracing.
        EXACT COPY of existing method - no changes to functionality!

        Args:
            model: nnsight LanguageModel to run inference on
            tokens: Tokenized input [batch, seq_len]
            mask: Attention mask [batch, seq_len]

        Returns:
            Activations tensor [samples, in/out, n_layer, d_model]
        """
        mlp_ins = []
        mlp_outs = []
        with model.trace(tokens) as tracer:

            for i in range(self.n_layers):
                mlp_in, mlp_out = self._layer_handles(model, i)
                mlp_ins.append(mlp_in.save())
                mlp_outs.append(mlp_out.save())

        mlp_ins = torch.stack(mlp_ins, dim=0)
        mlp_outs = torch.stack(mlp_outs, dim=0)

        mlp_acts = einops.rearrange(
            [mlp_ins, mlp_outs], "iO n_layer batch seq d_model -> (batch seq) iO n_layer d_model"
        )
        mask = einops.rearrange(mask, "batch seq -> (batch seq)").bool()
        mlp_acts = mlp_acts[mask]
        if self.zero_dimensions:
            invalid = [index for index in self.zero_dimensions if not 0 <= index < mlp_acts.shape[-1]]
            if invalid:
                raise ValueError(
                    f"zero_dimensions contains indices outside activation dimension "
                    f"{mlp_acts.shape[-1]}: {invalid}"
                )
            mlp_acts[..., list(self.zero_dimensions)] = 0
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

    def __init__(
        self,
        file_path: str,
        accessor: str = "tensor",
        dtype: torch.dtype = torch.float32,
    ):
        self.file_path = file_path
        self.accessor = accessor
        self.position = 0
        self.file_handle: Optional[h5py.File] = None
        self.tensor_handle: Optional[Any] = None
        self.dtype = dtype

        if self.is_available():
            self._setup_file()

    def _setup_file(self) -> None:
        """Open the HDF5 file and get tensor handle."""
        try:
            fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
            fapl.set_sieve_buf_size(32 * 1024 * 1024)  # 32 MB instead of 128 kB
            fid = h5py.h5f.open(self.file_path.encode(), h5py.h5f.ACC_RDONLY, fapl)
            self.file_handle = h5py.File(fid)
            self.tensor_handle = self.file_handle[self.accessor]

            # self.file_handle = h5py.File(
            #     self.file_path, "r", rdcc_nbytes=1024**3, rdcc_nslots=100003
            # )
            # self.tensor_handle = self.file_handle[self.accessor]
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
        return torch.tensor(data, dtype=self.dtype)

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


if __name__ == "__main__":
    import nnsight
    import torch

    gpt2 = nnsight.LanguageModel("openai-community/gpt2", device_map="cuda:0", dispatch=True)

    gpt2.requires_grad_(False)

    computer = ActivationComputer(gpt2.config.n_layer)
    tokens = torch.randint(0, gpt2.config.vocab_size, (5, 1024))
    actvs = computer.get_next_batch(gpt2, tokens)
    print(actvs.shape)
