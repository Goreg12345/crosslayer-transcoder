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

    `model_arch` selects the per-layer activation paths:
      gpt2  : transformer.h[i].ln_2.input  +  transformer.h[i].mlp.output
      gemma3: layers[i].pre_feedforward_layernorm.input
              + layers[i].post_feedforward_layernorm.output
              (Gemma3 normalizes the MLP delta before the residual add, so the
              "MLP out" a transcoder should reconstruct is the post-FF-norm output.)
    """

    def __init__(self, n_layers: int, model_arch: str = "gpt2"):
        self.n_layers = n_layers
        if model_arch not in ("gpt2", "gemma3"):
            raise ValueError(f"Unsupported model_arch {model_arch!r}; expected 'gpt2' or 'gemma3'")
        self.model_arch = model_arch
        # Cached layout probe: True iff Gemma3 was loaded as the multimodal class
        # (Gemma3ForConditionalGeneration → has `language_model` submodule). nnsight
        # proxies forward arbitrary attribute access so we can't try/except inside the
        # trace; we have to inspect the underlying HF module up front.
        self._gemma3_multimodal: Optional[bool] = None

    def _detect_gemma3_layout(self, model: Any) -> bool:
        if self._gemma3_multimodal is not None:
            return self._gemma3_multimodal
        underlying = getattr(model, "_model", None) or getattr(model, "local_model", None) or model
        self._gemma3_multimodal = hasattr(underlying, "language_model")
        return self._gemma3_multimodal

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

    def _layer_handles(self, model: Any, i: int):
        if self.model_arch == "gpt2":
            return (
                model.transformer.h[i].ln_2.input,
                model.transformer.h[i].mlp.output,
            )
        # nnsight 0.5 wraps the inner HF model under `model.model` (the top-level
        # Envoy). For Gemma3 we have to go through that envoy explicitly, since
        # `model.language_model` does NOT auto-delegate to an Envoy here (unlike
        # GPT-2's `model.transformer`, which does).
        #
        # Gemma3 layout under the multimodal `Gemma3ForConditionalGeneration`
        # checkpoint (used by google/gemma-3-4b-it):
        #   model.model.language_model.layers[i]   (language_model is Gemma3TextModel,
        #                                           which exposes .layers directly)
        # For a pure-text Gemma3ForCausalLM checkpoint:
        #   model.model.layers[i]
        if self._detect_gemma3_layout(model):
            layer = model.model.language_model.layers[i]
        else:
            layer = model.model.layers[i]
        return layer.pre_feedforward_layernorm.input, layer.post_feedforward_layernorm.output

    @torch.no_grad()
    def _extract_activations(self, model: Any, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Extract MLP input/output activations using nnsight tracing.

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
                mlp_in_proxy, mlp_out_proxy = self._layer_handles(model, i)
                mlp_ins.append(mlp_in_proxy.save())
                mlp_outs.append(mlp_out_proxy.save())

        mlp_ins = torch.stack(mlp_ins, dim=0)
        mlp_outs = torch.stack(mlp_outs, dim=0)

        mlp_acts = einops.rearrange(
            [mlp_ins, mlp_outs], "iO n_layer batch seq d_model -> (batch seq) iO n_layer d_model"
        )
        mask = einops.rearrange(mask, "batch seq -> (batch seq)").bool()
        mlp_acts = mlp_acts[mask]
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
