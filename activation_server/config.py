"""
Configuration settings for the activation server.
"""

import os
from dataclasses import dataclass
from typing import Literal, Optional

import torch


@dataclass
class ServerConfig:
    """Configuration for the activation server."""

    # Buffer settings
    buffer_size: int = 10000000  # 10M samples (adjust based on memory)
    n_in_out: int = 2  # Input and output activations for each layer
    n_layers: int = 12  # Number of layers in the model
    activation_dim: int = 768  # Dimension of activation vectors
    dtype: torch.dtype = torch.float32
    max_batch_size: int = 500_000  # Allow larger batches for high-throughput clients

    # Model settings
    model_name: str = "openai-community/gpt2"  # Smaller model for testing
    model_dtype: torch.dtype = torch.float32  # Use float32 for fast CPU inference

    # Dataset settings
    dataset_name: str = "Skylion007/openwebtext"  # or "wikitext" for smaller testing
    dataset_split: str = "train"
    max_sequence_length: int = 1024

    # Generation settings
    generation_batch_size: int = 32  # How many samples to generate at once
    refresh_interval: float = 0.1  # Seconds between checking for refresh requests

    init_file: Optional[str] = None

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create config from environment variables."""
        config = cls()

        # Override with environment variables if present
        if os.getenv("BUFFER_SIZE"):
            config.buffer_size = int(os.getenv("BUFFER_SIZE"))

        if os.getenv("ACTIVATION_DIM"):
            config.activation_dim = int(os.getenv("ACTIVATION_DIM"))

        if os.getenv("MODEL_NAME"):
            config.model_name = os.getenv("MODEL_NAME")

        if os.getenv("DATASET_NAME"):
            config.dataset_name = os.getenv("DATASET_NAME")

        if os.getenv("MAX_BATCH_SIZE"):
            config.max_batch_size = int(os.getenv("MAX_BATCH_SIZE"))

        if os.getenv("HOST"):
            config.host = os.getenv("HOST")

        if os.getenv("PORT"):
            config.port = int(os.getenv("PORT"))

        return config

    def get_memory_estimate_gb(self) -> float:
        """Estimate total memory usage in GB."""
        # Buffer memory: [buffer_size, n_in_out, n_layers, activation_dim]
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        buffer_memory = (
            self.buffer_size
            * self.n_in_out
            * self.n_layers
            * self.activation_dim
            * element_size
        )

        # Validity mask memory
        validity_memory = self.buffer_size  # 1 byte per sample

        # Total shared memory
        shared_memory = buffer_memory + validity_memory

        return shared_memory / (1024**3)

    def validate(self):
        """Validate configuration settings."""
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")

        if self.activation_dim <= 0:
            raise ValueError("activation_dim must be positive")

        if self.max_batch_size <= 0 or self.max_batch_size > self.buffer_size:
            raise ValueError("max_batch_size must be positive and <= buffer_size")

        if self.generation_batch_size <= 0:
            raise ValueError("generation_batch_size must be positive")

        memory_gb = self.get_memory_estimate_gb()
        if memory_gb > 100:  # Warning for very large memory usage
            print(f"Warning: Estimated memory usage is {memory_gb:.2f} GB")


# Default configurations for different use cases


def get_test_config() -> ServerConfig:
    """Get configuration for testing/development."""
    return ServerConfig(
        buffer_size=100_000,  # Small buffer for testing
        n_in_out=2,
        n_layers=12,  # GPT-2 has 12 layers
        activation_dim=768,
        model_name="openai-community/gpt2",  # Use GPT-2 for consistency
        dataset_name="Skylion007/openwebtext",  # Use openwebtext like production
        dataset_split="train",
        generation_batch_size=2,
        init_file="/var/local/glang/activations/clt-activations-1M-shuffled.h5",
    )


def get_production_config() -> ServerConfig:
    """Get configuration for production use."""
    return ServerConfig(
        buffer_size=1_000_000,  # 10M samples
        n_in_out=2,
        n_layers=12,  # GPT-2 has 12 layers
        activation_dim=768,  # GPT-2 dimension
        model_name="openai-community/gpt2",  # Production model
        dataset_name="Skylion007/openwebtext",
        generation_batch_size=2,
    )
