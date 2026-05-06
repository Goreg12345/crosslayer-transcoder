"""
Lightning DataModule for activation data loading.
Replaces config.py and factory.py with Lightning CLI-compatible structure.
"""

import logging
import os
from typing import Optional

import lightning as L
import torch

from crosslayer_transcoder.data.data_generator import DataGeneratorProcess
from crosslayer_transcoder.data.deployment_policy import DeploymentPolicy
from crosslayer_transcoder.data.shared_memory import SharedActivationBuffer

logger = logging.getLogger(__name__)


class ActivationDataModule(L.LightningDataModule):
    """
    Lightning DataModule for loading model activation data.
    Supports both shared memory streaming and simple buffer-based loading.
    """

    def __init__(
        self,
        # Buffer settings
        buffer_size: int = 10_000_000,
        n_in_out: int = 2,
        n_layers: int = 12,
        activation_dim: int = 768,
        dtype: str = "float16",
        max_batch_size: int = 50_000,
        minimum_fill_threshold: float = 0.0,
        # Model settings for activation generation
        model_name: str = "openai-community/gpt2",
        model_dtype: str = "float32",
        # Dataset settings
        dataset_name: str = "Skylion007/openwebtext",
        dataset_split: str = "train",
        max_sequence_length: int = 1024,
        # Generation settings
        generation_batch_size: int = 32,
        refresh_interval: float = 0.1,
        # Memory settings
        shared_memory_name: str = "activation_buffer",
        timeout_seconds: int = 30,
        # File paths
        init_file: Optional[str] = None,
        # DataLoader settings
        batch_size: int = 4000,
        num_workers: int = 20,
        prefetch_factor: int = 2,
        shuffle: bool = True,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        # Advanced settings
        use_shared_memory: bool = True,  # Use shared memory streaming by default
        device_map: str = "auto",
        # Deployment policy
        deployment_policy: str = "dynamic",  # CPU/GPU deployment policy: "cpu_only", "gpu_only", or "dynamic"
        # Activation extraction model architecture: "gpt2" or "gemma3"
        model_arch: str = "gpt2",
        # Producer/consumer split.
        #   "self":   default. This rank creates and feeds its own buffer (DDP fan-out:
        #            each rank gets its own producer on its own GPU; existing behavior).
        #   "client": this rank attaches to an EXTERNAL producer's shared buffer. Used by
        #            the 1-producer + N-consumer-DDP topology. The standalone producer
        #            (see crosslayer_transcoder.data.standalone_producer) must be running
        #            and must have created the buffer with the same `shared_memory_name`.
        producer_mode: str = "self",
        # WandB logging configuration
        wandb_logging: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize Activation DataModule.

        Args:
            # Buffer settings
            buffer_size: Number of activation samples in buffer
            n_in_out: Number of input/output activations per layer (usually 2)
            n_layers: Number of model layers
            activation_dim: Dimension of activation vectors
            dtype: Data type ("float16" or "float32")
            max_batch_size: Maximum batch size for high-throughput clients
            minimum_fill_threshold: Minimum buffer fill ratio (0.0-1.0) before providing activations

            # Model settings
            model_name: HuggingFace model name for activation generation
            model_dtype: Model data type ("float16" or "float32")

            # Dataset settings
            dataset_name: HuggingFace dataset name for text data
            dataset_split: Dataset split to use ("train", "validation", etc.)
            max_sequence_length: Maximum sequence length for text processing

            # Generation settings
            generation_batch_size: Batch size for activation generation
            refresh_interval: Seconds between checking for refresh requests

            # Memory settings
            shared_memory_name: Name for shared memory buffer
            timeout_seconds: Timeout for buffer operations

            # File paths
            init_file: Path to initial HDF5 activation file

            # DataLoader settings
            batch_size: Training batch size
            num_workers: Number of DataLoader workers
            prefetch_factor: DataLoader prefetch factor
            shuffle: Whether to shuffle data
            persistent_workers: Whether to keep workers persistent
            pin_memory: Whether to pin memory

            # Advanced settings
            use_shared_memory: Whether to use shared memory streaming
            device_map: Device map for model loading ("cpu", "auto", "cuda:0", "cuda:0,1,2,3")
            deployment_policy: GPT-2 deployment policy ("cpu_only", "gpu_only", or "dynamic")

            # WandB logging configuration
            wandb_logging: WandB logging configuration
        """
        super().__init__()
        self.save_hyperparameters()

        # Buffer settings
        self.buffer_size = buffer_size
        self.n_in_out = n_in_out
        self.n_layers = n_layers
        self.activation_dim = activation_dim
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.minimum_fill_threshold = minimum_fill_threshold

        # Model settings for activation generation
        self.model_name = model_name
        self.model_dtype = model_dtype

        # Dataset settings
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.max_sequence_length = max_sequence_length

        # Generation settings
        self.generation_batch_size = generation_batch_size
        self.refresh_interval = refresh_interval

        # Memory settings
        self.shared_memory_name = shared_memory_name
        self.timeout_seconds = timeout_seconds

        # File paths
        self.init_file = init_file

        # DataLoader settings
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle = shuffle
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

        # Advanced settings
        self.use_shared_memory = use_shared_memory
        self.deployment_policy = DeploymentPolicy.from_string(deployment_policy)
        self.model_arch = model_arch
        if producer_mode not in ("self", "client"):
            raise ValueError(f"producer_mode must be 'self' or 'client', got {producer_mode!r}")
        self.producer_mode = producer_mode

        # WandB configuration
        self.wandb_logging = wandb_logging or {}

        # Convert dtype string to torch.dtype
        self.torch_dtype = getattr(torch, dtype)
        self.model_torch_dtype = getattr(torch, model_dtype)

        # Will be created in setup
        self.data_loader = None
        self.shared_buffer = None
        self.data_generator = None

        # Device configuration
        self.device_map = device_map

    def get_memory_estimate_gb(self) -> float:
        """Estimate total memory usage in GB."""
        # Buffer memory: [buffer_size, n_in_out, n_layers, activation_dim]
        element_size = torch.tensor([], dtype=self.torch_dtype).element_size()
        buffer_memory = self.buffer_size * self.n_in_out * self.n_layers * self.activation_dim * element_size

        # Validity mask memory
        validity_memory = self.buffer_size  # 1 byte per sample

        # Total shared memory
        shared_memory = buffer_memory + validity_memory

        return shared_memory / (1024**3)

    def validate_config(self):
        """Validate configuration settings."""
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")

        if self.activation_dim <= 0:
            raise ValueError("activation_dim must be positive")

        if self.generation_batch_size <= 0:
            raise ValueError("generation_batch_size must be positive")

        memory_gb = self.get_memory_estimate_gb()
        if memory_gb > 100:  # Warning for very large memory usage
            logger.warning(f"Estimated memory usage is {memory_gb:.2f} GB")

    def setup(self, stage: str = None):
        """
        Set up the data loader.
        Incorporates factory logic directly instead of using separate factory.
        """
        if stage == "fit" or stage is None:
            logger.info("Setting up activation data loader...")

            # Validate configuration
            self.validate_config()

            logger.info(f"Buffer size: {self.buffer_size:,} samples")
            logger.info(f"Estimated memory: {self.get_memory_estimate_gb():.2f} GB")

            if self.use_shared_memory:
                self._setup_shared_memory_loader()
            else:
                self._setup_simple_buffer_loader()

            logger.info("Data loader setup complete")

    def _setup_shared_memory_loader(self):
        """Set up shared memory streaming data loader (factory logic)."""
        # Set spawn method for shared memory mode (needed for PyTorch tensor sharing)
        import torch.multiprocessing as mp

        try:
            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)
            logger.info("Set multiprocessing method to 'spawn' for shared memory compatibility")
        except RuntimeError as e:
            logger.warning(f"Could not set spawn method: {e}. Disabling multiprocessing in DataLoader")
            # If we can't set spawn, disable multiprocessing in the DataLoader

        rank = self.trainer.global_rank if self.trainer is not None else 0
        local_rank = self.trainer.local_rank if self.trainer is not None else 0
        world_size = self.trainer.world_size if self.trainer is not None else 1

        if self.producer_mode == "client":
            # Attach to an externally-launched standalone producer's buffer.
            # All consumer ranks share ONE buffer with no per-rank suffix; sharded
            # reads avoid double-consumption (see SharedActivationBuffer).
            logger.info(
                f"producer_mode=client: rank {rank}/{world_size} attaching to existing "
                f"shared buffer {self.shared_memory_name!r}"
            )
            self.shared_buffer = SharedActivationBuffer(
                buffer_size=self.buffer_size,
                n_in_out=self.n_in_out,
                n_layers=self.n_layers,
                activation_dim=self.activation_dim,
                dtype=self.torch_dtype,
                shared_memory_name=self.shared_memory_name,
                timeout_seconds=self.timeout_seconds,
                generation_batch_size=self.generation_batch_size,
                max_sequence_length=self.max_sequence_length,
                minimum_fill_threshold=self.minimum_fill_threshold,
                batch_size=self.batch_size,
                create=False,
                consumer_rank=rank,
                consumer_world_size=world_size,
            )
            self.data_generator = None
        else:
            # Under DDP each rank runs its own generator on its own GPU. We must
            # use the *absolute* physical device id (cuda:local_rank), not "cuda:0":
            # Lightning's DDP doesn't filter CUDA_VISIBLE_DEVICES for child processes
            # spawned by the rank, so a relative "cuda:0" in rank N>0 maps to physical
            # cuda:0 and competes with rank 0's training memory. Each rank also gets
            # a unique shared-memory name to avoid /dev/shm collisions.
            if world_size > 1:
                shm_name = f"{self.shared_memory_name}_rank{rank}"
                gen_device = f"cuda:{local_rank}"
                wandb_cfg = dict(self.wandb_logging or {})
                if wandb_cfg.get("enabled"):
                    base_run = wandb_cfg.get("run_name") or "data-generator"
                    wandb_cfg["run_name"] = f"{base_run}-rank{rank}"
            else:
                shm_name = self.shared_memory_name
                gen_device = self.device_map
                wandb_cfg = self.wandb_logging

            self.shared_buffer = SharedActivationBuffer(
                buffer_size=self.buffer_size,
                n_in_out=self.n_in_out,
                n_layers=self.n_layers,
                activation_dim=self.activation_dim,
                dtype=self.torch_dtype,
                shared_memory_name=shm_name,
                timeout_seconds=self.timeout_seconds,
                generation_batch_size=self.generation_batch_size,
                max_sequence_length=self.max_sequence_length,
                minimum_fill_threshold=self.minimum_fill_threshold,
                batch_size=self.batch_size,
            )

            self.data_generator = DataGeneratorProcess(
                shared_buffer=self.shared_buffer,
                buffer_size=self.buffer_size,
                n_in_out=self.n_in_out,
                n_layers=self.n_layers,
                activation_dim=self.activation_dim,
                dtype=self.torch_dtype,
                max_batch_size=self.max_batch_size,
                model_name=self.model_name,
                model_dtype=self.model_torch_dtype,
                dataset_name=self.dataset_name,
                dataset_split=self.dataset_split,
                max_sequence_length=self.max_sequence_length,
                generation_batch_size=self.generation_batch_size,
                refresh_interval=self.refresh_interval,
                deployment_policy=self.deployment_policy,
                init_file=self.init_file,
                device_map=gen_device,
                wandb_logging=wandb_cfg,
                model_arch=self.model_arch,
            )

            logger.info("Starting data generator process...")
            self.data_generator.start()
            logger.info("Data generator process started")

        self.data_loader = torch.utils.data.DataLoader(
            self.shared_buffer,
            batch_size=None,
            num_workers=1,
            prefetch_factor=3,
            shuffle=False,
            pin_memory=True,
        )

    def _setup_simple_buffer_loader(self):
        """Set up simple buffer-based data loader."""
        # For simple buffer mode, prefer fork method for h5py compatibility
        # (h5py objects can't be pickled, but work with fork memory sharing)
        import sys

        import torch.multiprocessing as mp

        if sys.platform.startswith("linux") and mp.get_start_method(allow_none=True) != "fork":
            try:
                mp.set_start_method("fork", force=True)
                logger.info("Set multiprocessing method to 'fork' for h5py compatibility")
            except RuntimeError:
                logger.warning("Could not set fork method, using num_workers=0 for h5py safety")
                # If we can't set fork, force single-threaded to avoid pickling issues
                self.num_workers = 0
                self.persistent_workers = False

        if not self.init_file or not os.path.exists(self.init_file):
            raise ValueError(
                f"init_file must be provided and exist for simple buffer loading: {self.init_file}"
            )

        # Use simple DiscBuffer approach (like current train.py)
        from utils.buffer import DiscBuffer

        buffer = DiscBuffer(self.init_file, "tensor")

        self.data_loader = torch.utils.data.DataLoader(
            buffer,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=self.shuffle,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        """Return training data loader."""
        return self.data_loader

    def val_dataloader(self):
        """Return validation data loader (same as training for now)."""
        return self.data_loader

    def teardown(self, stage: str = None):
        """Clean up resources."""
        logger.info("Cleaning up activation data loader...")

        if self.data_loader is not None and hasattr(self.data_loader, "cleanup"):
            self.data_loader.cleanup()

        if self.data_generator and self.data_generator.is_alive():
            logger.info("Terminating data generator process...")
            self.data_generator.terminate()
            self.data_generator.join(timeout=5.0)

            if self.data_generator.is_alive():
                logger.warning("Force killing data generator process...")
                self.data_generator.kill()
                self.data_generator.join()

        if self.shared_buffer:
            self.shared_buffer.cleanup()

        logger.info("Cleanup complete")
