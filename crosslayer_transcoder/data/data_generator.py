"""
Data generation process for creating activation data.
Uses the new component architecture with clean separation of responsibilities.
"""

import logging
import multiprocessing as mp
import os
from typing import Optional

import nnsight
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from crosslayer_transcoder.data import text_dataset
from crosslayer_transcoder.data.activation_sources import (
    ActivationComputer,
    DiskActivationSource,
)
from crosslayer_transcoder.data.deployment_policy import DeploymentPolicy

# DataLoaderConfig no longer needed - using individual parameters
from crosslayer_transcoder.data.generation_loop import DataGenerationLoop
from crosslayer_transcoder.data.process_monitor import (
    ProcessMonitor,
    WandBProcessMonitor,
)
from crosslayer_transcoder.data.shared_memory import SharedActivationBuffer

logger = logging.getLogger(__name__)


class DataGeneratorProcess(mp.Process):
    """
    Background process for generating activation data.
    Now focused only on process lifecycle - delegates actual work to specialized components.
    """

    def __init__(
        self,
        shared_buffer: SharedActivationBuffer,
        buffer_size: int,
        n_in_out: int,
        n_layers: int,
        activation_dim: int,
        dtype: torch.dtype,
        max_batch_size: int,
        model_name: str,
        model_dtype: torch.dtype,
        dataset_name: str,
        dataset_split: str,
        max_sequence_length: int,
        generation_batch_size: int,
        refresh_interval: float,
        deployment_policy: DeploymentPolicy = DeploymentPolicy.DYNAMIC,
        init_file: Optional[str] = None,
        device_map: str = "auto",
        wandb_logging: Optional[dict] = None,
    ):
        super().__init__(
            daemon=False
        )  # Can't be daemon if we want to use DataLoader workers
        self.shared_buffer = shared_buffer

        # Store parameters directly instead of config object
        self.buffer_size = buffer_size
        self.n_in_out = n_in_out
        self.n_layers = n_layers
        self.activation_dim = activation_dim
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.model_name = model_name
        self.model_dtype = model_dtype
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.max_sequence_length = max_sequence_length
        self.generation_batch_size = generation_batch_size
        self.refresh_interval = refresh_interval
        self.deployment_policy = deployment_policy
        self.init_file = init_file
        self.device_map = device_map

        # WandB configuration
        self.wandb_logging = wandb_logging or {}

        # Components will be created in the process
        self.generation_loop: Optional[DataGenerationLoop] = None
        self.disk_source: Optional[DiskActivationSource] = None
        self.monitor: Optional[ProcessMonitor] = None

    def run(self):
        """Main process loop."""
        try:
            logger.info("Starting data generator process...")
            self.setup()

        except Exception as e:
            logger.error(f"Data generator process error: {e}")
        finally:
            self.cleanup()

    def setup(self):
        """Initialize components using the new architecture."""
        # 1. Create models in the process (avoid pickle issues)
        # 7. Initial population from disk if available
        # 5. Setup disk source if available
        if self.init_file and os.path.exists(self.init_file):
            logger.info(f"Setting up disk source: {self.init_file}")
            self.disk_source = DiskActivationSource(self.init_file, dtype=self.dtype)
        else:
            self.disk_source = None
        print("disk source available", self.disk_source is not None)

        # Create monitor based on WandB configuration
        if self.wandb_logging.get("enabled", False):
            # Create WandB configuration with data generation parameters
            wandb_config = {
                "buffer_size": self.buffer_size,
                "n_in_out": self.n_in_out,
                "n_layers": self.n_layers,
                "activation_dim": self.activation_dim,
                "model_name": self.model_name,
                "dataset_name": self.dataset_name,
                "generation_batch_size": self.generation_batch_size,
                "max_sequence_length": self.max_sequence_length,
                "device_map": self.device_map,
            }

            self.monitor = WandBProcessMonitor(
                project=self.wandb_logging.get("project", "crosslayer-transcoder"),
                group=self.wandb_logging.get("group"),
                run_name=self.wandb_logging.get("run_name", "data-generator"),
                tags=self.wandb_logging.get("tags", ["data-generation"]),
                config=wandb_config,
                save_dir=self.wandb_logging.get("save_dir", "./wandb"),
            )

            # Update logging interval if specified
            if "log_interval" in self.wandb_logging:
                self.monitor._wandb_log_interval = self.wandb_logging["log_interval"]

            logger.info("Using WandB process monitor")
        else:
            self.monitor = ProcessMonitor()
            logger.info("Using standard process monitor")

        # 6. Create generation loop with all dependencies
        # Note: DataGenerationLoop will need to be updated to accept individual parameters too
        self.generation_loop = DataGenerationLoop(
            shared_buffer=self.shared_buffer,
            buffer_size=self.buffer_size,
            n_in_out=self.n_in_out,
            n_layers=self.n_layers,
            activation_dim=self.activation_dim,
            dtype=self.dtype,
            max_batch_size=self.max_batch_size,
            refresh_interval=self.refresh_interval,
            monitor=self.monitor,
            deployment_policy=self.deployment_policy,
            device_map=self.device_map,
            disk_source=self.disk_source,
            generation_batch_size=self.generation_batch_size,
            max_sequence_length=self.max_sequence_length,
        )

        self.generation_loop.refill_from_disk()

        # 2. Load dataset in the process
        logger.info(f"Loading dataset: {self.dataset_name}")
        dataset = load_dataset(
            self.dataset_name,
            split=self.dataset_split,
            trust_remote_code=True,
        )

        # 3. Create components
        activation_computer = ActivationComputer(self.n_layers)

        # Set dataset reference for the loop and start generation
        # Text dataset creation is now handled in generation_loop after models are set up
        self.generation_loop.set_dataset(dataset)
        self.generation_loop.generation_loop(
            model_name=self.model_name,
            model_dtype=self.model_dtype,
            activation_computer=activation_computer,
        )

        logger.info("Data generator setup complete!")

    def cleanup(self):
        """Clean up resources and terminate any child processes."""
        logger.info("Cleaning up data generator...")

        # Stop the generation loop
        if self.generation_loop:
            self.generation_loop.stop()

        # Finish WandB logging if using WandB monitor
        if self.monitor and hasattr(self.monitor, "finish"):
            self.monitor.finish()

        # Close disk source
        if self.disk_source:
            self.disk_source.close()

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Kill any remaining child processes
        try:
            import psutil

            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except:
                    pass
        except ImportError:
            pass  # psutil not available

        logger.info("Data generator cleanup complete")

    def terminate(self):
        """Terminate the process gracefully."""
        logger.info("Terminating data generator process...")
        self.cleanup()
        super().terminate()
