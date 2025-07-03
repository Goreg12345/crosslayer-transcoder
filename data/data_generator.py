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

from data import text_dataset
from data.activation_sources import ActivationComputer, DiskActivationSource

# DataLoaderConfig no longer needed - using individual parameters
from data.generation_loop import DataGenerationLoop
from data.process_monitor import ProcessMonitor
from data.shared_memory import SharedActivationBuffer

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
        init_file: Optional[str] = None,
        device_map: str = "auto",
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
        self.init_file = init_file
        self.device_map = device_map
        # Components will be created in the process
        self.generation_loop: Optional[DataGenerationLoop] = None
        self.disk_source: Optional[DiskActivationSource] = None

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

        monitor = ProcessMonitor()

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
            monitor=monitor,
            disk_source=self.disk_source,
            generation_batch_size=self.generation_batch_size,
            max_sequence_length=self.max_sequence_length,
        )

        self.generation_loop.refill_from_disk()

        logger.info(f"Loading CPU model: {self.model_name}")
        cpu_model = nnsight.LanguageModel(
            self.model_name,
            device_map="cpu",
            dispatch=True,
            torch_dtype=self.model_dtype,
        )
        cpu_model.requires_grad_(False)

        logger.info(f"Loading GPU model: {self.model_name}")
        gpu_model = nnsight.LanguageModel(
            self.model_name,
            device_map=self.device_map,
            dispatch=True,
            torch_dtype=self.model_dtype,
        )
        gpu_model.requires_grad_(False)

        # 2. Load dataset in the process
        logger.info(f"Loading dataset: {self.dataset_name}")
        dataset = load_dataset(self.dataset_name, split=self.dataset_split)

        # 3. Create text dataset with tokenization
        logger.info("Creating text dataset loader...")
        token_dataset = text_dataset.TextDataset(
            dataset,
            cpu_model.tokenizer,
            self.generation_batch_size,
            drop_last_batch=False,
            seq_len=self.max_sequence_length - 1,  # -1 for BOS token
        )

        text_dataset_loader = DataLoader(
            token_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,  # Reduced from 8 - fewer workers = faster startup
            # prefetch_factor=2,  # Reduced prefetch factor
            # worker_init_fn=text_dataset.worker_init_fn,
        )
        text_dataset_loader = iter(text_dataset_loader)

        # 4. Create components
        activation_computer = ActivationComputer(self.n_layers)

        # Set dataset reference for the loop
        self.generation_loop.set_dataset(dataset)
        self.generation_loop.generation_loop(
            cpu_model=cpu_model,
            gpu_model=gpu_model,
            text_dataset_loader=text_dataset_loader,
            activation_computer=activation_computer,
        )

        logger.info("Data generator setup complete!")

    def cleanup(self):
        """Clean up resources and terminate any child processes."""
        logger.info("Cleaning up data generator...")

        # Stop the generation loop
        if self.generation_loop:
            self.generation_loop.stop()

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
