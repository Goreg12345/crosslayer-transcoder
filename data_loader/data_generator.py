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

from data_loader import text_dataset
from data_loader.activation_sources import ActivationComputer, DiskActivationSource
from data_loader.config import DataLoaderConfig
from data_loader.generation_loop import DataGenerationLoop
from data_loader.process_monitor import ProcessMonitor
from data_loader.shared_memory import SharedActivationBuffer

logger = logging.getLogger(__name__)


class DataGeneratorProcess(mp.Process):
    """
    Background process for generating activation data.
    Now focused only on process lifecycle - delegates actual work to specialized components.
    """

    def __init__(
        self, 
        shared_buffer: SharedActivationBuffer, 
        config: DataLoaderConfig
    ):
        super().__init__(
            daemon=False
        )  # Can't be daemon if we want to use DataLoader workers
        self.shared_buffer = shared_buffer
        self.config = config
        self.running = False

        # Components will be created in the process
        self.generation_loop: Optional[DataGenerationLoop] = None
        self.monitor: Optional[ProcessMonitor] = None
        self.disk_source: Optional[DiskActivationSource] = None

    def run(self):
        """Main process loop."""
        try:
            logger.info("Starting data generator process...")
            self.setup()
            self.running = True
            
            # Start the generation loop
            self.generation_loop.generation_loop()
            
        except Exception as e:
            logger.error(f"Data generator process error: {e}")
        finally:
            self.cleanup()

    def setup(self):
        """Initialize components using the new architecture."""
        # 1. Create models in the process (avoid pickle issues)
        logger.info(f"Loading CPU model: {self.config.model_name}")
        cpu_model = nnsight.LanguageModel(
            self.config.model_name,
            device_map="cpu",
            dispatch=True,
            torch_dtype=self.config.model_dtype,
        )
        cpu_model.requires_grad_(False)

        logger.info(f"Loading GPU model: {self.config.model_name}")
        gpu_model = nnsight.LanguageModel(
            self.config.model_name,
            device_map="auto",
            dispatch=True,
            torch_dtype=self.config.model_dtype,
        )
        gpu_model.requires_grad_(False)

        # 2. Load dataset in the process
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        dataset = load_dataset(
            self.config.dataset_name, split=self.config.dataset_split
        )

        # 3. Create text dataset with tokenization
        logger.info("Creating text dataset loader...")
        token_dataset = text_dataset.TextDataset(
            dataset,
            cpu_model.tokenizer,
            self.config.generation_batch_size,
            drop_last_batch=False,
            seq_len=self.config.max_sequence_length - 1,  # -1 for BOS token
        )

        text_dataset_loader = iter(
            DataLoader(
                token_dataset,
                batch_size=None,
                shuffle=False,
                num_workers=8,  # Optimal for performance
                prefetch_factor=4,  # Optimal for performance
                worker_init_fn=text_dataset.worker_init_fn,
            )
        )

        # 4. Create components
        activation_computer = ActivationComputer(self.config)
        self.monitor = ProcessMonitor()
        
        # 5. Setup disk source if available
        if self.config.init_file and os.path.exists(self.config.init_file):
            logger.info(f"Setting up disk source: {self.config.init_file}")
            self.disk_source = DiskActivationSource(self.config.init_file)
        else:
            self.disk_source = None

        # 6. Create generation loop with all dependencies
        self.generation_loop = DataGenerationLoop(
            shared_buffer=self.shared_buffer,
            config=self.config,
            cpu_model=cpu_model,
            gpu_model=gpu_model,
            text_dataset_loader=text_dataset_loader,
            activation_computer=activation_computer,
            monitor=self.monitor,
            disk_source=self.disk_source,
        )
        
        # Set dataset reference for the loop
        self.generation_loop.set_dataset(dataset)

        # 7. Initial population from disk if available
        if self.disk_source:
            refilled_count = self.generation_loop.refill_from_disk()
            logger.info(f"Initial buffer population: {refilled_count} samples loaded from file")

        logger.info("Data generator setup complete!")

    def cleanup(self):
        """Clean up resources and terminate any child processes."""
        logger.info("Cleaning up data generator...")
        self.running = False

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