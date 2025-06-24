"""
Data generation loop that orchestrates activation generation and buffer management.
Handles model selection, buffer monitoring, and coordinates between different data sources.
"""

import logging
import time
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from data_loader import text_dataset
from data_loader.activation_sources import ActivationComputer, DiskActivationSource
from data_loader.config import DataLoaderConfig
from data_loader.process_monitor import ProcessMonitor
from data_loader.shared_memory import SharedActivationBuffer

logger = logging.getLogger(__name__)


class DataGenerationLoop:
    """
    Manages the main data generation loop with device optimization and source coordination.
    Handles model management, buffer monitoring, and decision making between compute vs file sources.
    """

    def __init__(
        self,
        shared_buffer: SharedActivationBuffer,
        config: DataLoaderConfig,
        cpu_model: Any,
        gpu_model: Any,
        text_dataset_loader: Any,
        activation_computer: ActivationComputer,
        monitor: ProcessMonitor,
        disk_source: Optional[DiskActivationSource] = None,
    ):
        self.shared_buffer = shared_buffer
        self.config = config
        self.cpu_model = cpu_model
        self.gpu_model = gpu_model
        self.text_dataset_loader = text_dataset_loader
        self.activation_computer = activation_computer
        self.monitor = monitor
        self.disk_source = disk_source

        # Current model and device state
        self.current_model = cpu_model  # Start with CPU
        self.current_device = "cpu"

        # Runtime state
        self.running = True
        self.dataset = None  # Will be set by parent process

    def set_dataset(self, dataset: Any) -> None:
        """Set the dataset reference for text data loading."""
        self.dataset = dataset

    def generation_loop(self):
        """Main generation loop - extracted from existing code."""
        self.monitor.log_generation_start()

        while self.running:
            # Check for indices that need refreshing (invalid indices)
            indices_to_refresh = self.shared_buffer._get_invalid_indices()

            if len(indices_to_refresh) == 0:
                # Update dashboard when sleeping
                stats = self.shared_buffer.get_stats()
                self.monitor.update_dashboard("SLEEPING", stats, self.current_device)
                time.sleep(self.config.refresh_interval)
                continue

            # Generate new activations
            gen_start = time.time()
            activations = self._generate_activations()
            gen_time = time.time() - gen_start

            # Take only as many activations as we have indices
            num_indices = len(indices_to_refresh)
            if len(activations) >= num_indices:
                activations = activations[:num_indices]
                indices_to_refresh = indices_to_refresh[:num_indices]
            elif len(activations) < num_indices:
                # refresh part of the buffer and let the next loop refresh the rest
                indices_to_refresh = indices_to_refresh[: len(activations)]

            # Update shared buffer
            self.shared_buffer.set_activations(indices_to_refresh, activations)

            # Calculate and update refresh rate
            refresh_rate = len(indices_to_refresh) / gen_time if gen_time > 0 else 0
            self.monitor.set_refresh_rate(refresh_rate)

            # Update dashboard
            stats = self.shared_buffer.get_stats()
            self.monitor.update_dashboard("GENERATING", stats, self.current_device)

    def _generate_activations(self) -> torch.Tensor:
        """
        Generate activation data by processing text through the model.
        Extracted from existing generate_activations method.

        Returns:
            Tensor of activations [batch*seq_len, in/out, n_layers, d_model]
        """
        self._select_device()

        try:
            batch = next(self.text_dataset_loader)
        except StopIteration:
            # Dataset exhausted, recreate loader
            self.monitor.log_dataset_exhausted()
            token_dataset = text_dataset.TextDataset(
                self.dataset,
                self.cpu_model.tokenizer,  # Use CPU model tokenizer (consistent)
                self.config.generation_batch_size,
                drop_last_batch=False,
                seq_len=self.config.max_sequence_length - 1,
            )
            self.text_dataset_loader = iter(
                DataLoader(
                    token_dataset,
                    batch_size=None,
                    shuffle=False,
                    num_workers=8,  # Optimal for performance
                    prefetch_factor=4,  # Optimal for performance
                    worker_init_fn=text_dataset.worker_init_fn,
                )
            )
            batch = next(self.text_dataset_loader)

        # Move to current device (adaptive CPU/GPU)
        batch = batch.to(self.current_device)

        # Prepend BOS token (like in benchmark)
        batch = torch.roll(batch, shifts=1, dims=1)
        batch[:, 0] = self.current_model.config.bos_token_id

        # Extract activations using the activation computer
        mlp_acts = self.activation_computer.get_next_batch(self.current_model, batch)

        return mlp_acts

    def _should_move_to_gpu(self, valid_percentage: float) -> bool:
        """Check if model should be moved to GPU based on buffer fill level."""
        return valid_percentage < 50.0 and self.current_device == "cpu"

    def _should_move_to_cpu(self, valid_percentage: float) -> bool:
        """Check if model should be moved to CPU based on buffer fill level."""
        return valid_percentage > 80.0 and self.current_device == "cuda"

    def _move_model_to_device(self, target_device: str) -> None:
        """Switch between pre-loaded CPU and GPU models."""
        if target_device == "cuda":
            self.current_model = self.gpu_model
            self.current_device = "cuda"
        else:
            self.current_model = self.cpu_model
            self.current_device = "cpu"

        # Clear GPU cache to clean up activation residue
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _select_device(self) -> None:
        """Select optimal device based on buffer fill level."""
        stats = self.shared_buffer.get_stats()
        valid_percentage = stats["valid_percentage"]

        if self._should_move_to_gpu(valid_percentage):
            self._move_model_to_device("cuda")
        elif self._should_move_to_cpu(valid_percentage):
            self._move_model_to_device("cpu")

    def refill_from_disk(self, batch_size: int = 10000) -> int:
        """
        Refill invalid indices using disk source if available.
        
        Args:
            batch_size: Batch size for disk reads
            
        Returns:
            Number of indices successfully refilled
        """
        if not self.disk_source or not self.disk_source.is_available():
            self.monitor.log_warning("No disk source available for refilling")
            return 0

        # Get all invalid indices
        invalid_indices = self.shared_buffer._get_invalid_indices()
        if len(invalid_indices) == 0:
            return 0

        total_refilled = 0

        # Process in batches
        for i in range(0, len(invalid_indices), batch_size):
            batch_indices = invalid_indices[i : i + batch_size]
            samples_requested = len(batch_indices)

            try:
                # Get samples from disk source
                samples = self.disk_source.get_next_batch(samples_requested)
                
                # Only fill as many indices as we have samples
                indices_to_fill = batch_indices[:len(samples)]
                
                # Update buffer
                self.shared_buffer.set_activations(indices_to_fill, samples)
                total_refilled += len(indices_to_fill)
                
                self.monitor.log_refill_progress(len(indices_to_fill), "disk")
                
            except Exception as e:
                self.monitor.log_error("disk refill", e)
                break

        return total_refilled

    def stop(self) -> None:
        """Stop the generation loop."""
        self.running = False