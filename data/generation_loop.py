"""
Data generation loop that orchestrates activation generation and buffer management.
Handles model selection, buffer monitoring, and coordinates between different data sources.
"""

import logging
import time
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from data import text_dataset
from data.activation_sources import ActivationComputer, DiskActivationSource
from data.deployment_policy import BaseDeploymentPolicy, DeploymentPolicy, create_deployment_policy

# DataLoaderConfig no longer needed - using individual parameters
from data.process_monitor import ProcessMonitor
from data.shared_memory import SharedActivationBuffer

logger = logging.getLogger(__name__)


class DataGenerationLoop:
    """
    Manages the main data generation loop with device optimization and source coordination.
    Handles model management, buffer monitoring, and decision making between compute vs file sources.
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
        refresh_interval: float,
        generation_batch_size: int,
        max_sequence_length: int,
        monitor: ProcessMonitor,
        deployment_policy: DeploymentPolicy = DeploymentPolicy.DYNAMIC,
        device_map: str = "auto",
        disk_source: Optional[DiskActivationSource] = None,
    ):
        self.shared_buffer = shared_buffer
        self.buffer_size = buffer_size
        self.n_in_out = n_in_out
        self.n_layers = n_layers
        self.activation_dim = activation_dim
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.refresh_interval = refresh_interval
        self.generation_batch_size = generation_batch_size
        self.max_sequence_length = max_sequence_length
        self.text_dataset_loader = None
        self.activation_computer = None
        self.monitor = monitor
        self.monitor.log_generation_start()
        self.disk_source = disk_source

        # Create deployment policy instance
        self.deployment_policy = create_deployment_policy(deployment_policy, device_map)

        # Store model setup parameters for later use
        self.model_setup_params = {}

        # Runtime state
        self.running = True
        self.dataset = None  # Will be set by parent process

    def set_dataset(self, dataset: Any) -> None:
        """Set the dataset reference for text data loading."""
        self.dataset = dataset

    def _setup_text_dataset_loader(self) -> None:
        """Create text dataset loader using the tokenizer from current model."""
        if self.dataset is None:
            raise RuntimeError("Dataset must be set before setting up text dataset loader")

        # Get tokenizer from current model (any model will do since tokenizer is the same)
        current_model = self.deployment_policy.get_current_model()
        if current_model is None:
            raise RuntimeError("Models must be set up before creating text dataset loader")

        logger.info("Creating text dataset loader...")
        token_dataset = text_dataset.TextDataset(
            self.dataset,
            current_model.tokenizer,
            self.generation_batch_size,
            drop_last_batch=False,
            seq_len=self.max_sequence_length - 1,  # -1 for BOS token
        )

        text_dataset_loader = DataLoader(
            token_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=8,  # Optimal for performance
            prefetch_factor=4,  # Optimal for performance
            worker_init_fn=text_dataset.worker_init_fn,
        )
        self.text_dataset_loader = iter(text_dataset_loader)

    def generation_loop(
        self,
        model_name: str,
        model_dtype: torch.dtype,
        activation_computer: ActivationComputer,
    ):
        """Main generation loop - models are now managed by deployment policy."""
        # Setup models through deployment policy
        self.deployment_policy.setup_models(model_name, model_dtype, **self.model_setup_params)

        # Create text dataset loader now that we have models and can access tokenizer
        self._setup_text_dataset_loader()

        self.activation_computer = activation_computer

        last_time = time.time()

        while self.running:
            # Check for indices that need refreshing (invalid indices)
            indices_to_refresh = self.shared_buffer._get_invalid_indices()

            if len(indices_to_refresh) == 0:
                # Update dashboard when sleeping
                stats = self.shared_buffer.get_stats()
                self.monitor.update_dashboard("SLEEPING", stats, self.deployment_policy.get_current_device())
                time.sleep(self.refresh_interval)
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
            it_time = time.time() - last_time
            last_time = time.time()
            refresh_rate = len(indices_to_refresh) / it_time if it_time > 0 else 0
            self.monitor.set_refresh_rate(refresh_rate)

            # Update dashboard
            stats = self.shared_buffer.get_stats()
            self.monitor.update_dashboard("GENERATING", stats, self.deployment_policy.get_current_device())

    def _generate_activations(self) -> torch.Tensor:
        """
        Generate activation data by processing text through the model.
        Uses deployment policy for device and model management.

        Returns:
            Tensor of activations [batch*seq_len, in/out, n_layers, d_model]
        """
        # Let deployment policy select device based on buffer state
        stats = self.shared_buffer.get_stats()
        current_device = self.deployment_policy.select_device(stats)
        current_model = self.deployment_policy.get_current_model()

        try:
            batch, mask = next(self.text_dataset_loader)
        except StopIteration:
            # Dataset exhausted, recreate loader
            self.monitor.log_dataset_exhausted()
            token_dataset = text_dataset.TextDataset(
                self.dataset,
                current_model.tokenizer,  # Use current model tokenizer
                self.generation_batch_size,
                drop_last_batch=False,
                seq_len=self.max_sequence_length - 1,
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
            batch, mask = next(self.text_dataset_loader)

        # Move to current device
        batch = batch.to(current_device)
        mask = mask.to(current_device)

        # Prepend BOS token (like in benchmark)
        batch = torch.roll(batch, shifts=1, dims=1)
        batch[:, 0] = current_model.config.bos_token_id

        # Extract activations using the activation computer
        mlp_acts = self.activation_computer.get_next_batch(current_model, batch, mask)

        return mlp_acts

    def refill_from_disk(self, batch_size: int = 40_000):
        """
        Refill invalid indices using disk source if available.

        Args:
            batch_size: Batch size for disk reads

        Returns:
            Number of indices successfully refilled
        """
        if not self.disk_source or not self.disk_source.is_available():
            self.monitor.log_warning("No disk source available for refilling")
            return

        # Get all invalid indices
        invalid_indices = self.shared_buffer._get_invalid_indices()
        if len(invalid_indices) == 0:
            return

        # Track refill timing and progress
        refill_start_time = time.time()
        total_refilled = 0

        # Process in batches
        for i in range(0, len(invalid_indices), batch_size):
            batch_start_time = time.time()
            batch_indices = invalid_indices[i : i + batch_size]
            samples_requested = len(batch_indices)

            try:
                # Update dashboard to show refill in progress
                stats = self.shared_buffer.get_stats()
                self.monitor.update_dashboard("REFILLING", stats, "disk")

                # Get samples from disk source
                samples = self.disk_source.get_next_batch(samples_requested)

                # Only fill as many indices as we have samples
                indices_to_fill = batch_indices[: len(samples)]

                # Update buffer
                self.shared_buffer.set_activations(indices_to_fill, samples)

                # Calculate refill rate for this batch
                batch_time = time.time() - batch_start_time
                batch_refilled = len(indices_to_fill)
                total_refilled += batch_refilled

                # Calculate running average refill rate
                total_time = time.time() - refill_start_time
                refill_rate = total_refilled / total_time if total_time > 0 else 0
                self.monitor.set_refresh_rate(refill_rate)

                # Update dashboard with progress
                updated_stats = self.shared_buffer.get_stats()
                self.monitor.log_refill_progress(batch_refilled, "disk", updated_stats)

            except Exception as e:
                self.monitor.log_error("disk refill", e)
                break

    def stop(self) -> None:
        """Stop the generation loop."""
        self.running = False
