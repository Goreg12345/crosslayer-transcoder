"""
Data generation process for creating activation data.
Samples tokens from webtext, feeds through LLM, and stores activations in shared buffer.
"""

import logging
import multiprocessing as mp
import os
import threading
import time
from typing import Any, Dict, List, Optional

import h5py
import nnsight
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from activation_server import text_dataset
from activation_server.config import ServerConfig
from activation_server.shared_memory import SharedActivationBuffer

logger = logging.getLogger(__name__)


class DataGeneratorProcess(mp.Process):
    """
    Background process for generating activation data.
    Continuously samples text, processes through LLM, and updates shared buffer.
    """

    def __init__(self, shared_buffer: SharedActivationBuffer, config: ServerConfig):
        super().__init__(
            daemon=False
        )  # Can't be daemon if we want to use DataLoader workers
        self.shared_buffer = shared_buffer
        self.config = config
        self.running = False

        # Will be initialized in the process
        self.cpu_model = None
        self.gpu_model = None
        self.current_model = None
        self.dataset = None
        self.text_dataset_loader = None

        # For initialization from file
        self.init_file_handle = None
        self.init_file_tensor = None
        self.init_file_position = 0

    def run(self):
        """Main process loop."""
        try:
            logger.info("Starting data generator process...")
            self.setup()
            self.running = True
            self.generation_loop()
        except Exception as e:
            logger.error(f"Data generator process error: {e}")
        finally:
            self.cleanup()

    def setup(self):
        """Initialize model, tokenizer, and dataset in the process."""

        # Initialize device management
        self.cpu_device = torch.device("cpu")
        self.gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cpu_model = nnsight.LanguageModel(
            self.config.model_name,
            device_map="cpu",
            dispatch=True,
            torch_dtype=self.config.model_dtype,
        )
        self.cpu_model.requires_grad_(False)

        logger.info(f"Loading GPU model: {self.config.model_name}")
        self.gpu_model = nnsight.LanguageModel(
            self.config.model_name,
            device_map="auto",
            dispatch=True,
            torch_dtype=self.config.model_dtype,
        )
        self.gpu_model.requires_grad_(False)

        # Start with CPU model
        self.current_model = self.cpu_model
        self.current_device = "cpu"

        # Load dataset
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        self.dataset = load_dataset(
            self.config.dataset_name, split=self.config.dataset_split
        )

        # Create text dataset with tokenization
        logger.info("Creating text dataset loader...")
        token_dataset = text_dataset.TextDataset(
            self.dataset,
            self.cpu_model.tokenizer,
            self.config.generation_batch_size,
            drop_last_batch=False,
            seq_len=self.config.max_sequence_length - 1,  # -1 for BOS token
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

        logger.info("Data generator setup complete!")

        # Initialize from file if provided
        if self.config.init_file and os.path.exists(self.config.init_file):
            self._setup_init_file()
            # Initial population of the buffer from file
            refilled_count = self.refill_from_file()
            logger.info(
                f"Initial buffer population: {refilled_count} samples loaded from file"
            )

    def _setup_init_file(self):
        """Setup initialization from pre-stored activations file."""
        try:
            logger.info(f"Setting up initialization from file: {self.config.init_file}")
            self.init_file_handle = h5py.File(self.config.init_file, "r")
            self.init_file_tensor = self.init_file_handle["tensor"]
            self.init_file_position = 0

            logger.info(
                f"Initialization file loaded: {self.init_file_tensor.shape} samples"
            )
        except Exception as e:
            logger.error(f"Failed to setup init file: {e}")
            self.init_file_handle = None
            self.init_file_tensor = None

    def _should_move_to_gpu(self, valid_percentage):
        """Check if model should be moved to GPU based on buffer fill level."""
        return valid_percentage < 50.0 and self.current_device == "cpu"

    def _should_move_to_cpu(self, valid_percentage):
        """Check if model should be moved to CPU based on buffer fill level."""
        return valid_percentage > 80.0 and self.current_device == "cuda"

    def _move_model_to_device(self, target_device):
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

    def _select_device(self):
        """Select optimal device based on buffer fill level."""
        stats = self.shared_buffer.get_stats()
        valid_percentage = stats["valid_percentage"]

        if self._should_move_to_gpu(valid_percentage):
            self._move_model_to_device("cuda")
        elif self._should_move_to_cpu(valid_percentage):
            self._move_model_to_device("cpu")

    def generation_loop(self):
        """Main generation loop."""
        logger.info("Starting generation loop...")

        # Initialize dashboard tracking
        self._dashboard_start_time = time.time()
        self._last_dashboard_update = self._dashboard_start_time
        self._last_refresh_rate = 0

        while self.running:
            # Check for indices that need refreshing (invalid indices)
            indices_to_refresh = self.shared_buffer._get_invalid_indices()

            if len(indices_to_refresh) == 0:
                # Update dashboard when sleeping
                self._update_dashboard("SLEEPING")
                time.sleep(self.config.refresh_interval)
                continue

            # Generate new activations
            gen_start = time.time()
            activations = self.generate_activations()
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

            # Store refresh rate for dashboard
            self._last_refresh_rate = (
                len(indices_to_refresh) / gen_time if gen_time > 0 else 0
            )

            # Update dashboard
            self._update_dashboard("GENERATING")

    def _update_dashboard(self, status):
        """Update the CLI dashboard with current buffer stats."""
        # Initialize dashboard timing if not set
        if not hasattr(self, "_dashboard_start_time"):
            self._dashboard_start_time = time.time()
        if not hasattr(self, "_last_dashboard_update"):
            self._last_dashboard_update = self._dashboard_start_time
        if not hasattr(self, "_last_refresh_rate"):
            self._last_refresh_rate = 0

        current_time = time.time()
        dashboard_update_interval = 0.05  # Update every 500ms to reduce flickering

        # Only update if enough time has passed
        if current_time - self._last_dashboard_update < dashboard_update_interval:
            return

        self._last_dashboard_update = current_time

        # Get buffer stats
        stats = self.shared_buffer.get_stats()
        valid_samples = stats["valid_samples"]
        total_samples = stats["buffer_size"]
        valid_percentage = stats["valid_percentage"]

        # Calculate uptime
        uptime = current_time - self._dashboard_start_time

        # Determine refresh rate
        refresh_rate = self._last_refresh_rate if status == "GENERATING" else 0

        # Format uptime with more stable display
        if uptime > 60:
            uptime_str = f"{int(uptime/60)}:{int(uptime%60):02d}"
        else:
            uptime_str = f"{int(uptime)}s"

        # Get current device info
        device_str = "GPU" if self.current_device == "cuda" else "CPU"

        # Create shorter dashboard line with fixed-width formatting
        dashboard = (
            f"Buffer: {valid_samples:6,}/{total_samples:,} ({valid_percentage:5.1f}%) | "
            f"Rate: {refresh_rate:4.0f}/s | "
            f"Up: {uptime_str:>6} | "
            f"Device: {device_str} | "
            f"{status:>10}"
        )

        # Use ANSI escape codes to clear line and move cursor to beginning
        print(f"\033[2K\r{dashboard}", end="", flush=True)

    @torch.no_grad()
    def extract_activations(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Extract MLP input/output activations using nnsight tracing.

        Args:
            tokens: Tokenized input [batch, seq_len]

        Returns:
            Activations tensor [batch, seq_len, in/out, n_layer, d_model]
        """
        with self.current_model.trace(tokens) as tracer:
            mlp_ins = []
            mlp_outs = []

            # Extract from all transformer layers
            for i in range(self.config.n_layers):
                # MLP input (after layer norm)
                mlp_in = self.current_model.transformer.h[i].ln_2.input.save()
                mlp_ins.append(mlp_in)

                # MLP output
                mlp_out = self.current_model.transformer.h[i].mlp.output.save()
                mlp_outs.append(mlp_out)

        # Stack: [batch, seq_len, n_layer, d_model]
        mlp_ins = torch.stack(mlp_ins, dim=2)
        mlp_outs = torch.stack(mlp_outs, dim=2)

        # Combine input/output: [batch, seq_len, in/out, n_layer, d_model]
        mlp_acts = torch.stack([mlp_ins, mlp_outs], dim=2)

        # Fuse batch and sequence length dimensions: [batch*seq_len, in/out, n_layer, d_model]
        mlp_acts = mlp_acts.reshape(-1, *mlp_acts.shape[2:])

        return mlp_acts

    def refill_from_file(self) -> int:
        """
        Refill invalid indices in the shared buffer with data from the initialization file.
        Processes in batches of 10000 for memory efficiency.

        Returns:
            Number of indices successfully refilled
        """
        if not self.init_file_tensor:
            logger.warning("No initialization file available for refilling")
            return 0

        # Get all invalid indices
        invalid_indices = self.shared_buffer._get_invalid_indices()
        if len(invalid_indices) == 0:
            logger.info("No invalid indices to refill")
            return 0

        total_refilled = 0
        batch_size = 10000
        total_samples = self.init_file_tensor.shape[0]

        # Process in batches of 10000
        for i in range(0, len(invalid_indices), batch_size):
            batch_indices = invalid_indices[i : i + batch_size]
            samples_requested = len(batch_indices)

            # Handle wraparound if we exceed file size
            if self.init_file_position + samples_requested >= total_samples:
                self.init_file_position = 0

            # Calculate how many we can actually refill in this batch
            available_samples = min(
                samples_requested, total_samples - self.init_file_position
            )
            indices_to_fill = batch_indices[:available_samples]

            # Load the samples from file
            end_pos = self.init_file_position + available_samples
            samples = self.init_file_tensor[self.init_file_position : end_pos]

            # Convert to torch tensor with correct dtype
            activations = torch.from_numpy(samples.astype(np.float32))

            # Set the activations in the shared buffer
            self.shared_buffer.set_activations(indices_to_fill, activations)

            # Update file position and tracking
            self.init_file_position = end_pos
            total_refilled += available_samples

            # Update dashboard after each batch
            self._update_dashboard(f"REFILL {i//batch_size + 1}")

            logger.debug(
                f"Refilled batch {i//batch_size + 1}: {available_samples} samples (total: {total_refilled})"
            )

        logger.info(
            f"Refilled {total_refilled} indices from file in {(len(invalid_indices) + batch_size - 1) // batch_size} batches"
        )
        return total_refilled

    def generate_activations(self) -> torch.Tensor:
        """
        Generate activation data by processing text through the model.

        Returns:
            Tensor of activations [batch, in/out, n_layers, d_model] or None on error
        """
        self._select_device()

        try:
            batch = next(self.text_dataset_loader)
        except StopIteration:
            # Dataset exhausted, recreate loader
            logger.info("Dataset exhausted, recreating loader...")
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

        # Extract activations: [batch*seq_len, in/out, n_layer, d_model]
        mlp_acts = self.extract_activations(batch)

        return mlp_acts

    def cleanup(self):
        """Clean up resources and terminate any child processes."""
        logger.info("Cleaning up data generator...")
        self.running = False

        # Close initialization file if open
        if self.init_file_handle:
            try:
                self.init_file_handle.close()
                logger.info("Initialization file closed")
            except:
                pass
            self.init_file_handle = None
            self.init_file_tensor = None

        # Force cleanup of DataLoader workers
        if (
            hasattr(self, "text_dataset_loader")
            and self.text_dataset_loader is not None
        ):
            try:
                # Try to properly close the DataLoader
                if hasattr(self.text_dataset_loader, "_iterator"):
                    self.text_dataset_loader._iterator = None
            except:
                pass

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
