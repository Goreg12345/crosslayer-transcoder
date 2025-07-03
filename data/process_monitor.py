"""
Process monitoring and dashboard for the data generation process.
Handles all logging, progress reporting, and CLI dashboard updates.
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ProcessMonitor:
    """
    Handles process monitoring, dashboard updates, and performance tracking.
    Separated from generation logic for clean separation of concerns.
    """

    def __init__(self):
        # Dashboard timing
        self._dashboard_start_time = time.time()
        self._last_dashboard_update = self._dashboard_start_time
        self._last_refresh_rate = 0.0
        self._dashboard_update_interval = 0.5  # Update every 500ms

    def set_refresh_rate(self, refresh_rate: float) -> None:
        """Update the current refresh rate for dashboard display."""
        self._last_refresh_rate = refresh_rate

    def update_dashboard(
        self, status: str, buffer_stats: Dict[str, Any], current_device: str
    ) -> None:
        """
        Update the CLI dashboard with current buffer stats.
        Extracted from existing _update_dashboard method.

        Args:
            status: Current process status (e.g., "GENERATING", "SLEEPING")
            buffer_stats: Statistics from shared buffer
            current_device: Current device ("cpu" or "cuda")
        """
        current_time = time.time()

        # Rate limit dashboard updates
        if current_time - self._last_dashboard_update < self._dashboard_update_interval:
            return

        self._last_dashboard_update = current_time

        # Extract buffer statistics
        valid_samples = buffer_stats["valid_samples"]
        total_samples = buffer_stats["buffer_size"]
        valid_percentage = buffer_stats["valid_percentage"]

        # Calculate uptime and refresh rate
        uptime = current_time - self._dashboard_start_time
        refresh_rate = self._last_refresh_rate

        # Format uptime with more stable display
        if uptime > 60:
            uptime_str = f"{int(uptime/60)}:{int(uptime%60):02d}"
        else:
            uptime_str = f"{int(uptime)}s"

        # Get current device info
        if current_device == "disk":
            device_str = "DISK"
        elif current_device == "cuda":
            device_str = "GPU"
        else:
            device_str = "CPU"

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

    def log_refill_progress(
        self, refilled_count: int, source: str, buffer_stats: Dict[str, Any]
    ) -> None:
        """Log progress for buffer refilling operations and update dashboard."""
        logger.info(f"Refilled {refilled_count} samples from {source}")

        # Update dashboard to show refill status with disk device
        self.update_dashboard("REFILLING", buffer_stats, source)

    def log_dataset_exhausted(self) -> None:
        """Log when dataset is exhausted and needs recreation."""
        logger.info("Dataset exhausted, recreating loader...")

    def log_generation_start(self) -> None:
        """Log start of generation loop."""
        logger.info("Starting generation loop...")

    def log_device_switch(self, from_device: str, to_device: str, reason: str) -> None:
        """Log device switching for performance optimization."""
        logger.info(f"Switching model from {from_device} to {to_device}: {reason}")

    def log_error(self, operation: str, error: Exception) -> None:
        """Log errors during operations."""
        logger.error(f"Error during {operation}: {error}")

    def log_warning(self, message: str) -> None:
        """Log warning messages."""
        logger.warning(message)

    def get_uptime(self) -> float:
        """Get process uptime in seconds."""
        return time.time() - self._dashboard_start_time

    def reset_start_time(self) -> None:
        """Reset the start time (useful for process restarts)."""
        self._dashboard_start_time = time.time()
        self._last_dashboard_update = self._dashboard_start_time


class WandBProcessMonitor(ProcessMonitor):
    """
    Enhanced ProcessMonitor with WandB logging capabilities.
    Creates a separate WandB run grouped with the training run.
    """

    def __init__(
        self,
        project: str = "crosslayer-transcoder",
        group: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[dict] = None,
        save_dir: str = "./wandb",
    ):
        super().__init__()

        # WandB configuration
        self.project = project
        self.group = group
        self.run_name = run_name or "data-generator"
        self.tags = tags or ["data-generation"]
        self.config = config or {}
        self.save_dir = save_dir

        # WandB logging state
        self._wandb_run = None
        self._wandb_available = False
        self._last_wandb_log = time.time()
        self._wandb_log_interval = 5.0  # Log to WandB every 5 seconds

        # Status and device mappings for categorical logging (avoid WandB string/media issues)
        self._status_mapping = {
            "GENERATING": 0,
            "SLEEPING": 1,
            "REFILLING": 2,
            "IDLE": 3,
            "ERROR": 4,
        }

        self._device_mapping = {
            "cpu": 0,
            "cuda": 1,
            "disk": 2,
            "gpu": 1,  # Alias for cuda
        }

        # Initialize WandB
        self._init_wandb()

    def _init_wandb(self) -> None:
        """Initialize WandB run for data generation logging."""
        try:
            import wandb

            # Add mappings to config for reference
            config_with_mappings = self.config.copy()
            config_with_mappings.update(
                {
                    "status_mapping": self._status_mapping,
                    "device_mapping": self._device_mapping,
                }
            )

            # Initialize WandB run
            self._wandb_run = wandb.init(
                project=self.project,
                name=self.run_name,
                group=self.group,
                tags=self.tags,
                config=config_with_mappings,
                dir=self.save_dir,
                job_type="data-generation",
                reinit=True,  # Allow reinit in case wandb is already initialized
            )

            self._wandb_available = True
            logger.info(
                f"WandB initialized for data generation: {self._wandb_run.name}"
            )

        except ImportError:
            logger.warning("WandB not available, skipping WandB logging")
            self._wandb_available = False
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            self._wandb_available = False

    def _log_to_wandb(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to WandB if available."""
        if not self._wandb_available or not self._wandb_run:
            return

        try:
            self._wandb_run.log(metrics)
        except Exception as e:
            logger.error(f"Failed to log to WandB: {e}")

    def update_dashboard(
        self, status: str, buffer_stats: Dict[str, Any], current_device: str
    ) -> None:
        """
        Update dashboard and log metrics to WandB.
        Extends parent method with WandB logging.
        """
        # Call parent method for dashboard update
        super().update_dashboard(status, buffer_stats, current_device)

        # Log to WandB periodically
        current_time = time.time()
        if current_time - self._last_wandb_log >= self._wandb_log_interval:
            self._last_wandb_log = current_time

            # Convert categorical strings to integers to avoid WandB media type issues
            status_code = self._status_mapping.get(status.upper(), -1)
            device_code = self._device_mapping.get(current_device.lower(), -1)

            # Prepare metrics for WandB
            metrics = {
                "data_generation/buffer_valid_samples": buffer_stats["valid_samples"],
                "data_generation/buffer_total_samples": buffer_stats["buffer_size"],
                "data_generation/buffer_fill_percentage": buffer_stats[
                    "valid_percentage"
                ],
                "data_generation/generation_rate_samples_per_sec": self._last_refresh_rate,
                "data_generation/uptime_seconds": self.get_uptime(),
                "data_generation/status_code": status_code,  # Integer instead of string
                "data_generation/device_code": device_code,  # Integer instead of string
            }

            self._log_to_wandb(metrics)

    def log_refill_progress(
        self, refilled_count: int, source: str, buffer_stats: Dict[str, Any]
    ) -> None:
        """Log refill progress to console and WandB."""
        super().log_refill_progress(refilled_count, source, buffer_stats)

        # Convert source to device code for consistent logging
        source_code = self._device_mapping.get(source.lower(), -1)

        # Log refill event to WandB
        metrics = {
            "data_generation/refill_count": refilled_count,
            "data_generation/refill_source_code": source_code,  # Integer instead of string
            "data_generation/buffer_fill_percentage": buffer_stats["valid_percentage"],
        }
        self._log_to_wandb(metrics)

    def log_device_switch(self, from_device: str, to_device: str, reason: str) -> None:
        """Log device switching to console and WandB."""
        super().log_device_switch(from_device, to_device, reason)

        # Convert device names to codes for consistent logging
        from_code = self._device_mapping.get(from_device.lower(), -1)
        to_code = self._device_mapping.get(to_device.lower(), -1)

        # Log device switch event to WandB
        metrics = {
            "data_generation/device_switch_from_code": from_code,  # Integer instead of string
            "data_generation/device_switch_to_code": to_code,  # Integer instead of string
            # Keep reason as string in a separate metric that's logged less frequently
        }
        self._log_to_wandb(metrics)

        # Log the reason separately as a one-time event to avoid string logging issues
        if self._wandb_available and self._wandb_run:
            try:
                # Use wandb.log with commit=False to avoid creating a new step
                import wandb

                wandb.log(
                    {
                        "data_generation/last_device_switch_reason": f"{from_device}->{to_device}: {reason}"
                    },
                    commit=False,
                )
            except Exception as e:
                logger.debug(f"Failed to log device switch reason: {e}")

    def log_error(self, operation: str, error: Exception) -> None:
        """Log errors to console and WandB."""
        super().log_error(operation, error)

        # Log error count instead of strings to avoid WandB media issues
        metrics = {
            "data_generation/error_count": 1,  # Simple counter
        }
        self._log_to_wandb(metrics)

        # Log error details as a one-time event
        if self._wandb_available and self._wandb_run:
            try:
                import wandb

                wandb.log(
                    {
                        "data_generation/last_error": f"{operation}: {str(error)[:200]}..."  # Truncate long errors
                    },
                    commit=False,
                )
            except Exception as e:
                logger.debug(f"Failed to log error details: {e}")

    def finish(self) -> None:
        """Finish WandB run and clean up."""
        if self._wandb_available and self._wandb_run:
            try:
                self._wandb_run.finish()
                logger.info("WandB run finished")
            except Exception as e:
                logger.error(f"Error finishing WandB run: {e}")

    def __del__(self):
        """Ensure WandB run is finished on deletion."""
        self.finish()
