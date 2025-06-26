"""
Process monitoring and dashboard for the data generation process.
Handles all logging, progress reporting, and CLI dashboard updates.
"""

import logging
import time
from typing import Any, Dict

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
