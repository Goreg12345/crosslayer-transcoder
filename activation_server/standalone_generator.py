#!/usr/bin/env python3
"""
Standalone data generator that runs independently of FastAPI workers.
Creates and manages shared activation buffer that multiple server workers can access.
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

from activation_server.config import ServerConfig, get_production_config
from activation_server.data_generator import DataGeneratorProcess
from activation_server.shared_memory import SharedActivationBuffer

logger = logging.getLogger(__name__)


class StandaloneDataGenerator:
    """Standalone data generator with shared memory management."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.shared_buffer = None
        self.data_generator_process = None
        self.running = False

        # Store shared memory info for server workers to access
        self.shm_info_file = Path("/tmp/activation_server_shm_info.json")

    def start(self):
        """Start the data generator and shared memory."""
        if self.running:
            logger.warning("Data generator already running")
            return

        logger.info("Initializing shared memory buffer...")
        self.shared_buffer = SharedActivationBuffer(
            buffer_size=self.config.buffer_size,
            n_in_out=self.config.n_in_out,
            n_layers=self.config.n_layers,
            activation_dim=self.config.activation_dim,
            dtype=self.config.dtype,
        )

        # Save shared memory info for server workers
        shm_info = {
            "buffer_size": self.config.buffer_size,
            "n_in_out": self.config.n_in_out,
            "n_layers": self.config.n_layers,
            "activation_dim": self.config.activation_dim,
            "dtype": str(self.config.dtype),
            "pid": os.getpid(),
            "started_at": time.time(),
        }

        with open(self.shm_info_file, "w") as f:
            json.dump(shm_info, f)

        logger.info(f"Shared memory info saved to {self.shm_info_file}")

        logger.info("Starting data generator process...")
        self.data_generator_process = DataGeneratorProcess(
            shared_buffer=self.shared_buffer, config=self.config
        )
        self.data_generator_process.start()

        self.running = True
        logger.info("Standalone data generator started successfully")

    def stop(self):
        """Stop the data generator and cleanup."""
        if not self.running:
            return

        logger.info("Stopping standalone data generator...")

        if self.data_generator_process and self.data_generator_process.is_alive():
            self.data_generator_process.terminate()
            self.data_generator_process.join(timeout=10)
            if self.data_generator_process.is_alive():
                logger.warning("Force killing data generator process")
                self.data_generator_process.kill()

        if self.shared_buffer:
            self.shared_buffer.cleanup()

        # Clean up shared memory info file
        if self.shm_info_file.exists():
            self.shm_info_file.unlink()

        self.running = False
        logger.info("Standalone data generator stopped")

    def run_forever(self):
        """Run the data generator until interrupted."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.stop()
            sys.exit(0)

        # Handle shutdown signals
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        self.start()

        try:
            logger.info("Data generator running. Press Ctrl+C to stop.")
            while self.running:
                time.sleep(1)

                # Optionally print stats
                if hasattr(self.shared_buffer, "get_stats"):
                    stats = self.shared_buffer.get_stats()
                    if int(time.time()) % 30 == 0:  # Every 30 seconds
                        logger.info(
                            f"Buffer stats: {stats['valid_samples']}/{stats['buffer_size']} valid samples"
                        )

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Standalone Activation Data Generator")
    parser.add_argument(
        "--config",
        choices=["production", "test"],
        default="production",
        help="Configuration profile to use",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get configuration
    if args.config == "production":
        config = get_production_config()
    else:
        from activation_server.config import get_test_config

        config = get_test_config()

    # Create and run generator
    generator = StandaloneDataGenerator(config)
    generator.run_forever()


if __name__ == "__main__":
    main()
