#!/usr/bin/env python3
"""
Main entry point for the activation server.
"""

import argparse
import logging
import sys
from typing import Optional

from activation_server.config import (
    ServerConfig,
    get_production_config,
    get_test_config,
)
from activation_server.server import run_server


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("activation_server.log"),
        ],
    )


def main():
    """Main function to run the activation server."""
    parser = argparse.ArgumentParser(description="Activation Server")

    parser.add_argument(
        "--config",
        choices=["test", "production", "env"],
        default="test",
        help="Configuration preset to use",
    )

    parser.add_argument("--host", default=None, help="Host address to bind to")

    parser.add_argument("--port", type=int, default=None, help="Port to bind to")

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument(
        "--mock", action="store_true", help="Use mock data generator for testing"
    )

    parser.add_argument("--buffer-size", type=int, help="Override buffer size")

    parser.add_argument(
        "--activation-dim", type=int, help="Override activation dimension"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Get configuration
    if args.config == "test":
        config = get_test_config()
    elif args.config == "production":
        config = get_production_config()
    elif args.config == "env":
        config = ServerConfig.from_env()
    else:
        config = ServerConfig()

    # Apply command line overrides
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port

    if args.buffer_size:
        config.buffer_size = args.buffer_size
    if args.activation_dim:
        config.activation_dim = args.activation_dim

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Log configuration
    logger.info(f"Starting activation server with config: {args.config}")
    logger.info(f"Buffer size: {config.buffer_size:,} samples")
    logger.info(f"Activation dimension: {config.activation_dim}")
    logger.info(f"Estimated memory usage: {config.get_memory_estimate_gb():.2f} GB")
    logger.info(f"Host: {config.host}:{config.port}")

    # Start server
    try:
        run_server(
            host=config.host,
            port=config.port,
            reload=False,  # Don't use reload in production
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
