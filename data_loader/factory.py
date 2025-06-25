"""
Factory functions for creating activation data loaders.
Handles all IO and dependency injection at the top level.
"""

import logging
from typing import Optional

from data_loader.config import DataLoaderConfig
from data_loader.data_generator import DataGeneratorProcess
from data_loader.dataset import SharedMemoryDataLoader, SharedMemoryDataset
from data_loader.shared_memory import SharedActivationBuffer

logger = logging.getLogger(__name__)


def actvs_loader_from_config(
    config: Optional[DataLoaderConfig] = None,
    batch_size: int = 1000,
    timeout: float = 30.0,
) -> SharedMemoryDataLoader:
    """
    Factory function to create a complete activation data loader.
    Handles all IO and dependency creation at the top level.

    Args:
        config: Data loader configuration (uses default if None)
        batch_size: Batch size for the data loader
        timeout: Timeout for data operations

    Returns:
        SharedMemoryDataLoader ready for use
    """
    if config is None:
        config = DataLoaderConfig()

    config.validate()

    logger.info("Creating activation data loader...")
    logger.info(f"Buffer size: {config.buffer_size:,} samples")
    logger.info(f"Estimated memory: {config.get_memory_estimate_gb():.2f} GB")

    # 1. Create shared memory buffer
    shared_buffer = SharedActivationBuffer(
        buffer_size=config.buffer_size,
        n_in_out=config.n_in_out,
        n_layers=config.n_layers,
        activation_dim=config.activation_dim,
        dtype=config.dtype,
        config=config,
    )

    # 2. Create data generator (models will be created inside the process)
    data_generator = DataGeneratorProcess(
        shared_buffer=shared_buffer,
        config=config,
    )

    # 3. Start the data generator process
    print("Starting data generator process...")
    data_generator.start()
    print("Data generator process started")

    # 4. Create shared memory dataset
    dataset = SharedMemoryDataset(shared_buffer, config)

    # 5. Create and return the data loader
    return SharedMemoryDataLoader(
        shared_buffer=shared_buffer,
        dataset=dataset,
        data_generator=data_generator,
        batch_size=batch_size,
        config=config,
    )


def actvs_loader_from_test_config(batch_size: int = 1000) -> SharedMemoryDataLoader:
    """
    Convenience function to create a test activation data loader.

    Args:
        batch_size: Batch size for the data loader

    Returns:
        SharedMemoryDataLoader configured for testing
    """
    from data_loader.config import get_test_config

    return actvs_loader_from_config(get_test_config(), batch_size)


def actvs_loader_from_production_config(
    batch_size: int = 5000,
    buffer_size: Optional[int] = None,
    generation_batch_size: Optional[int] = None,
    timeout: float = 30.0,
) -> SharedMemoryDataLoader:
    """
    Convenience function to create a production activation data loader.

    Args:
        batch_size: Batch size for the data loader
        buffer_size: Override buffer size (uses config default if None)
        generation_batch_size: Override generation batch size (uses config default if None)

    Returns:
        SharedMemoryDataLoader configured for production
    """
    from data_loader.config import get_production_config

    config = get_production_config()

    # Apply overrides if provided
    if buffer_size is not None:
        config.buffer_size = buffer_size
    if generation_batch_size is not None:
        config.generation_batch_size = generation_batch_size

    return actvs_loader_from_config(config, batch_size)
