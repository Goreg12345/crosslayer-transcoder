#!/usr/bin/env python3
"""
Comprehensive data loader benchmark for ActivationDataModule.
Tests both shared memory and simple buffer modes.
"""
import os
import sys
import time
from typing import Any, Dict, Optional

# Add project root to Python path for reliable imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)
print(f"Working directory: {os.getcwd()}")

import torch

from crosslayer_transcoder.data.datamodule import ActivationDataModule


def benchmark_loader(loader, mode_name: str, duration: float = 30.0) -> float:
    """
    Generic benchmark function for any loader.

    Args:
        loader: Data loader to benchmark
        mode_name: Name of the mode being tested
        duration: How long to run the benchmark in seconds

    Returns:
        Throughput in MB/s
    """
    print(f"\n📊 Benchmarking {mode_name}")
    print(f"   Duration: {duration}s")

    # Get first batch for measurements
    first_batch = next(iter(loader))
    batch_shape = first_batch.shape
    batch_size = batch_shape[0]

    bytes_per_element = first_batch.element_size()
    elements_per_batch = first_batch.numel()
    bytes_per_batch = bytes_per_element * elements_per_batch
    mb_per_batch = bytes_per_batch / (1024 * 1024)

    print(f"   Batch shape: {batch_shape}")
    print(f"   Batch dtype: {first_batch.dtype}")
    print(f"   Memory per batch: {mb_per_batch:.2f} MB")
    print(f"   Elements per batch: {elements_per_batch:,}")

    # Wait for buffer to fill (for shared memory modes)
    if hasattr(loader, "get_stats"):
        print("   Waiting 30s for shared memory buffer to fill...")
        time.sleep(30)
        stats = loader.get_stats()
        print(f"   Buffer status: {stats['valid_percentage']:.1f}% full")
    else:
        print("   No buffer stats available (likely simple buffer mode)")
        time.sleep(5)  # Short wait for simple buffer

    # Run benchmark
    start_time = time.time()
    batch_count = 0
    total_bytes = 0

    print(f"   Starting {duration}s benchmark...")
    loader = iter(loader)

    try:
        while time.time() - start_time < duration:
            batch_start = time.time()
            _ = next(loader)
            batch_time = time.time() - batch_start

            batch_count += 1
            total_bytes += bytes_per_batch

            if batch_count % 5 == 0:
                elapsed = time.time() - start_time
                current_mb_s = (total_bytes / (1024 * 1024)) / elapsed
                print(
                    f"   Batch {batch_count}: {batch_time*1000:.1f}ms, "
                    f"Running avg: {current_mb_s:.1f} MB/s"
                )

    except Exception as e:
        print(f"   ⚠️  Benchmark stopped early due to: {e}")

    # Calculate results
    total_time = time.time() - start_time
    total_mb = total_bytes / (1024 * 1024)
    mb_per_second = total_mb / total_time if total_time > 0 else 0
    samples_per_second = (batch_count * batch_size) / total_time if total_time > 0 else 0

    print(f"\n   ✨ {mode_name} Results:")
    print(f"   Actual duration: {total_time:.2f}s")
    print(f"   Batches loaded: {batch_count}")
    print(f"   Total samples: {batch_count * batch_size:,}")
    print(f"   Total data: {total_mb:.1f} MB")
    print(f"   Throughput: {mb_per_second:.1f} MB/s")
    print(f"   Sample rate: {samples_per_second:,.0f} samples/s")

    # Show buffer stats if available
    if hasattr(loader, "get_stats"):
        stats = loader.get_stats()
        print(f"   Final buffer status:")
        print(f"     Valid samples: {stats['valid_samples']:,} ({stats['valid_percentage']:.1f}%)")
        print(f"     Buffer memory: {stats['total_memory_gb']:.2f} GB")

    return mb_per_second


def benchmark_datamodule_shared_memory(batch_size: int = 5000, duration: float = 30.0) -> Optional[float]:
    """
    Benchmark using ActivationDataModule with shared memory mode.

    Args:
        batch_size: Batch size for testing
        duration: Benchmark duration in seconds

    Returns:
        Throughput in MB/s or None if failed
    """
    print("\n1️⃣ DATAMODULE SHARED MEMORY MODE")
    print("   Using ActivationDataModule with use_shared_memory=True")

    datamodule = None
    try:
        # Create DataModule (production-like usage)
        datamodule = ActivationDataModule(
            # Buffer settings
            buffer_size=2_000_000,  # 2M samples for faster startup
            n_in_out=2,
            n_layers=12,
            activation_dim=768,
            dtype="float16",
            # Model settings
            model_name="openai-community/gpt2",
            model_dtype="float32",
            # Dataset settings
            dataset_name="Skylion007/openwebtext",
            dataset_split="train",
            max_sequence_length=1024,
            generation_batch_size=32,
            refresh_interval=0.1,
            # File paths
            init_file="/var/local/glang/activations/clt-activations-10M-shuffled_fp16.h5",
            # DataLoader settings
            batch_size=batch_size,
            num_workers=0,  # Disable for cleaner benchmarking
            shuffle=False,  # Disable for consistent timing
            persistent_workers=False,
            pin_memory=True,
            # Enable shared memory mode
            use_shared_memory=True,
            shared_memory_name="benchmark_buffer",
            timeout_seconds=60,
        )

        print(f"   Estimated memory usage: {datamodule.get_memory_estimate_gb():.2f} GB")

        # Setup (Lightning lifecycle)
        print("   Setting up DataModule...")
        datamodule.setup("fit")

        # Get data loader
        loader = datamodule.train_dataloader()
        print("   DataModule setup complete")

        result = benchmark_loader(loader, "DataModule SharedMemory", duration)
        return result

    except Exception as e:
        print(f"   ❌ DataModule shared memory benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        if datamodule:
            print("   Cleaning up DataModule...")
            datamodule.teardown("fit")


def benchmark_datamodule_simple_buffer(batch_size: int = 5000, duration: float = 30.0) -> Optional[float]:
    """
    Benchmark using ActivationDataModule with simple buffer mode.

    Args:
        batch_size: Batch size for testing
        duration: Benchmark duration in seconds

    Returns:
        Throughput in MB/s or None if failed
    """
    print("\n2️⃣ DATAMODULE SIMPLE BUFFER MODE")
    print("   Using ActivationDataModule with use_shared_memory=False")

    datamodule = None
    try:
        datamodule = ActivationDataModule(
            # DataLoader settings
            batch_size=batch_size,
            num_workers=20,  # Can use workers in simple mode
            prefetch_factor=2,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True,
            # File path (required for simple buffer mode)
            init_file="/var/local/glang/activations/clt-activations-10M-shuffled_fp16.h5",
            # Disable shared memory mode
            use_shared_memory=False,
            # Other settings (not used in simple mode but required for validation)
            buffer_size=1000,  # Minimal value
            n_in_out=2,
            n_layers=12,
            activation_dim=768,
            dtype="float16",
        )

        print("   Setting up DataModule...")
        datamodule.setup("fit")

        loader = datamodule.train_dataloader()
        print("   DataModule setup complete")

        result = benchmark_loader(loader, "DataModule SimpleBuffer", duration)
        return result

    except Exception as e:
        print(f"   ❌ DataModule simple buffer benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        if datamodule:
            print("   Cleaning up DataModule...")
            datamodule.teardown("fit")


def main():
    """Run comprehensive benchmark comparing DataModule approaches."""
    print("🔥 ActivationDataModule Benchmark")
    print("   Testing: Shared Memory vs Simple Buffer modes")
    print("=" * 60)

    # Configuration
    batch_size = 5000
    duration = 30.0

    print(f"⚙️  Configuration:")
    print(f"   Batch size: {batch_size:,}")
    print(f"   Test duration: {duration}s per mode")
    print(f"   PyTorch version: {torch.__version__}")

    results: Dict[str, float] = {}

    # 1. DataModule shared memory benchmark
    result = benchmark_datamodule_shared_memory(batch_size, duration)
    if result:
        results["DataModule Shared"] = result

    # 2. DataModule simple buffer benchmark
    result = benchmark_datamodule_simple_buffer(batch_size, duration)
    if result:
        results["DataModule Simple"] = result

    # Final comparison
    print(f"\n🏆 FINAL COMPARISON")
    print("=" * 60)
    if results:
        max_name_len = max(len(name) for name in results.keys())
        for mode, mb_s in results.items():
            print(f"   {mode:<{max_name_len}} {mb_s:8.1f} MB/s")

        # Performance analysis
        if len(results) > 1:
            best_mode = max(results.items(), key=lambda x: x[1])
            print(f"\n🥇 Best performing: {best_mode[0]} ({best_mode[1]:.1f} MB/s)")

            shared_result = results.get("DataModule Shared")
            simple_result = results.get("DataModule Simple")

            if shared_result and simple_result:
                if shared_result > simple_result:
                    improvement = (shared_result - simple_result) / simple_result * 100
                    print(f"   Shared memory is {improvement:+.1f}% faster than simple buffer")
                else:
                    improvement = (simple_result - shared_result) / shared_result * 100
                    print(f"   Simple buffer is {improvement:+.1f}% faster than shared memory")
    else:
        print("   ❌ No successful benchmarks completed")

    print(f"\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
