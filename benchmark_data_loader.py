#!/usr/bin/env python3
"""
Benchmark script to test the SharedMemoryDataLoader performance.
Tests throughput in MB/s for 10 seconds with batch size 5000.
"""

import time
import torch
import numpy as np

from data_loader import SharedMemoryDataLoader, get_test_config


def benchmark_shared_memory_loader():
    """Benchmark the SharedMemoryDataLoader with production config."""
    print("🚀 Benchmarking SharedMemoryDataLoader...")
    print("   Configuration: PRODUCTION")
    print("   Initialization wait: 30 seconds")
    print("   Test duration: 40 seconds")
    print("   Batch size: 5000")
    
    # Get production configuration
    from data_loader import get_production_config
    config = get_production_config()
    config.generation_batch_size = 32  # Reasonable generation batch
    
    batch_size = 5000
    init_wait = 30.0  # seconds
    duration = 40.0  # seconds
    
    try:
        with SharedMemoryDataLoader(
            config=config,
            batch_size=batch_size,
            timeout=180.0  # Longer timeout for production
        ) as loader:
            
            print(f"   Waiting {init_wait}s for data generator initialization...")
            
            # Wait for initialization period
            time.sleep(init_wait)
            
            # Get first batch to establish shape and calculate memory per batch
            first_batch = next(iter(loader))
            print(f"   First batch loaded!")
            print(f"   Batch shape: {first_batch.shape}")
            print(f"   Batch dtype: {first_batch.dtype}")
            
            # Calculate bytes per batch
            bytes_per_element = first_batch.element_size()
            elements_per_batch = first_batch.numel()
            bytes_per_batch = bytes_per_element * elements_per_batch
            mb_per_batch = bytes_per_batch / (1024 * 1024)
            
            print(f"   Memory per batch: {mb_per_batch:.2f} MB")
            print(f"   Elements per batch: {elements_per_batch:,}")
            
            # Benchmark for 40 seconds
            print(f"\n   Starting {duration}s benchmark...")
            start_time = time.time()
            batch_count = 0
            total_bytes = 0
            
            while time.time() - start_time < duration:
                batch_start = time.time()
                batch = next(iter(loader))
                batch_time = time.time() - batch_start
                
                batch_count += 1
                total_bytes += bytes_per_batch
                
                if batch_count % 5 == 0:
                    elapsed = time.time() - start_time
                    current_mb_s = (total_bytes / (1024 * 1024)) / elapsed
                    print(f"   Batch {batch_count}: {batch_time*1000:.1f}ms, "
                          f"Running avg: {current_mb_s:.1f} MB/s")
            
            # Final calculations
            total_time = time.time() - start_time
            total_mb = total_bytes / (1024 * 1024)
            mb_per_second = total_mb / total_time
            samples_per_second = (batch_count * batch_size) / total_time
            
            print(f"\n📊 RESULTS:")
            print(f"   Duration: {total_time:.2f}s")
            print(f"   Batches loaded: {batch_count}")
            print(f"   Total samples: {batch_count * batch_size:,}")
            print(f"   Total data: {total_mb:.1f} MB")
            print(f"   Throughput: {mb_per_second:.1f} MB/s")
            print(f"   Sample rate: {samples_per_second:,.0f} samples/s")
            
            # Show buffer stats
            stats = loader.get_stats()
            print(f"\n📈 BUFFER STATS:")
            print(f"   Buffer size: {stats['buffer_size']:,} samples")
            print(f"   Valid samples: {stats['valid_samples']:,} ({stats['valid_percentage']:.1f}%)")
            print(f"   Buffer memory: {stats['total_memory_gb']:.2f} GB")
            
            return mb_per_second
            
    except Exception as e:
        print(f"   ❌ SharedMemoryDataLoader failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main benchmark function."""
    print("🔥 SharedMemoryDataLoader Benchmark")
    print("=" * 50)
    
    mb_per_s = benchmark_shared_memory_loader()
    
    if mb_per_s:
        print(f"\n✨ Final Result: {mb_per_s:.1f} MB/s")
    else:
        print("\n❌ Benchmark failed")


if __name__ == "__main__":
    main()