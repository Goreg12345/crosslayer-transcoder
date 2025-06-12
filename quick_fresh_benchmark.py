#!/usr/bin/env python3
"""
Quick Fresh DataLoader Benchmark

Creates a smaller 2GB dataset for faster testing of real NFS vs SSD performance.
"""

import torch
import time
import shutil
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import h5py
from buffer import DiscBuffer

def create_small_dataset(path, size_gb=2):
    """Create a smaller HDF5 dataset with random data"""
    print(f"Creating {size_gb}GB dataset: {path}")
    
    # Calculate dimensions
    bytes_per_sample = 2 * 12 * 768 * 4  # (2, 12, 768) float32
    target_bytes = size_gb * 1024 * 1024 * 1024
    num_samples = int(target_bytes / bytes_per_sample)
    
    print(f"  Samples: {num_samples:,}")
    
    start_time = time.time()
    
    with h5py.File(path, 'w') as f:
        # Create dataset
        dataset = f.create_dataset('tensor', (num_samples, 2, 12, 768), dtype=np.float32)
        
        # Fill with random data in larger chunks for speed
        chunk_size = 5000
        
        for i in range(0, num_samples, chunk_size):
            end_idx = min(i + chunk_size, num_samples)
            chunk_samples = end_idx - i
            
            chunk_data = np.random.randn(chunk_samples, 2, 12, 768).astype(np.float32)
            dataset[i:end_idx] = chunk_data
    
    create_time = time.time() - start_time
    actual_size_gb = os.path.getsize(path) / (1024**3)
    
    print(f"  Created in {create_time:.1f}s ({actual_size_gb:.2f} GB)")
    return actual_size_gb


def quick_test(data_path, num_workers, test_batches=20):
    """Quick test with fixed number of batches"""
    
    buffer = DiscBuffer(data_path, "tensor")
    
    loader = torch.utils.data.DataLoader(
        buffer,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        batch_size=1000,
        shuffle=True,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
    )
    
    print(f"  {num_workers:2d} workers: ", end="", flush=True)
    
    # Warmup
    loader_iter = iter(loader)
    for _ in range(2):
        next(loader_iter)
    
    # Test
    start_time = time.perf_counter()
    
    for _ in range(test_batches):
        next(loader_iter)
    
    total_time = time.perf_counter() - start_time
    batches_per_second = test_batches / total_time
    samples_per_second = batches_per_second * 1000
    
    print(f"{batches_per_second:5.1f} batches/s ({samples_per_second:7.0f} samples/s)")
    
    del loader, buffer
    return batches_per_second, samples_per_second


def main():
    """Run quick fresh benchmark"""
    
    print("=== Quick Fresh DataLoader Benchmark ===\n")
    
    worker_counts = [1, 2, 4, 8, 16]
    
    # Create small fresh dataset on NFS
    nfs_path = "/var/local/glang/activations/small_fresh_dataset.h5"
    
    print("1. Creating fresh 2GB dataset on NFS...")
    actual_size = create_small_dataset(nfs_path, 2)
    
    # Test NFS
    print(f"\n2. Testing NFS Performance (fresh data):")
    nfs_results = []
    
    for workers in worker_counts:
        try:
            _, samples_ps = quick_test(nfs_path, workers)
            nfs_results.append(samples_ps)
        except Exception as e:
            print(f"  {workers:2d} workers: FAILED - {e}")
            nfs_results.append(0)
    
    # Copy to SSD  
    print(f"\n3. Copying to SSD...")
    temp_dir = tempfile.mkdtemp()
    ssd_path = os.path.join(temp_dir, "dataset.h5")
    
    copy_start = time.time()
    shutil.copy2(nfs_path, ssd_path)
    copy_time = time.time() - copy_start
    print(f"   Copied in {copy_time:.1f}s ({actual_size/copy_time:.1f} GB/s)")
    
    # Test SSD
    print(f"\n4. Testing SSD Performance:")
    ssd_results = []
    
    for workers in worker_counts:
        try:
            _, samples_ps = quick_test(ssd_path, workers)
            ssd_results.append(samples_ps)
        except Exception as e:
            print(f"  {workers:2d} workers: FAILED - {e}")
            ssd_results.append(0)
    
    # Results
    print(f"\n=== Results Summary ===")
    max_nfs = max(nfs_results) if nfs_results else 0
    max_ssd = max(ssd_results) if ssd_results else 0
    
    print(f"Fresh NFS peak: {max_nfs:.0f} samples/s")
    print(f"Fresh SSD peak: {max_ssd:.0f} samples/s")
    
    if max_nfs > 0:
        speedup = max_ssd / max_nfs
        print(f"SSD speedup: {speedup:.1f}x")
    
    print(f"Training speed: 5,600 samples/s")
    
    if max_ssd > 8000:
        print("✓ Fresh data loading still exceeds training")
    else:
        print("⚠ Fresh data loading may be limiting factor")
    
    # Quick plot
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(worker_counts, [s/1000 for s in nfs_results], 'o-', label='Fresh NFS', linewidth=2)
        plt.plot(worker_counts, [s/1000 for s in ssd_results], 's-', label='Fresh SSD', linewidth=2)
        plt.axhline(y=5.6, linestyle='--', label='Training Speed', color='gray')
        plt.xlabel('Workers')
        plt.ylabel('Samples/sec (thousands)')
        plt.title('Fresh DataLoader Performance: NFS vs SSD')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('fresh_benchmark_quick.png', dpi=150, bbox_inches='tight')
        print("   Plot saved: fresh_benchmark_quick.png")
    except Exception as e:
        print(f"   Plot failed: {e}")
    
    # Cleanup
    print(f"\n5. Cleaning up...")
    os.remove(nfs_path)
    shutil.rmtree(temp_dir)
    print("   Done!")


if __name__ == "__main__":
    main()