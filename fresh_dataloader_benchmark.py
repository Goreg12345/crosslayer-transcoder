#!/usr/bin/env python3
"""
Fresh DataLoader Benchmark with Uncached Data

Creates a new 10GB HDF5 file with random data to avoid cache effects,
benchmarks NFS vs SSD performance, then cleans up.
"""

import torch
import time
import shutil
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from buffer import DiscBuffer

def create_fresh_dataset(path, size_gb=10):
    """Create a fresh HDF5 dataset with random data"""
    print(f"Creating fresh {size_gb}GB dataset: {path}")
    
    # Calculate dimensions to reach target size
    # Each sample: (2, 12, 768) float32 = 2 * 12 * 768 * 4 bytes = 73,728 bytes
    bytes_per_sample = 2 * 12 * 768 * 4
    target_bytes = size_gb * 1024 * 1024 * 1024
    num_samples = int(target_bytes / bytes_per_sample)
    
    print(f"  Target: {size_gb} GB = {target_bytes:,} bytes")
    print(f"  Samples needed: {num_samples:,} ({bytes_per_sample} bytes each)")
    
    start_time = time.time()
    
    with h5py.File(path, 'w') as f:
        # Create dataset with same structure as your training data
        dataset = f.create_dataset('tensor', (num_samples, 2, 12, 768), dtype=np.float32, chunks=True)
        
        # Fill with random data in chunks to avoid memory issues
        chunk_size = 1000  # Process 1000 samples at a time
        
        for i in range(0, num_samples, chunk_size):
            end_idx = min(i + chunk_size, num_samples)
            chunk_samples = end_idx - i
            
            # Generate random data for this chunk
            chunk_data = np.random.randn(chunk_samples, 2, 12, 768).astype(np.float32)
            dataset[i:end_idx] = chunk_data
            
            if i % (chunk_size * 10) == 0:
                progress = (i / num_samples) * 100
                print(f"  Progress: {progress:.1f}% ({i:,}/{num_samples:,} samples)")
    
    create_time = time.time() - start_time
    actual_size_gb = os.path.getsize(path) / (1024**3)
    
    print(f"  Created in {create_time:.1f}s")
    print(f"  Actual size: {actual_size_gb:.2f} GB")
    print(f"  Write speed: {actual_size_gb/create_time:.1f} GB/s")
    
    return actual_size_gb


def clear_file_cache(filepath):
    """Try to clear file from cache (limited without sudo)"""
    # This won't actually clear the cache without sudo, but we can try
    # to minimize cache effects by creating the file fresh
    pass


def test_dataloader_performance(data_path, num_workers, batch_size=1000, test_time=15):
    """Test dataloader for a fixed time period"""
    
    buffer = DiscBuffer(data_path, "tensor")
    
    loader = torch.utils.data.DataLoader(
        buffer,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
    )
    
    print(f"  {num_workers:2d} workers: ", end="", flush=True)
    
    # Quick warmup
    warmup_count = 3
    loader_iter = iter(loader)
    for i in range(warmup_count):
        try:
            next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            next(loader_iter)
    
    # Benchmark for fixed time
    start_time = time.perf_counter()
    batch_count = 0
    
    try:
        while True:
            next(loader_iter)
            batch_count += 1
            current_time = time.perf_counter()
            if current_time - start_time >= test_time:
                break
    except StopIteration:
        pass
    
    total_time = time.perf_counter() - start_time
    batches_per_second = batch_count / total_time if total_time > 0 else 0
    samples_per_second = batches_per_second * batch_size
    
    print(f"{batches_per_second:5.1f} batches/s ({samples_per_second:7.0f} samples/s)")
    
    del loader, buffer
    return batches_per_second, samples_per_second


def create_plot(results):
    """Create a nice plot"""
    
    # Modern styling
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.alpha': 0.3,
        'legend.frameon': False,
    })
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.patch.set_facecolor('white')
    
    worker_counts = results['worker_counts']
    nfs_samples = [s/1000 for s in results['nfs_samples_per_sec']]
    ssd_samples = [s/1000 for s in results['ssd_samples_per_sec']]
    
    # Colors
    nfs_color = '#e74c3c'  # Red
    ssd_color = '#3498db'  # Blue
    
    # Plot lines with markers
    ax.plot(worker_counts, nfs_samples, 'o-', linewidth=3, markersize=8, 
            label='NFS Storage', color=nfs_color, markerfacecolor='white',
            markeredgewidth=2, markeredgecolor=nfs_color)
    ax.plot(worker_counts, ssd_samples, 's-', linewidth=3, markersize=8, 
            label='SSD Storage', color=ssd_color, markerfacecolor='white',
            markeredgewidth=2, markeredgecolor=ssd_color)
    
    # Training speed reference
    ax.axhline(y=5.6, color='#95a5a6', linestyle='--', linewidth=2,
               label='4-GPU Training (5.6k samples/s)')
    
    ax.set_xlabel('Number of DataLoader Workers', fontsize=14, fontweight='bold')
    ax.set_ylabel('Samples per Second (thousands)', fontsize=14, fontweight='bold')
    ax.set_title('Fresh DataLoader Performance: NFS vs SSD (10GB Dataset)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Annotations for peak values
    max_nfs_idx = np.argmax(nfs_samples)
    max_ssd_idx = np.argmax(ssd_samples)
    
    ax.annotate(f'Peak NFS\n{max(nfs_samples):.1f}k', 
                xy=(worker_counts[max_nfs_idx], nfs_samples[max_nfs_idx]),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=nfs_color, alpha=0.2),
                arrowprops=dict(arrowstyle='->', color=nfs_color))
    
    ax.annotate(f'Peak SSD\n{max(ssd_samples):.1f}k', 
                xy=(worker_counts[max_ssd_idx], ssd_samples[max_ssd_idx]),
                xytext=(10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=ssd_color, alpha=0.2),
                arrowprops=dict(arrowstyle='->', color=ssd_color))
    
    plt.tight_layout()
    return fig


def main():
    """Run fresh benchmark with uncached data"""
    
    print("=== Fresh DataLoader Benchmark (Uncached) ===\n")
    
    worker_counts = [1, 2, 4, 8, 16, 20]
    test_time = 12  # seconds per test
    dataset_size_gb = 10
    
    # Create fresh dataset on NFS
    nfs_dir = "/var/local/glang/activations"
    nfs_path = os.path.join(nfs_dir, "fresh_benchmark_dataset.h5")
    
    print("1. Creating fresh dataset on NFS...")
    actual_size = create_fresh_dataset(nfs_path, dataset_size_gb)
    
    # Test NFS performance
    print(f"\n2. Testing NFS Performance:")
    nfs_results = {'batches_per_sec': [], 'samples_per_sec': []}
    
    for workers in worker_counts:
        try:
            batches_ps, samples_ps = test_dataloader_performance(nfs_path, workers, test_time=test_time)
            nfs_results['batches_per_sec'].append(batches_ps)
            nfs_results['samples_per_sec'].append(samples_ps)
        except Exception as e:
            print(f"  {workers:2d} workers: FAILED - {e}")
            nfs_results['batches_per_sec'].append(0)
            nfs_results['samples_per_sec'].append(0)
    
    # Copy to SSD
    print(f"\n3. Copying fresh dataset to SSD...")
    temp_dir = tempfile.mkdtemp(prefix="fresh_benchmark_")
    ssd_path = os.path.join(temp_dir, "fresh_dataset.h5")
    
    copy_start = time.time()
    shutil.copy2(nfs_path, ssd_path)
    copy_time = time.time() - copy_start
    copy_speed = actual_size / copy_time
    print(f"   Copied {actual_size:.2f} GB in {copy_time:.1f}s ({copy_speed:.1f} GB/s)")
    
    # Test SSD performance
    print(f"\n4. Testing SSD Performance:")
    ssd_results = {'batches_per_sec': [], 'samples_per_sec': []}
    
    for workers in worker_counts:
        try:
            batches_ps, samples_ps = test_dataloader_performance(ssd_path, workers, test_time=test_time)
            ssd_results['batches_per_sec'].append(batches_ps)
            ssd_results['samples_per_sec'].append(samples_ps)
        except Exception as e:
            print(f"  {workers:2d} workers: FAILED - {e}")
            ssd_results['batches_per_sec'].append(0)
            ssd_results['samples_per_sec'].append(0)
    
    # Create plot
    results = {
        'worker_counts': worker_counts,
        'nfs_samples_per_sec': nfs_results['samples_per_sec'],
        'ssd_samples_per_sec': ssd_results['samples_per_sec'],
    }
    
    try:
        print(f"\n5. Creating visualization...")
        fig = create_plot(results)
        output_path = 'fresh_dataloader_benchmark.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   Plot saved as: {output_path}")
    except Exception as e:
        print(f"   Plot creation failed: {e}")
    
    # Summary
    print(f"\n=== Fresh Data Results ===")
    max_nfs = max(nfs_results['samples_per_sec']) if nfs_results['samples_per_sec'] else 0
    max_ssd = max(ssd_results['samples_per_sec']) if ssd_results['samples_per_sec'] else 0
    
    if max_nfs > 0:
        best_nfs_workers = worker_counts[nfs_results['samples_per_sec'].index(max_nfs)]
        speedup = max_ssd / max_nfs if max_nfs > 0 else 0
        
        print(f"Fresh NFS (uncached): {max_nfs:.0f} samples/s ({best_nfs_workers} workers)")
        print(f"Fresh SSD: {max_ssd:.0f} samples/s")
        print(f"SSD speedup: {speedup:.1f}x")
        print(f"Training speed: 5,600 samples/s (4-GPU)")
        
        if max_ssd > 8000:
            print("✓ Even fresh data loading exceeds training speed")
        else:
            print("⚠ Fresh data loading may limit training speed")
    
    # Cleanup
    print(f"\n6. Cleaning up...")
    os.remove(nfs_path)
    print(f"   Removed NFS file: {nfs_path}")
    shutil.rmtree(temp_dir)
    print(f"   Removed SSD files: {temp_dir}")
    print("   Cleanup complete!")


if __name__ == "__main__":
    main()