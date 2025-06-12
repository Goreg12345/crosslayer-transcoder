#!/usr/bin/env python3
"""
Comprehensive AllReduce Bandwidth Scaling Test

Tests if 6.5 GB/s is fundamental system limit or tensor-size dependent.
Runs on all 4 GPUs and tests wide range of tensor sizes with proper cleanup.

Usage: torchrun --nproc_per_node=4 bandwidth_scaling_test.py
"""

import torch
import torch.distributed as dist
import time
import os
import gc
import numpy as np


def setup_distributed():
    """Initialize distributed training"""
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_memory():
    """Aggressive GPU memory cleanup"""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()


def benchmark_tensor_size(size_mb, iterations=15):
    """Benchmark AllReduce for specific tensor size with cleanup"""
    rank = dist.get_rank()
    device = torch.cuda.current_device()
    
    # Calculate elements needed for this size
    elements = int(size_mb * 1024 * 1024 / 4)  # float32 = 4 bytes
    
    # Create tensor
    try:
        tensor = torch.randn(elements, device=device, dtype=torch.float32)
        actual_size_mb = tensor.numel() * 4 / (1024 * 1024)
    except RuntimeError as e:
        if rank == 0:
            print(f"{size_mb:5d} MB: FAILED - {e}")
        return None, None, None
    
    # Warmup
    try:
        for _ in range(3):
            test_tensor = tensor.clone()
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            del test_tensor
    except RuntimeError as e:
        if rank == 0:
            print(f"{size_mb:5d} MB: FAILED during warmup - {e}")
        del tensor
        cleanup_memory()
        return None, None, None
    
    # Benchmark
    times = []
    try:
        for i in range(iterations):
            test_tensor = tensor.clone()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append(end - start)
            del test_tensor
            
            # Periodic cleanup for large tensors
            if i % 5 == 0:
                cleanup_memory()
                
    except RuntimeError as e:
        if rank == 0:
            print(f"{size_mb:5d} MB: FAILED during benchmark - {e}")
        del tensor
        cleanup_memory()
        return None, None, None
    
    # Cleanup
    del tensor
    cleanup_memory()
    
    # Calculate stats
    times = np.array(times)
    avg_time = times.mean()
    std_time = times.std()
    bandwidth_gbps = (actual_size_mb / 1024) / avg_time
    
    return avg_time, std_time, bandwidth_gbps


def test_bandwidth_scaling():
    """Test bandwidth across wide range of tensor sizes"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(f"Bandwidth Scaling Test - {world_size} GPUs")
        print(f"Testing AllReduce bandwidth vs tensor size")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB each")
        print()
    
    # Test sizes from 1 MB to maximum feasible
    # Start small, go up to ~6GB per GPU (24GB total with 4 GPUs leaves room for overhead)
    test_sizes_mb = [
        # Small sizes
        1, 2, 5, 10, 20, 50,
        # Medium sizes  
        100, 200, 500,
        # Large sizes (your model range)
        1000, 2000, 3000, 4000, 5000,
        # Very large sizes (test system limits)
        6000, 7000, 8000
    ]
    
    results = []
    
    if rank == 0:
        print("Size (MB)   Time (ms)   Std (ms)   Bandwidth (GB/s)")
        print("-" * 55)
    
    for size_mb in test_sizes_mb:
        avg_time, std_time, bandwidth = benchmark_tensor_size(size_mb)
        
        if avg_time is not None:
            results.append({
                'size_mb': size_mb,
                'time_ms': avg_time * 1000,
                'std_ms': std_time * 1000,
                'bandwidth_gbps': bandwidth
            })
            
            if rank == 0:
                print(f"{size_mb:5d}       {avg_time*1000:7.1f}     {std_time*1000:5.1f}      {bandwidth:6.1f}")
        
        # Clean up between tests
        cleanup_memory()
        dist.barrier()  # Synchronize all processes
    
    return results


def analyze_results(results):
    """Analyze bandwidth scaling patterns"""
    rank = dist.get_rank()
    
    if rank != 0 or not results:
        return
    
    print(f"\n=== Bandwidth Analysis ===")
    
    # Find bandwidth plateaus
    sizes = [r['size_mb'] for r in results]
    bandwidths = [r['bandwidth_gbps'] for r in results]
    
    # Small tensors (< 100 MB)
    small_bw = [r['bandwidth_gbps'] for r in results if r['size_mb'] < 100]
    medium_bw = [r['bandwidth_gbps'] for r in results if 100 <= r['size_mb'] < 1000]
    large_bw = [r['bandwidth_gbps'] for r in results if r['size_mb'] >= 1000]
    
    if small_bw:
        print(f"Small tensors (<100 MB): {np.mean(small_bw):.1f} ± {np.std(small_bw):.1f} GB/s")
    if medium_bw:
        print(f"Medium tensors (100-1000 MB): {np.mean(medium_bw):.1f} ± {np.std(medium_bw):.1f} GB/s")
    if large_bw:
        print(f"Large tensors (>1000 MB): {np.mean(large_bw):.1f} ± {np.std(large_bw):.1f} GB/s")
    
    # Find peak bandwidth
    max_bw_idx = np.argmax(bandwidths)
    max_bw_result = results[max_bw_idx]
    
    print(f"\nPeak bandwidth: {max_bw_result['bandwidth_gbps']:.1f} GB/s at {max_bw_result['size_mb']} MB")
    
    # Check if bandwidth plateaus
    plateau_threshold = 0.1  # 0.1 GB/s variation
    plateau_sizes = []
    plateau_bws = []
    
    for r in results:
        if r['size_mb'] >= 500:  # Look for plateau in larger sizes
            plateau_sizes.append(r['size_mb'])
            plateau_bws.append(r['bandwidth_gbps'])
    
    if plateau_bws:
        plateau_std = np.std(plateau_bws)
        plateau_mean = np.mean(plateau_bws)
        
        print(f"Large tensor plateau (≥500 MB): {plateau_mean:.1f} ± {plateau_std:.1f} GB/s")
        
        if plateau_std < plateau_threshold:
            print(f"✓ Bandwidth plateaus at {plateau_mean:.1f} GB/s for large tensors")
            print(f"This suggests {plateau_mean:.1f} GB/s is your system's AllReduce limit")
        else:
            print(f"✗ Bandwidth varies significantly ({plateau_std:.1f} GB/s std)")
    
    # Compare to your model
    model_size = 2808
    model_results = [r for r in results if abs(r['size_mb'] - model_size) < 500]
    if model_results:
        closest = min(model_results, key=lambda x: abs(x['size_mb'] - model_size))
        print(f"\nYour model gradients (~{model_size} MB): ~{closest['bandwidth_gbps']:.1f} GB/s")
        print(f"This matches tensor at {closest['size_mb']} MB: {closest['time_ms']:.1f} ms")


def system_info():
    """Print system information"""
    rank = dist.get_rank()
    
    if rank == 0:
        print("=== System Information ===")
        print(f"GPUs: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}")
        
        # Memory info
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.total_memory / (1024**3):.1f} GB, {props.multi_processor_count} SMs")
        
        # NCCL info
        try:
            nccl_version = torch.cuda.nccl.version()
            print(f"NCCL version: {nccl_version}")
        except:
            print("NCCL version: Unknown")
        
        print()


def main():
    rank, local_rank, world_size = setup_distributed()
    
    if world_size != 4:
        if rank == 0:
            print("This script requires exactly 4 GPUs")
        return
    
    system_info()
    results = test_bandwidth_scaling()
    analyze_results(results)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()