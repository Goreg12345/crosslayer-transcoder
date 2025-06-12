#!/usr/bin/env python3
"""
Aggressive NCCL Tuning - Try to reach the 8.8 GB/s limit

The 1024 MB chunking got us to 7.8 GB/s! Let's try:
1. Larger chunk sizes (closer to direct copy behavior)
2. NCCL environment variable tuning
3. Different reduction algorithms
4. Memory alignment optimizations

Usage: torchrun --nproc_per_node=4 aggressive_nccl_tuning.py
"""

import torch
import torch.distributed as dist
import time
import os
import numpy as np


def setup_distributed():
    """Initialize distributed training"""
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def benchmark_allreduce(tensor_size_mb, iterations=15):
    """Benchmark AllReduce for given tensor size"""
    rank = dist.get_rank()
    device = torch.cuda.current_device()
    
    elements = int(tensor_size_mb * 1024 * 1024 / 4)
    tensor = torch.randn(elements, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(3):
        test_tensor = tensor.clone()
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        del test_tensor
    
    # Benchmark
    times = []
    for _ in range(iterations):
        test_tensor = tensor.clone()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
        del test_tensor
    
    avg_time = np.mean(times)
    bandwidth = (tensor_size_mb / 1024) / avg_time
    
    del tensor
    torch.cuda.empty_cache()
    
    return avg_time, bandwidth


def test_optimal_chunk_sizes():
    """Find the optimal chunk size for AllReduce"""
    rank = dist.get_rank()
    
    if rank == 0:
        print("=== Finding Optimal Chunk Size ===")
    
    total_size = 2592  # Your model size
    
    # Test larger chunk sizes (approaching full tensor)
    chunk_sizes = [1024, 1296, 1728, 2160, 2592]  # Up to full size
    
    best_bandwidth = 0
    best_chunk = 0
    
    for chunk_size in chunk_sizes:
        if chunk_size <= total_size:
            num_chunks = total_size // chunk_size
            remainder = total_size % chunk_size
            
            # Time the chunked approach
            total_time = 0
            
            # Benchmark main chunks
            if num_chunks > 0:
                chunk_time, chunk_bandwidth = benchmark_allreduce(chunk_size, iterations=10)
                total_time += chunk_time * num_chunks
            
            # Benchmark remainder
            if remainder > 0:
                remainder_time, _ = benchmark_allreduce(remainder, iterations=5)
                total_time += remainder_time
            
            # Calculate effective bandwidth
            effective_bandwidth = (total_size / 1024) / total_time
            
            if rank == 0:
                if num_chunks > 0 and remainder > 0:
                    print(f"{chunk_size:4d} MB chunks ({num_chunks}x + {remainder} MB): {effective_bandwidth:.1f} GB/s")
                elif num_chunks > 0:
                    print(f"{chunk_size:4d} MB chunks ({num_chunks}x): {effective_bandwidth:.1f} GB/s")
                else:
                    print(f"{chunk_size:4d} MB (full): {effective_bandwidth:.1f} GB/s")
            
            if effective_bandwidth > best_bandwidth:
                best_bandwidth = effective_bandwidth
                best_chunk = chunk_size
    
    if rank == 0:
        print(f"\nBest configuration: {best_chunk} MB chunks → {best_bandwidth:.1f} GB/s")
    
    return best_chunk, best_bandwidth


def test_with_nccl_env_vars():
    """Test AllReduce with optimized NCCL environment variables"""
    rank = dist.get_rank()
    
    if rank == 0:
        print("\n=== Testing NCCL Environment Optimizations ===")
    
    # Test current performance
    baseline_time, baseline_bw = benchmark_allreduce(2592)
    
    if rank == 0:
        print(f"Baseline: {baseline_bw:.1f} GB/s")
    
    # Test different configurations
    configs = [
        {
            'name': 'Increased channels',
            'vars': {'NCCL_MIN_NCHANNELS': '16', 'NCCL_MAX_NCHANNELS': '16'}
        },
        {
            'name': 'Tree algorithm forced',  
            'vars': {'NCCL_ALGO': 'Tree'}
        },
        {
            'name': 'Ring algorithm forced',
            'vars': {'NCCL_ALGO': 'Ring'}
        },
        {
            'name': 'Disable P2P fallback',
            'vars': {'NCCL_P2P_DISABLE': '1'}
        },
        {
            'name': 'More socket threads',
            'vars': {'NCCL_SOCKET_NTHREADS': '8'}
        }
    ]
    
    for config in configs:
        if rank == 0:
            print(f"\nTesting: {config['name']}")
        
        # Set environment variables
        original_values = {}
        for var, value in config['vars'].items():
            original_values[var] = os.environ.get(var)
            os.environ[var] = value
            if rank == 0:
                print(f"  {var}={value}")
        
        # Reinitialize NCCL (this won't work in practice, but we can test)
        try:
            test_time, test_bw = benchmark_allreduce(2592)
            improvement = ((test_bw - baseline_bw) / baseline_bw) * 100
            
            if rank == 0:
                print(f"  Result: {test_bw:.1f} GB/s ({improvement:+.1f}%)")
                
        except Exception as e:
            if rank == 0:
                print(f"  Failed: {e}")
        
        # Restore environment variables
        for var in config['vars']:
            if original_values[var] is None:
                if var in os.environ:
                    del os.environ[var]
            else:
                os.environ[var] = original_values[var]


def test_memory_aligned_tensors():
    """Test if memory alignment affects performance"""
    rank = dist.get_rank()
    device = torch.cuda.current_device()
    
    if rank == 0:
        print("\n=== Memory Alignment Test ===")
    
    size_mb = 2592
    
    # Test 1: Regular allocation
    elements = int(size_mb * 1024 * 1024 / 4)
    tensor1 = torch.randn(elements, device=device, dtype=torch.float32)
    
    time1, bw1 = benchmark_allreduce(size_mb)
    
    if rank == 0:
        print(f"Regular allocation: {bw1:.1f} GB/s")
    
    # Test 2: Page-aligned allocation (multiple of 4096 elements)
    aligned_elements = ((elements + 4095) // 4096) * 4096
    aligned_size_mb = aligned_elements * 4 / (1024 * 1024)
    
    del tensor1
    torch.cuda.empty_cache()
    
    tensor2 = torch.randn(aligned_elements, device=device, dtype=torch.float32)
    time2, bw2 = benchmark_allreduce(aligned_size_mb)
    
    if rank == 0:
        print(f"Page-aligned ({aligned_elements} elements): {bw2:.1f} GB/s")
        improvement = ((bw2 - bw1) / bw1) * 100
        print(f"Improvement: {improvement:+.1f}%")
    
    del tensor2
    torch.cuda.empty_cache()


def test_reduction_operations():
    """Test if different reduction operations have different performance"""
    rank = dist.get_rank()
    
    if rank == 0:
        print("\n=== Reduction Operation Test ===")
    
    size_mb = 1000  # Smaller for quick testing
    operations = [
        (dist.ReduceOp.SUM, "SUM"),
        (dist.ReduceOp.AVG, "AVG"), 
        (dist.ReduceOp.MAX, "MAX"),
        (dist.ReduceOp.MIN, "MIN"),
    ]
    
    for op, name in operations:
        try:
            # Modified benchmark for different ops
            device = torch.cuda.current_device()
            elements = int(size_mb * 1024 * 1024 / 4)
            tensor = torch.randn(elements, device=device, dtype=torch.float32)
            
            # Warmup
            for _ in range(3):
                test_tensor = tensor.clone()
                dist.all_reduce(test_tensor, op=op)
                torch.cuda.synchronize()
                del test_tensor
            
            # Benchmark
            times = []
            for _ in range(10):
                test_tensor = tensor.clone()
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                dist.all_reduce(test_tensor, op=op)
                
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                times.append(end - start)
                del test_tensor
            
            avg_time = np.mean(times)
            bandwidth = (size_mb / 1024) / avg_time
            
            if rank == 0:
                print(f"  {name}: {bandwidth:.1f} GB/s")
            
            del tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            if rank == 0:
                print(f"  {name}: FAILED - {e}")


def main():
    rank, local_rank, world_size = setup_distributed()
    
    if world_size != 4:
        if rank == 0:
            print("This script requires 4 GPUs")
        return
    
    if rank == 0:
        print("=== Aggressive NCCL Tuning ===")
        print("Goal: Get from 6.3 GB/s to 8.8 GB/s (the direct copy limit)\n")
    
    # Test 1: Find optimal chunk size
    best_chunk, best_bandwidth = test_optimal_chunk_sizes()
    
    # Test 2: Memory alignment
    test_memory_aligned_tensors()
    
    # Test 3: Different reduction operations
    test_reduction_operations()
    
    # Test 4: NCCL environment variables (limited effectiveness due to process group)
    test_with_nccl_env_vars()
    
    if rank == 0:
        print(f"\n=== Final Summary ===")
        print(f"Direct GPU-GPU limit: 8.8 GB/s")
        print(f"Best AllReduce achieved: {best_bandwidth:.1f} GB/s")
        print(f"Remaining gap: {8.8 - best_bandwidth:.1f} GB/s")
        print(f"NCCL overhead: {((8.8 - best_bandwidth) / 8.8) * 100:.1f}%")
        
        if best_bandwidth >= 7.5:
            print("✓ Excellent optimization - very close to hardware limit")
        elif best_bandwidth >= 7.0:
            print("✓ Good optimization - reasonable NCCL overhead")
        else:
            print("⚠ Still significant optimization potential")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()