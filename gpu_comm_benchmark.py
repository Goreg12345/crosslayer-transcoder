#!/usr/bin/env python3
"""
GPU Communication Benchmark for DDP Training

This script benchmarks:
1. P2P bandwidth between all GPU pairs
2. All-reduce communication patterns (similar to DDP)
3. Memory copy performance
4. NCCL topology information

No sudo required - uses PyTorch CUDA APIs.
"""

import torch
import torch.distributed as dist
import time
import numpy as np
from typing import List, Tuple
import subprocess
import os
import argparse


def get_gpu_info():
    """Get basic GPU information"""
    print("=== GPU Information ===")
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  - Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  - SM Count: {props.multi_processor_count}")
        print(f"  - Compute Capability: {props.major}.{props.minor}")
    
    return num_gpus


def benchmark_p2p_bandwidth(src_gpu: int, dst_gpu: int, sizes_mb: List[int]) -> List[float]:
    """Benchmark P2P bandwidth between two GPUs"""
    bandwidths = []
    
    for size_mb in sizes_mb:
        size_bytes = size_mb * 1024 * 1024
        size_elements = size_bytes // 4  # float32
        
        # Create tensors on source GPU
        with torch.cuda.device(src_gpu):
            src_tensor = torch.randn(size_elements, dtype=torch.float32, device=f'cuda:{src_gpu}')
        
        with torch.cuda.device(dst_gpu):
            dst_tensor = torch.empty(size_elements, dtype=torch.float32, device=f'cuda:{dst_gpu}')
        
        # Warmup
        for _ in range(5):
            dst_tensor.copy_(src_tensor, non_blocking=True)
            torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        num_iterations = max(10, 100 // size_mb)  # More iterations for smaller sizes
        for _ in range(num_iterations):
            dst_tensor.copy_(src_tensor, non_blocking=True)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        total_bytes = size_bytes * num_iterations
        bandwidth_gbps = (total_bytes / total_time) / (1024**3)
        bandwidths.append(bandwidth_gbps)
    
    return bandwidths


def test_p2p_access():
    """Test P2P access capabilities between GPUs"""
    print("\n=== P2P Access Test ===")
    num_gpus = torch.cuda.device_count()
    
    print("P2P Access Matrix (can GPU i access GPU j directly?):")
    print("     ", end="")
    for j in range(num_gpus):
        print(f"GPU{j:2d}", end=" ")
    print()
    
    for i in range(num_gpus):
        print(f"GPU{i:2d}:", end=" ")
        for j in range(num_gpus):
            if i == j:
                print("  --", end=" ")
            else:
                try:
                    can_access = torch.cuda.can_device_access_peer(i, j)
                    print("  ✓ " if can_access else "  ✗ ", end=" ")
                except:
                    print("  ? ", end=" ")
        print()


def benchmark_all_p2p(sizes_mb: List[int] = [1, 10, 100, 500]):
    """Benchmark P2P bandwidth between all GPU pairs"""
    print(f"\n=== P2P Bandwidth Test (sizes: {sizes_mb} MB) ===")
    num_gpus = torch.cuda.device_count()
    
    if num_gpus < 2:
        print("Need at least 2 GPUs for P2P testing")
        return
    
    results = {}
    
    for i in range(num_gpus):
        for j in range(num_gpus):
            if i != j:
                try:
                    bandwidths = benchmark_p2p_bandwidth(i, j, sizes_mb)
                    results[(i, j)] = bandwidths
                    print(f"GPU {i} -> GPU {j}: {bandwidths[-1]:.1f} GB/s (largest size)")
                except Exception as e:
                    print(f"GPU {i} -> GPU {j}: ERROR - {e}")
                    results[(i, j)] = [0.0] * len(sizes_mb)
    
    # Print detailed results table
    print(f"\nDetailed P2P Bandwidth Results:")
    print("Size (MB):", " ".join(f"{s:>8}" for s in sizes_mb))
    for i in range(num_gpus):
        for j in range(num_gpus):
            if i != j and (i, j) in results:
                bandwidths = results[(i, j)]
                print(f"GPU{i}->GPU{j}:", " ".join(f"{b:>8.1f}" for b in bandwidths))
    
    return results


def benchmark_allreduce_pattern(tensor_size_mb: int, num_iterations: int = 50):
    """Benchmark all-reduce pattern similar to DDP"""
    print(f"\n=== All-Reduce Benchmark ({tensor_size_mb} MB tensor) ===")
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("Need at least 2 GPUs for all-reduce testing")
        return
    
    size_elements = (tensor_size_mb * 1024 * 1024) // 4  # float32
    
    # Create tensors on each GPU
    tensors = []
    for i in range(num_gpus):
        with torch.cuda.device(i):
            tensor = torch.randn(size_elements, dtype=torch.float32, device=f'cuda:{i}')
            tensors.append(tensor)
    
    # Manual all-reduce implementation (sum across all GPUs)
    def manual_allreduce():
        # Sum all tensors to GPU 0
        with torch.cuda.device(0):
            result = tensors[0].clone()
            for i in range(1, num_gpus):
                temp = torch.empty_like(result, device='cuda:0')
                temp.copy_(tensors[i], non_blocking=True)
                result += temp
        
        # Broadcast result back to all GPUs
        for i in range(1, num_gpus):
            tensors[i].copy_(result, non_blocking=True)
        
        torch.cuda.synchronize()
    
    # Warmup
    for _ in range(5):
        manual_allreduce()
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for _ in range(num_iterations):
        manual_allreduce()
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iterations) * 1000
    
    # Calculate theoretical bandwidth
    # Each all-reduce moves tensor_size * (num_gpus - 1) data for gather + broadcast
    bytes_per_iteration = tensor_size_mb * 1024 * 1024 * (num_gpus - 1) * 2
    bandwidth_gbps = (bytes_per_iteration / (avg_time_ms / 1000)) / (1024**3)
    
    print(f"All-reduce time: {avg_time_ms:.2f} ms")
    print(f"Theoretical bandwidth: {bandwidth_gbps:.1f} GB/s")
    
    return avg_time_ms, bandwidth_gbps


def check_nccl_support():
    """Check NCCL availability and version"""
    print("\n=== NCCL Information ===")
    try:
        # Try to import and check if NCCL backend is available
        if torch.distributed.is_nccl_available():
            print("✓ NCCL backend is available")
            
            # Try to get NCCL version (may not work without distributed init)
            try:
                version = torch.cuda.nccl.version()
                print(f"NCCL version: {version}")
            except:
                print("NCCL version info not available (need distributed init)")
        else:
            print("✗ NCCL backend not available")
    except Exception as e:
        print(f"Error checking NCCL: {e}")


def get_topology_info():
    """Get GPU topology information"""
    print("\n=== GPU Topology Information ===")
    
    try:
        # Try to run nvidia-smi topo -m (may not work without sudo)
        result = subprocess.run(['nvidia-smi', 'topo', '-m'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("nvidia-smi topology matrix:")
            print(result.stdout)
        else:
            print("nvidia-smi topo not available (may need sudo)")
    except:
        print("nvidia-smi topo command not available")
    
    # Alternative: Check PCIe info through CUDA
    num_gpus = torch.cuda.device_count()
    print(f"\nPCIe Bus IDs:")
    for i in range(num_gpus):
        try:
            props = torch.cuda.get_device_properties(i)
            # PCIe info might be in the name or we can infer from device index
            print(f"GPU {i}: {props.name}")
        except:
            print(f"GPU {i}: Info not available")


def memory_bandwidth_test(gpu_id: int, size_mb: int = 1000):
    """Test memory bandwidth on a single GPU"""
    print(f"\n=== GPU {gpu_id} Memory Bandwidth Test ({size_mb} MB) ===")
    
    size_elements = (size_mb * 1024 * 1024) // 4
    
    with torch.cuda.device(gpu_id):
        # Test different memory operations
        src = torch.randn(size_elements, dtype=torch.float32, device=f'cuda:{gpu_id}')
        dst = torch.empty_like(src)
        
        # Copy test
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            dst.copy_(src)
        torch.cuda.synchronize()
        copy_time = (time.perf_counter() - start) / 10
        copy_bandwidth = (size_mb / copy_time) / 1024  # GB/s
        
        # Add test
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            result = src + dst
        torch.cuda.synchronize()
        add_time = (time.perf_counter() - start) / 10
        add_bandwidth = (size_mb * 2 / add_time) / 1024  # GB/s (read src + dst)
        
        print(f"Memory copy bandwidth: {copy_bandwidth:.1f} GB/s")
        print(f"Memory add bandwidth: {add_bandwidth:.1f} GB/s")


def main():
    parser = argparse.ArgumentParser(description='GPU Communication Benchmark')
    parser.add_argument('--sizes', nargs='+', type=int, default=[1, 10, 100, 500],
                       help='Transfer sizes in MB (default: 1 10 100 500)')
    parser.add_argument('--allreduce-size', type=int, default=100,
                       help='All-reduce tensor size in MB (default: 100)')
    parser.add_argument('--skip-p2p', action='store_true',
                       help='Skip P2P bandwidth tests')
    parser.add_argument('--skip-allreduce', action='store_true',
                       help='Skip all-reduce tests')
    
    args = parser.parse_args()
    
    print("GPU Communication Benchmark")
    print("=" * 50)
    
    # Basic GPU info
    num_gpus = get_gpu_info()
    if not num_gpus:
        return
    
    # Check NCCL support
    check_nccl_support()
    
    # Get topology info
    get_topology_info()
    
    # Test P2P access
    test_p2p_access()
    
    # Memory bandwidth test for each GPU
    for i in range(min(num_gpus, 2)):  # Test first 2 GPUs
        memory_bandwidth_test(i)
    
    if not args.skip_p2p:
        # P2P bandwidth tests
        benchmark_all_p2p(args.sizes)
    
    if not args.skip_allreduce:
        # All-reduce pattern test
        benchmark_allreduce_pattern(args.allreduce_size)
    
    print("\n=== Summary ===")
    print("Benchmark complete! Key insights for DDP:")
    print("1. Check P2P access matrix - direct GPU-GPU transfers are faster")
    print("2. P2P bandwidth shows communication speed between GPU pairs")
    print("3. All-reduce time indicates DDP gradient synchronization performance")
    print("4. Higher bandwidth = faster distributed training")


if __name__ == "__main__":
    main()