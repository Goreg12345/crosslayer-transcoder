#!/usr/bin/env python3
"""
Deep Bottleneck Analysis - Find the real limiting factor

The 126 GB/s -> 6.3 GB/s drop is suspicious. Let's investigate:
1. CPU-GPU memory transfer speeds
2. Host memory bandwidth
3. NUMA cross-socket bandwidth
4. PCIe utilization during AllReduce
5. Memory allocation patterns
6. NCCL configuration issues
"""

import torch
import time
import numpy as np
import os
import psutil
import subprocess


def test_host_to_device_bandwidth():
    """Test CPU-GPU memory transfer speeds"""
    print("=== Host-Device Memory Bandwidth Test ===")
    
    device = torch.cuda.current_device()
    sizes_mb = [100, 1000, 2000, 5000]
    
    for size_mb in sizes_mb:
        elements = int(size_mb * 1024 * 1024 / 4)
        
        # Host to Device
        host_tensor = torch.randn(elements, dtype=torch.float32, pin_memory=True)
        device_tensor = torch.empty(elements, dtype=torch.float32, device=device)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        device_tensor.copy_(host_tensor, non_blocking=True)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        h2d_bandwidth = (size_mb / 1024) / (end - start)
        
        # Device to Host
        torch.cuda.synchronize()
        start = time.perf_counter()
        host_tensor.copy_(device_tensor, non_blocking=True)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        d2h_bandwidth = (size_mb / 1024) / (end - start)
        
        print(f"{size_mb:4d} MB: H2D {h2d_bandwidth:5.1f} GB/s, D2H {d2h_bandwidth:5.1f} GB/s")
        
        del host_tensor, device_tensor
        torch.cuda.empty_cache()


def test_device_to_device_direct():
    """Test direct GPU-GPU memory copies (not AllReduce)"""
    print("\n=== Direct GPU-GPU Memory Copy Test ===")
    
    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs")
        return
    
    sizes_mb = [100, 1000, 2000, 5000]
    
    for size_mb in sizes_mb:
        elements = int(size_mb * 1024 * 1024 / 4)
        
        try:
            # Create tensors on different GPUs
            src_tensor = torch.randn(elements, device='cuda:0', dtype=torch.float32)
            dst_tensor = torch.empty(elements, device='cuda:1', dtype=torch.float32)
            
            # Warmup
            for _ in range(3):
                dst_tensor.copy_(src_tensor, non_blocking=True)
                torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for _ in range(10):
                torch.cuda.synchronize()
                start = time.perf_counter()
                dst_tensor.copy_(src_tensor, non_blocking=True)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times)
            bandwidth = (size_mb / 1024) / avg_time
            
            print(f"{size_mb:4d} MB: GPU0->GPU1 {bandwidth:5.1f} GB/s ({avg_time*1000:.1f} ms)")
            
            del src_tensor, dst_tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{size_mb:4d} MB: FAILED - {e}")


def test_numa_memory_bandwidth():
    """Test cross-NUMA CPU memory bandwidth"""
    print("\n=== NUMA Memory Bandwidth Test ===")
    
    try:
        # Use CPU tensors to test NUMA memory
        size_mb = 1000
        elements = int(size_mb * 1024 * 1024 / 4)
        
        # Create large CPU tensors
        tensor1 = torch.randn(elements, dtype=torch.float32)
        tensor2 = torch.empty(elements, dtype=torch.float32)
        
        # Memory copy test
        start = time.perf_counter()
        tensor2.copy_(tensor1)
        end = time.perf_counter()
        
        bandwidth = (size_mb / 1024) / (end - start)
        print(f"CPU memory copy: {bandwidth:.1f} GB/s")
        
        # Memory addition test (more realistic workload)
        tensor3 = torch.randn(elements, dtype=torch.float32)
        
        start = time.perf_counter()
        result = tensor1 + tensor2 + tensor3
        end = time.perf_counter()
        
        # 3 reads + 1 write = 4x data movement
        bandwidth_compute = (size_mb * 4 / 1024) / (end - start)
        print(f"CPU memory compute: {bandwidth_compute:.1f} GB/s")
        
        del tensor1, tensor2, tensor3, result
        
    except Exception as e:
        print(f"NUMA test failed: {e}")


def analyze_pcie_topology():
    """Analyze PCIe topology in detail"""
    print("\n=== Detailed PCIe Topology Analysis ===")
    
    try:
        # Get detailed lspci info for each GPU
        gpus = ['25:00.0', '5b:00.0', '9b:00.0', 'c8:00.0']
        
        for i, gpu_bus in enumerate(gpus):
            print(f"\nGPU {i} (Bus {gpu_bus}):")
            
            # Get PCIe tree path
            try:
                result = subprocess.run(['lspci', '-tv'], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if gpu_bus in line:
                            print(f"  PCIe tree: {line.strip()}")
                            break
            except:
                pass
            
            # Get bandwidth info
            try:
                result = subprocess.run(['lspci', '-vvv', '-s', gpu_bus], capture_output=True, text=True)
                if result.returncode == 0:
                    output = result.stdout
                    
                    # Find LnkCap and LnkSta
                    lnkcap_match = re.search(r'LnkCap:.*?Speed ([^,]+).*?Width (x\d+)', output)
                    if lnkcap_match:
                        print(f"  Link Capability: {lnkcap_match.group(1)} {lnkcap_match.group(2)}")
                    
                    lnksta_match = re.search(r'LnkSta:.*?Speed ([^,]+).*?Width (x\d+)', output)
                    if lnksta_match:
                        print(f"  Link Status: {lnksta_match.group(1)} {lnksta_match.group(2)}")
            except:
                pass
                
    except Exception as e:
        print(f"PCIe analysis failed: {e}")


def test_nccl_specific_bottlenecks():
    """Test for NCCL-specific issues"""
    print("\n=== NCCL Configuration Analysis ===")
    
    # Check NCCL environment variables
    nccl_vars = [
        'NCCL_DEBUG', 'NCCL_ALGO', 'NCCL_PROTO', 'NCCL_MIN_NCHANNELS',
        'NCCL_MAX_NCHANNELS', 'NCCL_TREE_THRESHOLD', 'NCCL_NET_GDR_LEVEL',
        'NCCL_P2P_LEVEL', 'NCCL_SHM_DISABLE', 'NCCL_SOCKET_NTHREADS'
    ]
    
    print("NCCL Environment Variables:")
    for var in nccl_vars:
        value = os.environ.get(var, "Not set")
        print(f"  {var}: {value}")
    
    # Test if P2P is actually working
    print(f"\nP2P Access Test:")
    for i in range(torch.cuda.device_count()):
        for j in range(torch.cuda.device_count()):
            if i != j:
                can_access = torch.cuda.can_device_access_peer(i, j)
                print(f"  GPU{i} -> GPU{j}: {'✓' if can_access else '✗'}")


def test_memory_allocation_patterns():
    """Test if memory allocation affects performance"""
    print("\n=== Memory Allocation Pattern Test ===")
    
    device = torch.cuda.current_device()
    size_mb = 2000
    elements = int(size_mb * 1024 * 1024 / 4)
    
    print("Testing different allocation patterns:")
    
    # Test 1: Single large allocation
    print("1. Single large tensor:")
    tensor = torch.randn(elements, device=device, dtype=torch.float32)
    
    start = time.perf_counter()
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    print(f"   Allocation time: {(end-start)*1000:.1f} ms")
    del tensor
    torch.cuda.empty_cache()
    
    # Test 2: Multiple smaller allocations
    print("2. Multiple smaller tensors:")
    num_chunks = 10
    chunk_elements = elements // num_chunks
    tensors = []
    
    start = time.perf_counter()
    for _ in range(num_chunks):
        tensors.append(torch.randn(chunk_elements, device=device, dtype=torch.float32))
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    print(f"   Allocation time: {(end-start)*1000:.1f} ms")
    del tensors
    torch.cuda.empty_cache()
    
    # Test 3: Pinned memory allocation
    print("3. Pinned host memory:")
    start = time.perf_counter()
    host_tensor = torch.randn(elements, dtype=torch.float32, pin_memory=True)
    end = time.perf_counter()
    
    print(f"   Allocation time: {(end-start)*1000:.1f} ms")
    del host_tensor


def test_concurrent_operations():
    """Test if concurrent GPU operations affect bandwidth"""
    print("\n=== Concurrent Operations Test ===")
    
    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs")
        return
    
    size_mb = 1000
    elements = int(size_mb * 1024 * 1024 / 4)
    
    # Test 1: Sequential operations
    print("1. Sequential GPU operations:")
    tensor0 = torch.randn(elements, device='cuda:0', dtype=torch.float32)
    tensor1 = torch.empty(elements, device='cuda:1', dtype=torch.float32)
    
    start = time.perf_counter()
    tensor1.copy_(tensor0, non_blocking=True)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    seq_time = end - start
    seq_bandwidth = (size_mb / 1024) / seq_time
    print(f"   Sequential: {seq_bandwidth:.1f} GB/s")
    
    # Test 2: Concurrent with computation
    print("2. Concurrent with computation:")
    compute_tensor = torch.randn(elements, device='cuda:0', dtype=torch.float32)
    
    start = time.perf_counter()
    
    # Start copy
    tensor1.copy_(tensor0, non_blocking=True)
    
    # Do computation while copying
    for _ in range(10):
        compute_tensor = compute_tensor * 1.1
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    concurrent_time = end - start
    print(f"   With computation: {concurrent_time:.3f}s")
    
    del tensor0, tensor1, compute_tensor
    torch.cuda.empty_cache()


def main():
    print("=== Deep Bottleneck Analysis ===")
    print("Investigating the 126 GB/s -> 6.3 GB/s drop\n")
    
    # Run all tests
    test_host_to_device_bandwidth()
    test_device_to_device_direct()
    test_numa_memory_bandwidth()
    analyze_pcie_topology()
    test_nccl_specific_bottlenecks()
    test_memory_allocation_patterns()
    test_concurrent_operations()
    
    print("\n=== Summary and Hypothesis ===")
    print("Key areas to investigate:")
    print("1. Host-Device bandwidth vs AllReduce bandwidth")
    print("2. Direct GPU-GPU copy vs NCCL AllReduce")
    print("3. NUMA memory subsystem performance")
    print("4. PCIe lane sharing/contention")
    print("5. NCCL algorithm selection")
    print("6. Memory allocation overhead")


if __name__ == "__main__":
    import re
    main()