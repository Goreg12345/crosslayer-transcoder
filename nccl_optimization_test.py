#!/usr/bin/env python3
"""
NCCL Optimization Test - Try to squeeze more performance

Now that we know direct GPU-GPU copy is 8.8 GB/s, let's see if we can:
1. Optimize NCCL settings to get closer to 8.8 GB/s
2. Test different AllReduce algorithms
3. Try batching/pipelining
4. Test overlapping computation with communication

Usage: torchrun --nproc_per_node=4 nccl_optimization_test.py
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


def benchmark_with_nccl_settings(tensor_size_mb=2592, iterations=15):
    """Test AllReduce with different NCCL optimizations"""
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
    std_time = np.std(times)
    bandwidth = (tensor_size_mb / 1024) / avg_time
    
    del tensor
    torch.cuda.empty_cache()
    
    return avg_time, std_time, bandwidth


def test_chunked_allreduce(total_size_mb=2592, chunk_size_mb=256, iterations=10):
    """Test chunked AllReduce to see if smaller chunks are faster"""
    rank = dist.get_rank()
    device = torch.cuda.current_device()
    
    if rank == 0:
        print(f"Testing chunked AllReduce: {total_size_mb} MB in {chunk_size_mb} MB chunks")
    
    num_chunks = total_size_mb // chunk_size_mb
    chunk_elements = int(chunk_size_mb * 1024 * 1024 / 4)
    
    # Create all chunks
    chunks = []
    for _ in range(num_chunks):
        chunks.append(torch.randn(chunk_elements, device=device, dtype=torch.float32))
    
    # Warmup
    for _ in range(2):
        for chunk in chunks:
            test_chunk = chunk.clone()
            dist.all_reduce(test_chunk, op=dist.ReduceOp.SUM)
            del test_chunk
        torch.cuda.synchronize()
    
    # Benchmark chunked approach
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for chunk in chunks:
            test_chunk = chunk.clone()
            dist.all_reduce(test_chunk, op=dist.ReduceOp.SUM)
            del test_chunk
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
    
    avg_time = np.mean(times)
    bandwidth = (total_size_mb / 1024) / avg_time
    
    # Cleanup
    del chunks
    torch.cuda.empty_cache()
    
    return avg_time, bandwidth


def test_overlapped_allreduce(tensor_size_mb=2592, num_streams=4, iterations=10):
    """Test overlapped AllReduce with multiple streams"""
    rank = dist.get_rank()
    device = torch.cuda.current_device()
    
    if rank == 0:
        print(f"Testing overlapped AllReduce with {num_streams} streams")
    
    # Create multiple streams
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    
    chunk_size = tensor_size_mb // num_streams
    chunk_elements = int(chunk_size * 1024 * 1024 / 4)
    
    # Create tensors for each stream
    tensors = []
    for _ in range(num_streams):
        tensors.append(torch.randn(chunk_elements, device=device, dtype=torch.float32))
    
    # Warmup
    for _ in range(2):
        for i, (tensor, stream) in enumerate(zip(tensors, streams)):
            with torch.cuda.stream(stream):
                test_tensor = tensor.clone()
                dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
                del test_tensor
        torch.cuda.synchronize()
    
    # Benchmark overlapped
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Launch AllReduce on all streams
        for i, (tensor, stream) in enumerate(zip(tensors, streams)):
            with torch.cuda.stream(stream):
                test_tensor = tensor.clone()
                dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
                del test_tensor
        
        # Wait for all streams
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
    
    avg_time = np.mean(times)
    bandwidth = (tensor_size_mb / 1024) / avg_time
    
    # Cleanup
    del tensors
    torch.cuda.empty_cache()
    
    return avg_time, bandwidth


def test_different_data_types():
    """Test if data type affects AllReduce performance"""
    rank = dist.get_rank()
    device = torch.cuda.current_device()
    
    if rank == 0:
        print("Testing different data types:")
    
    size_mb = 1000
    results = {}
    
    dtypes = [
        (torch.float32, "FP32"),
        (torch.float16, "FP16"), 
        (torch.bfloat16, "BF16"),
    ]
    
    for dtype, name in dtypes:
        try:
            element_size = torch.tensor([], dtype=dtype).element_size()
            elements = int(size_mb * 1024 * 1024 / element_size)
            
            tensor = torch.randn(elements, device=device, dtype=dtype)
            
            # Warmup
            for _ in range(3):
                test_tensor = tensor.clone()
                dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
                del test_tensor
            
            # Benchmark
            times = []
            for _ in range(10):
                test_tensor = tensor.clone()
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
                
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                times.append(end - start)
                del test_tensor
            
            avg_time = np.mean(times)
            bandwidth = (size_mb / 1024) / avg_time
            
            results[name] = bandwidth
            
            if rank == 0:
                print(f"  {name}: {bandwidth:.1f} GB/s")
            
            del tensor
            torch.cuda.empty_cache()
            
        except Exception as e:
            if rank == 0:
                print(f"  {name}: FAILED - {e}")
    
    return results


def main():
    rank, local_rank, world_size = setup_distributed()
    
    if world_size != 4:
        if rank == 0:
            print("This script requires 4 GPUs")
        return
    
    if rank == 0:
        print("=== NCCL Optimization Tests ===")
        print("Trying to improve on the 6.3 GB/s AllReduce baseline\n")
    
    # Test 1: Baseline (current settings)
    if rank == 0:
        print("1. Baseline AllReduce:")
    avg_time, std_time, bandwidth = benchmark_with_nccl_settings()
    if rank == 0:
        print(f"   {bandwidth:.1f} ± {(std_time/avg_time)*bandwidth:.1f} GB/s\n")
    
    # Test 2: Different data types  
    if rank == 0:
        print("2. Data type comparison:")
    test_different_data_types()
    print()
    
    # Test 3: Chunked AllReduce
    if rank == 0:
        print("3. Chunked AllReduce:")
    
    chunk_sizes = [128, 256, 512, 1024]
    for chunk_size in chunk_sizes:
        avg_time, bandwidth = test_chunked_allreduce(chunk_size_mb=chunk_size)
        if rank == 0:
            print(f"   {chunk_size} MB chunks: {bandwidth:.1f} GB/s")
    print()
    
    # Test 4: Overlapped streams
    if rank == 0:
        print("4. Overlapped streams:")
    
    for num_streams in [2, 4, 8]:
        try:
            avg_time, bandwidth = test_overlapped_allreduce(num_streams=num_streams)
            if rank == 0:
                print(f"   {num_streams} streams: {bandwidth:.1f} GB/s")
        except Exception as e:
            if rank == 0:
                print(f"   {num_streams} streams: FAILED - {e}")
    
    if rank == 0:
        print(f"\n=== Optimization Summary ===")
        print(f"Direct GPU-GPU copy limit: ~8.8 GB/s")
        print(f"Current AllReduce: ~6.3 GB/s")
        print(f"Gap to close: {8.8 - 6.3:.1f} GB/s ({((8.8-6.3)/6.3)*100:.0f}% improvement possible)")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()