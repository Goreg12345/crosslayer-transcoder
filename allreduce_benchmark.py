#!/usr/bin/env python3
"""
Simple AllReduce Benchmark - Measure REAL DDP Communication Speed

This script creates tensors matching your model's gradient sizes and measures
actual torch.distributed.all_reduce performance.

Usage:
  torchrun --nproc_per_node=2 allreduce_benchmark.py
  torchrun --nproc_per_node=4 allreduce_benchmark.py
"""

import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed training"""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def create_model_gradients():
    """Create tensors matching your CrossLayerTranscoder gradient sizes"""
    device = torch.cuda.current_device()

    # Your actual gradient tensor sizes (from the analysis)
    tensors = {
        "W_dec": torch.randn(
            12, 12, 6144, 768, device=device, dtype=torch.float32
        ),  # 2592 MB
        "W_enc": torch.randn(
            12, 768, 6144, device=device, dtype=torch.float32
        ),  # 216 MB
        "theta": torch.randn(1, 12, 6144, device=device, dtype=torch.float32),  # 0.3 MB
    }

    # Calculate sizes
    total_elements = sum(t.numel() for t in tensors.values())
    total_mb = total_elements * 4 / (1024 * 1024)  # float32 = 4 bytes

    return tensors, total_mb


def benchmark_allreduce(tensors, num_iterations=20):
    """Benchmark AllReduce on tensors matching your model"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        total_mb = sum(t.numel() * 4 / (1024 * 1024) for t in tensors.values())
        print(
            f"Benchmarking AllReduce with {len(tensors)} tensors, {total_mb:.1f} MB total"
        )
        print(f"World size: {world_size} GPUs")

    # Warmup
    for _ in range(3):
        for tensor in tensors.values():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

    # Benchmark individual tensors
    tensor_times = {}
    for name, tensor in tensors.items():
        times = []
        size_mb = tensor.numel() * 4 / (1024 * 1024)

        for _ in range(num_iterations):
            # Create a copy to avoid accumulation effects
            test_tensor = tensor.clone()

            torch.cuda.synchronize()
            start = time.perf_counter()

            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)

            torch.cuda.synchronize()
            end = time.perf_counter()

            times.append(end - start)

        avg_time = np.mean(times)
        bandwidth_gbps = (size_mb / 1024) / avg_time
        tensor_times[name] = {
            "size_mb": size_mb,
            "time_ms": avg_time * 1000,
            "bandwidth_gbps": bandwidth_gbps,
            "times": times,
        }

        if rank == 0:
            std_time = np.std(times) * 1000
            print(
                f"{name}: {size_mb:.1f} MB → {avg_time*1000:.1f}±{std_time:.1f} ms ({bandwidth_gbps:.1f} GB/s)"
            )

    # Benchmark all tensors together (like in real training)
    all_times = []
    total_mb = sum(t.numel() * 4 / (1024 * 1024) for t in tensors.values())

    for _ in range(num_iterations):
        # Create copies
        test_tensors = {name: tensor.clone() for name, tensor in tensors.items()}

        torch.cuda.synchronize()
        start = time.perf_counter()

        # AllReduce all tensors (simulating DDP gradient sync)
        for tensor in test_tensors.values():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        torch.cuda.synchronize()
        end = time.perf_counter()

        all_times.append(end - start)

    avg_all_time = np.mean(all_times)
    total_bandwidth = (total_mb / 1024) / avg_all_time

    if rank == 0:
        std_all_time = np.std(all_times) * 1000
        print(
            f"\nAll tensors: {total_mb:.1f} MB → {avg_all_time*1000:.1f}±{std_all_time:.1f} ms ({total_bandwidth:.1f} GB/s)"
        )
        print(f"This equals {1/(avg_all_time):.2f} DDP sync operations per second")

    return tensor_times, avg_all_time


def test_tensor_sizes():
    """Test various tensor sizes to understand scaling"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.cuda.current_device()

    if rank == 0:
        print(f"\n=== Tensor Size Scaling Test (World Size: {world_size}) ===")

    # Test different sizes
    test_sizes_mb = [1, 10, 100, 500, 1000, 2000]

    results = []
    for size_mb in test_sizes_mb:
        elements = int(size_mb * 1024 * 1024 / 4)  # float32
        tensor = torch.randn(elements, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(2):
            test_tensor = tensor.clone()
            dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()

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

        avg_time = np.mean(times)
        bandwidth_gbps = (size_mb / 1024) / avg_time

        results.append((size_mb, avg_time * 1000, bandwidth_gbps))

        if rank == 0:
            print(
                f"{size_mb:4d} MB: {avg_time*1000:6.1f} ms ({bandwidth_gbps:5.1f} GB/s)"
            )

    return results


def main():
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print(f"AllReduce Benchmark - World Size: {world_size}")
        print(
            f"GPU Devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
        )
        print(f"Using GPUs: {list(range(world_size))}")

    # Test 1: Your actual model gradient sizes
    if rank == 0:
        print(f"\n=== Model Gradient AllReduce Benchmark ===")

    tensors, total_mb = create_model_gradients()
    tensor_results, total_time = benchmark_allreduce(tensors)

    # Test 2: Size scaling
    size_results = test_tensor_sizes()

    # Summary
    if rank == 0:
        print(f"\n=== Summary for {world_size} GPUs ===")
        print(
            f"Your model gradients ({total_mb:.1f} MB): {total_time*1000:.1f} ms AllReduce"
        )
        print(
            f"This allows {1/total_time:.2f} training iterations per second (communication only)"
        )
        print(f"Effective bandwidth: {(total_mb/1024)/total_time:.1f} GB/s")

        # Compare to your training results
        print(f"\n=== Comparison to Training Results ===")
        print(f"Measured AllReduce time: {total_time*1000:.1f} ms")
        print(f"Your 4-GPU training: 1.4 it/s → {1000/1.4:.0f} ms per iteration")
        print(
            f"Communication overhead: {total_time*1000/(1000/1.4)*100:.1f}% of total iteration time"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
