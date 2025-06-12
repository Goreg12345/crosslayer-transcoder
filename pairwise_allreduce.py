#!/usr/bin/env python3
"""
Pairwise AllReduce Benchmark - Test specific GPU combinations

Usage:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 pairwise_allreduce.py
  CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 pairwise_allreduce.py
  CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 pairwise_allreduce.py
  etc.
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


def benchmark_model_gradients():
    """Benchmark AllReduce for your model's gradient size"""
    rank = dist.get_rank()
    device = torch.cuda.current_device()
    
    # Your largest gradient tensor (W_dec)
    tensor = torch.randn(12, 12, 6144, 768, device=device, dtype=torch.float32)
    size_mb = tensor.numel() * 4 / (1024 * 1024)
    
    if rank == 0:
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
        print(f"Testing GPUs: {visible_devices}")
        print(f"Tensor size: {size_mb:.1f} MB (W_dec equivalent)")
    
    # Warmup
    for _ in range(3):
        test_tensor = tensor.clone()
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for i in range(20):
        test_tensor = tensor.clone()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
        
        if rank == 0 and i % 5 == 0:
            print(f"Iteration {i+1}: {(end-start)*1000:.1f} ms")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    bandwidth_gbps = (size_mb / 1024) / avg_time
    
    if rank == 0:
        print(f"\nResults for GPUs {visible_devices}:")
        print(f"Average time: {avg_time*1000:.1f} ± {std_time*1000:.1f} ms")
        print(f"Bandwidth: {bandwidth_gbps:.1f} GB/s")
        print(f"AllReduce rate: {1/avg_time:.2f} ops/sec")
    
    return avg_time, bandwidth_gbps


def main():
    rank, local_rank, world_size = setup_distributed()
    
    if world_size != 2:
        if rank == 0:
            print("This script is designed for pairwise testing (2 GPUs)")
        return
    
    benchmark_model_gradients()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()