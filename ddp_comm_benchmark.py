#!/usr/bin/env python3
"""
DDP Communication Benchmark - Accurate Training Workload Simulation

This script measures DDP communication overhead by:
1. Creating the exact model architecture used in training
2. Simulating forward/backward passes with real gradient sizes
3. Measuring NCCL allreduce times for actual gradient tensors
4. Comparing single-GPU vs multi-GPU efficiency
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
from collections import defaultdict
import numpy as np

# Import your actual model
from clt import CrossLayerTranscoder
from jumprelu import JumpReLU


def setup_ddp(rank, world_size):
    """Initialize DDP process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP"""
    dist.destroy_process_group()


def create_model():
    """Create the exact same model as in training"""
    model = CrossLayerTranscoder(
        config={
            "d_acts": 768,
            "d_features": 768 * 8,
            "n_layers": 12,
            "lambda": 0.0002,
            "c": 0.1,
            "lr": 1e-3,
        },
        nonlinearity=JumpReLU(theta=0.03, bandwidth=1.0, n_layers=12, d_features=768 * 8),
    )
    return model


def benchmark_gradient_communication(rank, world_size, batch_size=4000, num_iterations=10):
    """Benchmark actual gradient communication for your model"""
    setup_ddp(rank, world_size)
    
    # Create model on this GPU
    model = create_model().cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # Create dummy data matching your training: (batch_size, 2, n_layers, d_acts)
    # batch[:, 0] = residual stream, batch[:, 1] = MLP output
    batch_data = torch.randn(batch_size, 2, 12, 768, device=rank, dtype=torch.float32)
    
    # Warmup
    for _ in range(3):
        # Normalize like in training
        mean = batch_data.mean(dim=-1, keepdim=True)
        std = batch_data.std(dim=-1, keepdim=True)
        acts_norm = (batch_data - mean) / std
        resid, mlp_out = acts_norm[:, 0], batch_data[:, 1]
        
        features, recons = model(resid)
        loss = ((recons - mlp_out) ** 2).mean()  # Simple MSE for benchmark
        loss.backward()
        model.zero_grad()
    
    # Collect gradient sizes for analysis
    if rank == 0:
        grad_sizes = []
        total_params = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                size_mb = param.grad.numel() * param.grad.element_size() / (1024 * 1024)
                grad_sizes.append((name, param.grad.shape, size_mb))
                total_params += param.grad.numel()
        
        print(f"Model has {total_params:,} parameters")
        print(f"Total gradient size: {total_params * 2 / (1024*1024):.1f} MB (FP16)")
        print("\nLargest gradient tensors:")
        grad_sizes.sort(key=lambda x: x[2], reverse=True)
        for name, shape, size_mb in grad_sizes[:5]:
            print(f"  {name}: {shape} -> {size_mb:.1f} MB")
    
    # Benchmark communication timing
    times = []
    torch.cuda.synchronize()
    
    for i in range(num_iterations):
        start_time = time.perf_counter()
        
        # Forward pass
        mean = batch_data.mean(dim=-1, keepdim=True)
        std = batch_data.std(dim=-1, keepdim=True)
        acts_norm = (batch_data - mean) / std
        resid, mlp_out = acts_norm[:, 0], batch_data[:, 1]
        
        features, recons = model(resid)
        loss = ((recons - mlp_out) ** 2).mean()
        
        # Backward pass (triggers DDP communication)
        loss.backward()
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        iteration_time = end_time - start_time
        times.append(iteration_time)
        
        if rank == 0:
            print(f"Iteration {i+1}: {iteration_time:.3f}s ({1/iteration_time:.2f} it/s)")
        
        model.zero_grad()
    
    # Analyze results
    if rank == 0:
        times = np.array(times)
        print(f"\n=== DDP Communication Benchmark Results ===")
        print(f"World size: {world_size} GPUs")
        print(f"Batch size: {batch_size}")
        print(f"Iterations: {num_iterations}")
        print(f"Average time per iteration: {times.mean():.3f}s")
        print(f"Average throughput: {1/times.mean():.2f} it/s")
        print(f"Std deviation: {times.std():.3f}s")
        print(f"Min time: {times.min():.3f}s ({1/times.min():.2f} it/s)")
        print(f"Max time: {times.max():.3f}s ({1/times.max():.2f} it/s)")
    
    cleanup_ddp()
    return times.mean() if rank == 0 else None


def benchmark_pure_allreduce(rank, world_size, tensor_size_mb=300, num_iterations=50):
    """Benchmark pure NCCL allreduce with gradient-sized tensors"""
    setup_ddp(rank, world_size)
    
    # Create tensor matching gradient size
    tensor_elements = int(tensor_size_mb * 1024 * 1024 / 2)  # FP16
    tensor = torch.randn(tensor_elements, device=rank, dtype=torch.float32)
    
    # Warmup
    for _ in range(5):
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for i in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        dist.all_reduce(tensor)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
    
    if rank == 0:
        times = np.array(times)
        bandwidth_gbps = (tensor_size_mb / 1024) / times.mean()
        print(f"\n=== Pure AllReduce Benchmark ===")
        print(f"Tensor size: {tensor_size_mb} MB")
        print(f"World size: {world_size}")
        print(f"Average allreduce time: {times.mean()*1000:.2f} ms")
        print(f"Effective bandwidth: {bandwidth_gbps:.1f} GB/s")
        print(f"Min time: {times.min()*1000:.2f} ms")
        print(f"Max time: {times.max()*1000:.2f} ms")
    
    cleanup_ddp()
    return times.mean() if rank == 0 else None


def compare_single_vs_multi_gpu():
    """Compare single GPU vs multi-GPU training speed"""
    print("=== Single GPU Baseline ===")
    
    # Single GPU test
    model = create_model().cuda(0)
    batch_data = torch.randn(1000, 2, 12, 768, device=0, dtype=torch.float32)  # Single GPU batch
    
    # Warmup
    for _ in range(3):
        mean = batch_data.mean(dim=-1, keepdim=True)
        std = batch_data.std(dim=-1, keepdim=True)
        acts_norm = (batch_data - mean) / std
        resid, mlp_out = acts_norm[:, 0], batch_data[:, 1]
        
        features, recons = model(resid)
        loss = ((recons - mlp_out) ** 2).mean()
        loss.backward()
        model.zero_grad()
    
    # Benchmark
    times = []
    for i in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        mean = batch_data.mean(dim=-1, keepdim=True)
        std = batch_data.std(dim=-1, keepdim=True)
        acts_norm = (batch_data - mean) / std
        resid, mlp_out = acts_norm[:, 0], batch_data[:, 1]
        
        features, recons = model(resid)
        loss = ((recons - mlp_out) ** 2).mean()
        loss.backward()
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
        model.zero_grad()
    
    single_gpu_time = np.mean(times)
    print(f"Single GPU (batch=1000): {single_gpu_time:.3f}s ({1/single_gpu_time:.2f} it/s)")
    
    return single_gpu_time


def run_ddp_benchmark(world_size):
    """Run DDP benchmark with specified world size"""
    mp.spawn(benchmark_gradient_communication, 
             args=(world_size, 4000, 10), 
             nprocs=world_size, 
             join=True)
    
    mp.spawn(benchmark_pure_allreduce, 
             args=(world_size, 300, 50), 
             nprocs=world_size, 
             join=True)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Single GPU baseline
    single_time = compare_single_vs_multi_gpu()
    
    # Multi-GPU benchmarks
    for world_size in [2, 4]:
        if world_size <= num_gpus:
            print(f"\n{'='*50}")
            print(f"Testing {world_size} GPUs")
            print('='*50)
            run_ddp_benchmark(world_size)
            
            # Theoretical vs actual scaling
            expected_time = single_time / world_size  # Perfect scaling
            print(f"\nScaling Analysis:")
            print(f"Single GPU time: {single_time:.3f}s ({1/single_time:.2f} it/s)")
            print(f"Expected {world_size}-GPU time: {expected_time:.3f}s ({1/expected_time:.2f} it/s)")


if __name__ == "__main__":
    main()