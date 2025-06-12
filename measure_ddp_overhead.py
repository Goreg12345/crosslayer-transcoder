#!/usr/bin/env python3
"""
Measure DDP gradient synchronization overhead in actual training

This script instruments your training loop to measure:
1. Forward pass time
2. Backward pass time (without DDP sync)
3. DDP gradient synchronization time
4. Total time per iteration

Identifies exactly where the DDP bottleneck is.
"""

import torch
import torch.distributed as dist
import time
import contextlib
from collections import defaultdict
import numpy as np

class DDPTimer:
    """Context manager to time DDP operations"""
    
    def __init__(self):
        self.times = defaultdict(list)
        self.current_timers = {}
    
    @contextlib.contextmanager
    def timer(self, name):
        """Time a specific operation"""
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            end = time.perf_counter()
            self.times[name].append(end - start)
    
    def start_timer(self, name):
        """Start a named timer"""
        torch.cuda.synchronize()
        self.current_timers[name] = time.perf_counter()
    
    def end_timer(self, name):
        """End a named timer"""
        torch.cuda.synchronize()
        if name in self.current_timers:
            end = time.perf_counter()
            duration = end - self.current_timers[name]
            self.times[name].append(duration)
            del self.current_timers[name]
            return duration
        return 0
    
    def report(self, rank=0):
        """Print timing summary"""
        if rank == 0:
            print("\n=== DDP Timing Analysis ===")
            for name, times in self.times.items():
                times = np.array(times)
                print(f"{name}:")
                print(f"  Average: {times.mean()*1000:.1f} ms")
                print(f"  Min: {times.min()*1000:.1f} ms")
                print(f"  Max: {times.max()*1000:.1f} ms")
                print(f"  Std: {times.std()*1000:.1f} ms")


def hook_ddp_communication(model, timer, rank):
    """Hook into DDP communication to measure sync time"""
    
    original_reducer_prepare_for_backward = model._reducer._prepare_for_backward
    original_reducer_rebuild_buckets = model._reducer._rebuild_buckets
    
    def timed_prepare_for_backward(outputs):
        timer.start_timer('ddp_prepare')
        result = original_reducer_prepare_for_backward(outputs)
        timer.end_timer('ddp_prepare')
        return result
    
    def timed_rebuild_buckets(self):
        timer.start_timer('ddp_bucket_rebuild')
        result = original_reducer_rebuild_buckets()
        timer.end_timer('ddp_bucket_rebuild')
        return result
    
    # Hook the communication
    model._reducer._prepare_for_backward = timed_prepare_for_backward
    model._reducer._rebuild_buckets = timed_rebuild_buckets
    
    # Hook allreduce operations
    if hasattr(model._reducer, '_bucket_ready_callback'):
        original_callback = model._reducer._bucket_ready_callback
        
        def timed_callback(bucket):
            timer.start_timer('allreduce_operation')
            result = original_callback(bucket)
            timer.end_timer('allreduce_operation')
            return result
        
        model._reducer._bucket_ready_callback = timed_callback


def benchmark_training_step_breakdown(batch_size=1000, num_iterations=20, world_size=2):
    """Benchmark individual components of training step"""
    
    # Setup basic imports
    import os
    os.environ['MASTER_ADDR'] = 'localhost' 
    os.environ['MASTER_PORT'] = '12356'
    
    rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    from clt import CrossLayerTranscoder
    from jumprelu import JumpReLU
    
    # Create model
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
    ).cuda(rank)
    
    # Wrap with DDP
    model_ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    # Setup timer and hooks
    timer = DDPTimer()
    hook_ddp_communication(model_ddp, timer, rank)
    
    # Create data
    batch_data = torch.randn(batch_size, 2, 12, 768, device=rank, dtype=torch.float32)
    
    # Warmup
    if rank == 0:
        print(f"Starting warmup with batch_size={batch_size}, world_size={world_size}")
    
    for _ in range(3):
        with timer.timer('warmup_total'):
            mean = batch_data.mean(dim=-1, keepdim=True)
            std = batch_data.std(dim=-1, keepdim=True)
            acts_norm = (batch_data - mean) / std
            resid, mlp_out = acts_norm[:, 0], batch_data[:, 1]
            
            features, recons = model_ddp(resid)
            loss = ((recons - mlp_out) ** 2).mean()
            loss.backward()
            model_ddp.zero_grad()
    
    # Clear warmup times
    timer.times.clear()
    
    # Benchmark
    if rank == 0:
        print(f"Starting benchmark: {num_iterations} iterations")
    
    for i in range(num_iterations):
        timer.start_timer('total_iteration')
        
        # Data preprocessing
        with timer.timer('data_preprocessing'):
            mean = batch_data.mean(dim=-1, keepdim=True)
            std = batch_data.std(dim=-1, keepdim=True)
            acts_norm = (batch_data - mean) / std
            resid, mlp_out = acts_norm[:, 0], batch_data[:, 1]
        
        # Forward pass
        with timer.timer('forward_pass'):
            features, recons = model_ddp(resid)
            loss = ((recons - mlp_out) ** 2).mean()
        
        # Backward pass (includes DDP sync)
        with timer.timer('backward_pass_total'):
            loss.backward()
        
        # Zero gradients
        with timer.timer('zero_grad'):
            model_ddp.zero_grad()
        
        iteration_time = timer.end_timer('total_iteration')
        
        if rank == 0 and i % 5 == 0:
            print(f"Iteration {i+1}: {iteration_time:.3f}s ({1/iteration_time:.2f} it/s)")
    
    # Report results
    timer.report(rank)
    
    # Calculate communication overhead
    if rank == 0:
        total_times = np.array(timer.times['total_iteration'])
        forward_times = np.array(timer.times['forward_pass'])
        backward_times = np.array(timer.times['backward_pass_total'])
        
        print(f"\n=== Performance Analysis ===")
        print(f"Average iteration time: {total_times.mean():.3f}s ({1/total_times.mean():.2f} it/s)")
        print(f"Forward pass: {forward_times.mean()*1000:.1f} ms ({forward_times.mean()/total_times.mean()*100:.1f}%)")
        print(f"Backward pass (total): {backward_times.mean()*1000:.1f} ms ({backward_times.mean()/total_times.mean()*100:.1f}%)")
        
        if 'allreduce_operation' in timer.times:
            allreduce_times = np.array(timer.times['allreduce_operation'])
            print(f"DDP AllReduce: {allreduce_times.mean()*1000:.1f} ms ({allreduce_times.mean()/total_times.mean()*100:.1f}%)")
        
        # Estimate what single GPU performance would be
        compute_time = forward_times.mean() + (backward_times.mean() - np.array(timer.times.get('allreduce_operation', [0])).mean())
        estimated_single_gpu = 1 / compute_time
        efficiency = (1/total_times.mean()) / estimated_single_gpu * world_size
        
        print(f"\nEstimated single GPU speed: {estimated_single_gpu:.2f} it/s")
        print(f"DDP efficiency: {efficiency*100:.1f}% (100% = perfect scaling)")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        world_size = int(sys.argv[1])
    else:
        world_size = 2
    
    # Run the benchmark
    benchmark_training_step_breakdown(batch_size=1000, num_iterations=20, world_size=world_size)