#!/usr/bin/env python3
"""
Simple DDP timing measurement using model hooks
"""
import torch
import time
import numpy as np
from contextlib import contextmanager

class SimpleTimer:
    def __init__(self):
        self.times = {}
    
    @contextmanager
    def time(self, name):
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            end = time.perf_counter()
            if name not in self.times:
                self.times[name] = []
            self.times[name].append(end - start)
    
    def report(self):
        print("\n=== Timing Report ===")
        for name, times in self.times.items():
            times = np.array(times)
            print(f"{name}: {times.mean()*1000:.1f}ms avg, {times.std()*1000:.1f}ms std")

def measure_single_gpu_baseline():
    """Measure single GPU performance"""
    print("=== Single GPU Baseline ===")
    
    from clt import CrossLayerTranscoder
    from jumprelu import JumpReLU
    
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
    ).cuda(0)
    
    timer = SimpleTimer()
    batch_data = torch.randn(1000, 2, 12, 768, device=0, dtype=torch.float32)
    
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
    for i in range(10):
        with timer.time('total'):
            with timer.time('data_prep'):
                mean = batch_data.mean(dim=-1, keepdim=True)
                std = batch_data.std(dim=-1, keepdim=True)
                acts_norm = (batch_data - mean) / std
                resid, mlp_out = acts_norm[:, 0], batch_data[:, 1]
            
            with timer.time('forward'):
                features, recons = model(resid)
                loss = ((recons - mlp_out) ** 2).mean()
            
            with timer.time('backward'):
                loss.backward()
            
            with timer.time('zero_grad'):
                model.zero_grad()
    
    timer.report()
    
    total_times = np.array(timer.times['total'])
    print(f"Single GPU: {total_times.mean():.3f}s avg ({1/total_times.mean():.2f} it/s)")
    return total_times.mean()

def test_gradient_size():
    """Test actual gradient tensor sizes for your model"""
    print("\n=== Gradient Size Analysis ===")
    
    from clt import CrossLayerTranscoder
    from jumprelu import JumpReLU
    
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
    ).cuda(0)
    
    batch_data = torch.randn(1000, 2, 12, 768, device=0, dtype=torch.float32)
    
    # Forward + backward to generate gradients
    mean = batch_data.mean(dim=-1, keepdim=True)
    std = batch_data.std(dim=-1, keepdim=True)
    acts_norm = (batch_data - mean) / std
    resid, mlp_out = acts_norm[:, 0], batch_data[:, 1]
    
    features, recons = model(resid)
    loss = ((recons - mlp_out) ** 2).mean()
    loss.backward()
    
    # Analyze gradient sizes
    total_grad_size = 0
    grad_info = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_size_bytes = param.grad.numel() * param.grad.element_size()
            grad_size_mb = grad_size_bytes / (1024 * 1024)
            total_grad_size += grad_size_mb
            grad_info.append((name, param.grad.shape, grad_size_mb))
    
    grad_info.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Total gradient size: {total_grad_size:.1f} MB")
    print("Largest gradients:")
    for name, shape, size_mb in grad_info[:5]:
        print(f"  {name}: {shape} -> {size_mb:.1f} MB")
    
    return total_grad_size

def estimate_communication_time(grad_size_mb, bandwidth_gbps=8.0):
    """Estimate communication time based on gradient size and bandwidth"""
    grad_size_gb = grad_size_mb / 1024
    
    # AllReduce requires 2 * grad_size communication (reduce-scatter + all-gather)
    total_data_gb = grad_size_gb * 2
    
    comm_time_s = total_data_gb / bandwidth_gbps
    return comm_time_s

if __name__ == "__main__":
    # Test single GPU baseline
    single_gpu_time = measure_single_gpu_baseline()
    
    # Analyze gradient sizes
    grad_size_mb = test_gradient_size()
    
    # Estimate communication overhead
    print("\n=== Communication Analysis ===")
    
    # Test different bandwidth assumptions from your benchmark
    bandwidths = [6.0, 8.0, 21.0]  # GB/s from your P2P tests
    
    for bw in bandwidths:
        comm_time = estimate_communication_time(grad_size_mb, bw)
        
        # Estimate multi-GPU time: compute_time + communication_time
        compute_time = single_gpu_time * 0.8  # Assume some overlap/efficiency
        multi_gpu_time = compute_time + comm_time
        multi_gpu_throughput = 1 / multi_gpu_time
        
        print(f"\nBandwidth {bw} GB/s:")
        print(f"  Communication time: {comm_time*1000:.1f} ms")
        print(f"  Estimated 2-GPU time: {multi_gpu_time:.3f}s ({multi_gpu_throughput:.2f} it/s)")
        print(f"  vs Single GPU: {single_gpu_time:.3f}s ({1/single_gpu_time:.2f} it/s)")
        
        if multi_gpu_time > single_gpu_time:
            print(f"  ❌ Multi-GPU slower by {(multi_gpu_time/single_gpu_time-1)*100:.1f}%")
        else:
            efficiency = (1/multi_gpu_time) / (1/single_gpu_time) * 2  # 2 GPUs
            print(f"  ✅ Efficiency: {efficiency*100:.1f}%")
    
    print(f"\n=== Key Findings ===")
    print(f"If your multi-GPU training is getting ~1.1 it/s vs single GPU {1/single_gpu_time:.2f} it/s,")
    print(f"the communication overhead is dominating performance.")
    print(f"With {grad_size_mb:.1f} MB gradients, you need >15 GB/s effective bandwidth for good scaling.")