#!/usr/bin/env python3
"""
Quick DataLoader Speed Test
"""

import torch
import time
from buffer import DiscBuffer

def test_dataloader_speed():
    """Test how many batches per second the dataloader can provide"""
    
    buffer = DiscBuffer("/var/local/glang/activations/clt-activations-10M.h5", "tensor")
    
    loader = torch.utils.data.DataLoader(
        buffer,
        num_workers=20,
        prefetch_factor=2,
        batch_size=1000,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
    )
    
    print("Testing DataLoader speed...")
    print(f"Batch size: 1000")
    print(f"Workers: 20")
    
    # Warmup
    for i, batch in enumerate(loader):
        if i >= 5:
            break
    
    # Benchmark
    start_time = time.perf_counter()
    batch_count = 0
    
    for i, batch in enumerate(loader):
        batch_count += 1
        if i >= 100:  # Test 100 batches
            break
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    batches_per_second = batch_count / total_time
    samples_per_second = batches_per_second * 1000
    
    print(f"\nResults:")
    print(f"Batches tested: {batch_count}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Batches per second: {batches_per_second:.1f}")
    print(f"Samples per second: {samples_per_second:.0f}")

if __name__ == "__main__":
    test_dataloader_speed()