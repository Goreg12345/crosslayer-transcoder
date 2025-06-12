import os
import time

import torch
import torch.distributed as dist

size = 1024 * 1024 * 1024  # 1 GiB tensor
iters = 20

dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()
torch.cuda.set_device(rank)

t = torch.empty(size // 4, dtype=torch.float32, device="cuda")

# warm-up
for _ in range(5):
    dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=False)

torch.cuda.synchronize()
t0 = time.perf_counter()

for _ in range(iters):
    dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=False)

torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

bytes_moved = 2 * (world - 1) / world * size * iters
print(f"Rank {rank}: {bytes_moved / elapsed / 1e9:.2f} GB/s effective")
