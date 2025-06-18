import time

import torch

src = 3
dst = 2
size_mb = 1000  # 1 GB
num_iters = 100

tensor = torch.randn(
    size_mb * 1024 * 1024 // 4, device=f"cuda:{src}", dtype=torch.float32
)
torch.cuda.synchronize()

# Warmup
for _ in range(5):
    tensor_dst = tensor.to(f"cuda:{dst}")
    torch.cuda.synchronize()

# Benchmark
start = time.perf_counter()
for _ in range(num_iters):
    tensor_dst = tensor.to(f"cuda:{dst}")
    torch.cuda.synchronize()
end = time.perf_counter()

elapsed = end - start
total_mb = size_mb * num_iters
bw = total_mb / elapsed / 1024  # GB/s

print(f"P2P Bandwidth from GPU {src} to {dst}: {bw:.2f} GB/s")
