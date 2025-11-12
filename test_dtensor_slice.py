import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, DTensor, Shard

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

batch_size, n_layers, d_features = 4, 12, 128
mesh = DeviceMesh("cuda", list(range(world_size)))

features_local = torch.randn(batch_size, n_layers, d_features, device=f"cuda:{rank}")
features = DTensor.from_local(features_local, device_mesh=mesh, placements=[Shard(2)])

# The slice operation from the real code
l = 5
selected_features = features[:, : l + 1]

if rank == 0:
    print(f"✓ Slice succeeded! Shape: {selected_features.shape}")
