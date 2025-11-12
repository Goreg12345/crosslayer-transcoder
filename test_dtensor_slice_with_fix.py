import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, DTensor, Shard
from torch.distributed.tensor._ops.utils import register_op_strategy

# Register the missing sharding strategy for aten.alias
aten = torch.ops.aten

def propagate_single_input_strategy(op_schema):
    """
    For ops with a single tensor input, propagate the input sharding to output.
    This is correct for view/alias operations that don't change tensor structure.
    """
    from torch.distributed._tensor import DTensorSpec, OpStrategy, PlacementStrategy
    
    first_input_strategy = op_schema.args_schema[0]
    out_strategy = OpStrategy([])
    
    for strat in first_input_strategy.strategies:
        out_strategy.strategies.append(
            PlacementStrategy(
                output_specs=strat.output_spec,
                input_specs=(strat.output_spec,)
            )
        )
    return out_strategy

# Register aten.alias with the single input strategy
try:
    register_op_strategy([aten.alias.default])(propagate_single_input_strategy)
    print("✓ Registered sharding strategy for aten.alias.default")
except Exception as e:
    print(f"Failed to register: {e}")

# Now test the slice operation
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

batch_size, n_layers, d_features = 4, 12, 128
mesh = DeviceMesh("cuda", list(range(world_size)))

features_local = torch.randn(batch_size, n_layers, d_features, device=f"cuda:{rank}")
features = DTensor.from_local(features_local, device_mesh=mesh, placements=[Shard(2)])

# The slice operation that was failing
l = 5
try:
    selected_features = features[:, :l + 1]
    if rank == 0:
        print(f"✓ Slice succeeded! Shape: {selected_features.shape}")
except Exception as e:
    if rank == 0:
        print(f"✗ Slice failed: {e}")



