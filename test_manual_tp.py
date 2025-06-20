#!/usr/bin/env python3
"""
Test manual tensor parallelism
Run with: torchrun --nproc_per_node=4 test_manual_tp.py
"""

import os

import torch
import torch.distributed as dist


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def main():
    rank, local_rank, world_size = setup_distributed()

    from clt_manual_tp import CrossLayerTranscoderManualTP
    from jumprelu import JumpReLU

    # Create model
    clt = CrossLayerTranscoderManualTP(
        config={
            "d_acts": 768,
            "d_features": 768 * 8,  # 6144 total, 1536 per device
            "n_layers": 12,
            "lambda": 0.0002,
            "c": 0.1,
            "lr": 1e-3,
        },
        nonlinearity=JumpReLU(
            theta=0.03, bandwidth=1.0, n_layers=12, d_features=768 * 8 // world_size
        ),
    )

    clt = clt.cuda(local_rank)

    # Test forward pass
    batch_size = 1000
    dummy_batch = torch.randn(batch_size, 2, 12, 768, device=f"cuda:{local_rank}")

    mean = dummy_batch.mean(dim=-1, keepdim=True)
    std = dummy_batch.std(dim=-1, keepdim=True)
    acts_norm = (dummy_batch - mean) / std
    resid = acts_norm[:, 0]

    print(f"Rank {rank}: Input shape = {resid.shape}")

    features_shard, recons = clt.forward(resid)

    print(f"Rank {rank}: Features shard shape = {features_shard.shape}")
    print(f"Rank {rank}: Recons shape = {recons.shape}")

    # Verify the shapes are correct
    expected_features_shard = (batch_size, 12, 768 * 8 // world_size)
    expected_recons = (batch_size, 12, 768)

    assert (
        features_shard.shape == expected_features_shard
    ), f"Expected {expected_features_shard}, got {features_shard.shape}"
    assert (
        recons.shape == expected_recons
    ), f"Expected {expected_recons}, got {recons.shape}"

    print(f"Rank {rank}: ✅ Shapes are correct!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
