#!/usr/bin/env python3
"""
Native PyTorch training with tensor parallelism
Run with: torchrun --nproc_per_node=4 train_native_tp.py
"""

import os

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

import wandb

# Set environment
os.environ["WANDB_DIR"] = f"{os.getcwd()}/wandb"
os.environ["WANDB_CACHE_DIR"] = f"{os.getcwd()}/wandb_cache"

from buffer import DiscBuffer
from clt import CrossLayerTranscoder
from jumprelu import JumpReLU


def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def setup_tensor_parallel_model(model, world_size):
    """Setup tensor parallelism for the model"""
    print(f"Setting up tensor parallelism across {world_size} devices")

    # Create device mesh
    tp_mesh = init_device_mesh(
        "cuda", (world_size,), mesh_dim_names=("tensor_parallel",)
    )

    # Define parallelization plan
    plan = {
        "W_enc": ColwiseParallel(),
        "W_dec": RowwiseParallel(),
    }

    print(f"Before TP - W_enc: {model.W_enc.shape}, W_dec: {model.W_dec.shape}")

    # Apply tensor parallelism
    parallelize_module(model, tp_mesh, plan)

    print(f"After TP - W_enc: {model.W_enc.shape}, W_dec: {model.W_dec.shape}")

    return model


def main():
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()

    print(f"Process {rank}/{world_size} on device {local_rank}")

    # Create model
    clt = CrossLayerTranscoder(
        config={
            "d_acts": 768,
            "d_features": 768 * 8,
            "n_layers": 12,
            "lambda": 0.0002,
            "c": 0.1,
            "lr": 1e-3,
        },
        nonlinearity=JumpReLU(
            theta=0.03, bandwidth=1.0, n_layers=12, d_features=768 * 8
        ),
    )

    # Move to device before tensor parallelism
    clt = clt.cuda(local_rank)

    # Setup tensor parallelism
    clt = setup_tensor_parallel_model(clt, world_size)

    # Setup optimizer and scaler for mixed precision
    optimizer = torch.optim.Adam(clt.parameters(), lr=1e-3)
    scaler = GradScaler()

    # Setup data
    buffer = DiscBuffer("/var/local/glang/activations/clt-activations-10M.h5", "tensor")
    loader = torch.utils.data.DataLoader(
        buffer,
        num_workers=20,
        prefetch_factor=2,
        batch_size=1000,
        shuffle=False,
        persistent_workers=True,
        pin_memory=True,
    )

    # Setup wandb (only on rank 0)
    if rank == 0:
        wandb.init(project="wandb_clt_native_tp")

    # Training loop
    clt.train()
    for step, batch in enumerate(loader):
        if step >= 2000:
            break

        batch = batch.cuda(local_rank, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            # Normalize activations
            mean = batch.mean(dim=-1, keepdim=True)
            std = batch.std(dim=-1, keepdim=True)
            acts_norm = (batch - mean) / std
            resid, mlp_out = acts_norm[:, 0], batch[:, 1]

            # Forward pass
            features, recons = clt.forward(resid)

            # Compute loss (same as in Lightning version)
            mse = ((recons - mlp_out) ** 2).mean()

            # Sparsity loss
            from einops import einsum

            masked_w = einsum(
                clt.W_dec,
                clt.mask,
                "from_layer to_layer d_features d_acts, from_layer to_layer -> from_layer to_layer d_features d_acts",
            )
            l1 = masked_w.norm(p=1, dim=[1, 3])
            tanh = torch.tanh(features * l1 * clt.c)
            sparsity = clt._lambda * tanh.sum(dim=[1, 2]).mean()

            loss = mse + sparsity

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Logging (only on rank 0)
        if rank == 0 and step % 10 == 0:
            l0_pct = 100 * (features > 0).float().mean()
            print(
                f"Step {step}: Loss={loss.item():.4f}, MSE={mse.item():.4f}, Sparsity={sparsity.item():.4f}, L0={l0_pct.item():.1f}%"
            )

            if wandb.run:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "train_mse": mse.item(),
                        "train_sparsity": sparsity.item(),
                        "L0_pct": l0_pct.item(),
                    },
                    step=step,
                )

    # Cleanup
    if rank == 0:
        print("Training complete!")
        if wandb.run:
            wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
