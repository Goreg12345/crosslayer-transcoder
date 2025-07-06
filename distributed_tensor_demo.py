import os

import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"


def run_distributed_training(rank, world_size):
    """Function to run on each process"""
    # Set environment variables for this process
    print(f"Process {rank} started")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Initialize distributed training
    torch.cuda.set_device(rank)
    init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    # Create a tensor on each process
    tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    tensor = tensor.to("cuda")

    print(f"Process {rank} tensor: {tensor}")

    print(f"Process {rank} completed")

    DeviceMesh()

    destroy_process_group()


print(__name__)
if __name__ == "__main__":
    world_size = 2  # Number of processes/GPUs
    mp.spawn(run_distributed_training, args=(world_size,), nprocs=world_size, join=True)
