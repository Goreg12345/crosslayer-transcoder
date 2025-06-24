"""
Activation Server Package

A multiprocessing FastAPI server for serving neural network activations
with shared PyTorch tensors and efficient data generation.
"""

import torch
import torch.multiprocessing as mp

# Set multiprocessing to use 'spawn' method for better compatibility with PyTorch
# This ensures shared tensors work properly across processes
mp.set_start_method("spawn", force=True)
