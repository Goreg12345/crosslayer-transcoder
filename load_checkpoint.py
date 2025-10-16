#!/usr/bin/env python3
"""
Script to programmatically load a CrossLayer Transcoder checkpoint and access the model.
"""

import os
import sys
from pathlib import Path

import lightning as L
import torch

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def load_checkpoint(path: str):


if __name__ == "__main__":
    # Run the example
    model = load_checkpoint("/var/local/glang/local/checkpoints/topk-control-epoch=0-step=93000.ckpt")

