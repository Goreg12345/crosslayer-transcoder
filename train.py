#!/usr/bin/env python3
"""
CrossLayer Transcoder Training Script
Uses Lightning CLI with configuration from default.yaml
"""

import os
import sys

# Add current directory to path for imports
sys.path.append(os.getcwd())

import lightning.pytorch as L
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)

from data.datamodule import ActivationDataModule
from model.clt_lightning import CrossLayerTranscoderModule

# Set up wandb directories
os.environ["WANDB_DIR"] = f"{os.getcwd()}/wandb"
os.environ["WANDB_CACHE_DIR"] = f"{os.getcwd()}/wandb_cache"


def main():

    # Create data module
    data_module = ActivationDataModule(
        # Buffer settings
        buffer_size=2_000_000,
        n_in_out=2,
        n_layers=12,
        activation_dim=768,
        dtype="float16",
        max_batch_size=50000,
        # Model settings for activation generation
        model_name="openai-community/gpt2",
        model_dtype="float32",
        # Dataset settings
        dataset_name="Skylion007/openwebtext",
        dataset_split="train",
        max_sequence_length=1024,
        # Generation settings
        generation_batch_size=2,
        refresh_interval=0.1,
        # Memory settings
        shared_memory_name="activation_buffer",
        timeout_seconds=30,
        # File paths
        init_file="/var/local/glang/activations/clt-activations-10M-shuffled_fp16.h5",
        # DataLoader settings
        batch_size=4000,
        num_workers=20,
        prefetch_factor=2,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
        use_shared_memory=True,
        device_map="cuda:2",
        # WandB logging configuration for data generation
        wandb_logging={
            "enabled": True,
            "project": "crosslayer-transcoder",
            "group": None,
            "run_name": "data-generator",
            "tags": ["data-generation"],
            "save_dir": "./wandb",
            "log_interval": 5.0,
        },
    )

    # Create replacement model accuracy metric
    # replacement_model_accuracy = ReplacementModelAccuracy(
    #     model_name="openai-community/gpt2", device_map="cuda:0", loader_batch_size=5
    # )

    # Create model
    model = CrossLayerTranscoderModule(
        # Model architecture parameters
        d_acts=768,
        d_features=768 * 8,  # 6144
        n_layers=12,
        # Nonlinearity parameters
        nonlinearity_theta=0.03,
        nonlinearity_bandwidth=1.0,
        # Standardizer parameters
        activation_dim=768,
        # Loss hyperparameters
        lambda_sparsity=0.0004,
        c_sparsity=0.1,
        # Optimization
        learning_rate=1e-3,
        # Metrics
        replacement_model_accuracy=None,  # replacement_model_accuracy,
        # Compilation
        compile=True,
    )

    model.configure_model()
    # Don't compile here - let Lightning handle compilation after model is fully initialized

    # Create logger
    logger = WandbLogger(project="crosslayer-transcoder", save_dir="./wandb")

    class TBProfilerCallback(L.Callback):
        def on_train_start(self, trainer, *_):
            self.prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=4, warmup=4, active=16),
                on_trace_ready=tensorboard_trace_handler("log/ddp"),
                record_shapes=True,
            )
            self.prof.__enter__()

        def on_train_batch_end(self, trainer, *_):
            self.prof.step()  # one step per batch

        def on_train_end(self, trainer, *_):
            self.prof.__exit__(None, None, None)

    print("Starting training")

    # Create trainer
    trainer = L.Trainer(
        logger=logger,
        max_steps=2500,
        val_check_interval=200,
        limit_val_batches=1,
        check_val_every_n_epoch=None,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        precision="16-mixed",
        accelerator="gpu",
        devices=1,
        accumulate_grad_batches=2,
        callbacks=[TBProfilerCallback()],
    )

    # Train the model
    trainer.fit(
        model=model,
        datamodule=data_module,
    )


if __name__ == "__main__":
    main()
