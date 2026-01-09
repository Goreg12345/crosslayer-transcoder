#!/usr/bin/env python3
"""
Lightning CLI for CrossLayer Transcoder training.
"""

import os

import lightning as L
from lightning.pytorch.cli import LightningCLI


class CrossLayerTranscoderCLI(LightningCLI):
    """
    Custom Lightning CLI for CrossLayer Transcoder training.
    """

    def add_arguments_to_parser(self, parser):
        """Add custom argument linking and configuration."""
        # Link model and data parameters that should be consistent
        parser.link_arguments("data.init_args.n_layers", "model.init_args.nonlinearity.init_args.n_layers")
        parser.link_arguments("data.init_args.n_layers", "model.init_args.n_layers")
        parser.link_arguments(
            "model.init_args.d_features",
            "model.init_args.nonlinearity.init_args.d_features",
        )


def main():
    """Main entry point for training."""
    # Set up wandb directories
    os.environ.setdefault("WANDB_DIR", f"{os.getcwd()}/wandb")
    os.environ.setdefault("WANDB_CACHE_DIR", f"{os.getcwd()}/wandb_cache")

    # Create CLI with subclass mode to support class_path configuration
    CrossLayerTranscoderCLI(
        model_class=L.LightningModule,  # Use base class for subclass mode
        datamodule_class=L.LightningDataModule,  # Use base class for subclass mode
        subclass_mode_model=True,  # Enable subclass mode for model
        subclass_mode_data=True,  # Enable subclass mode for datamodule
        seed_everything_default=42,
        save_config_callback=None,  # Disable automatic config saving for now
        parser_kwargs={
            "prog": "CrossLayer Transcoder Training",
            "description": "Train CrossLayer Transcoder models for neural network interpretability",
        },
    )


if __name__ == "__main__":
    main()
