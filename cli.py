#!/usr/bin/env python3
"""
Lightning CLI for CrossLayer Transcoder training.
"""

import os

from lightning.pytorch.cli import LightningCLI

from data.datamodule import ActivationDataModule
from model.clt import CrossLayerTranscoder


class CrossLayerTranscoderCLI(LightningCLI):
    """
    Custom Lightning CLI for CrossLayer Transcoder training.
    """

    def add_arguments_to_parser(self, parser):
        """Add custom argument linking and configuration."""
        # Link model and data parameters that should be consistent
        parser.link_arguments("data.n_layers", "model.nonlinearity.init_args.n_layers")
        parser.link_arguments("data.n_layers", "model.n_layers")
        parser.link_arguments(
            "model.d_features", "model.nonlinearity.init_args.d_features"
        )


def main():
    """Main entry point for training."""
    # Set up wandb directories
    os.environ.setdefault("WANDB_DIR", f"{os.getcwd()}/wandb")
    os.environ.setdefault("WANDB_CACHE_DIR", f"{os.getcwd()}/wandb_cache")

    # Create CLI
    cli = CrossLayerTranscoderCLI(
        model_class=CrossLayerTranscoder,
        datamodule_class=ActivationDataModule,
        seed_everything_default=42,
        save_config_callback=None,  # Disable automatic config saving for now
        parser_kwargs={
            "prog": "CrossLayer Transcoder Training",
            "description": "Train CrossLayer Transcoder models for neural network interpretability",
        },
    )


if __name__ == "__main__":
    main()
