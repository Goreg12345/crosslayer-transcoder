"""
Simple Lightning callbacks for CrossLayer Transcoder training.
"""

import logging
from pathlib import Path

import lightning as L
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)

logger = logging.getLogger(__name__)


class TensorBoardProfilerCallback(L.Callback):
    """TensorBoard profiler callback."""

    def __init__(self, log_dir: str = "log/profiler"):
        super().__init__()
        self.log_dir = log_dir
        self.prof = None

    def on_train_start(self, trainer, pl_module):
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=4, warmup=4, active=16),
            on_trace_ready=tensorboard_trace_handler(self.log_dir),
            record_shapes=True,
        )
        self.prof.__enter__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.prof:
            self.prof.step()

    def on_train_end(self, trainer, pl_module):
        if self.prof:
            self.prof.__exit__(None, None, None)


class EndOfTrainingCheckpointCallback(L.Callback):
    """Save checkpoint only at end of training."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)

    def on_train_end(self, trainer, pl_module):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / "clt.ckpt"
        trainer.save_checkpoint(checkpoint_path)


class SaveWeightsCallback(L.Callback):
    """Save encoder and decoder weights as separate .pt files for investigation."""

    def __init__(self, output_dir: str = "weights"):
        super().__init__()
        self.output_dir = Path(output_dir)

    def on_train_end(self, trainer, pl_module):
        import torch

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save encoder weights
        encoder_weights = {
            "W": pl_module.model.encoder.W.detach().cpu(),
            "b": pl_module.model.encoder.b.detach().cpu(),
        }
        torch.save(encoder_weights, self.output_dir / "encoder.pt")
        logger.info(f"Saved encoder weights to {self.output_dir / 'encoder.pt'}")

        # Save decoder weights
        decoder = pl_module.model.decoder
        if hasattr(decoder, "W"):
            # Per-layer decoder
            decoder_weights = {"W": decoder.W.detach().cpu()}
        else:
            # Cross-layer decoder
            decoder_weights = {}
            for i in range(decoder.n_layers):
                decoder_weights[f"W_{i}"] = decoder.get_parameter(f"W_{i}").detach().cpu()
            decoder_weights["b"] = decoder.b.detach().cpu()

        torch.save(decoder_weights, self.output_dir / "decoder.pt")
        logger.info(f"Saved decoder weights to {self.output_dir / 'decoder.pt'}")
