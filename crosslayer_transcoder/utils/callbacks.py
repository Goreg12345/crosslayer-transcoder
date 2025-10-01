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

from crosslayer_transcoder.utils.convert_to_circuit_tracer import (
    convert_model_to_circuit_tracer,
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


class CircuitTracerCallback(L.Callback):
    """Callback to convert the model to a circuit-tracer model."""

    def __init__(self, save_dir: str = "clt_module"):
        super().__init__()
        self.save_dir = Path(save_dir)

    def _convert_model(self, pl_module):
        convert_model_to_circuit_tracer(pl_module, self.save_dir.as_posix())
        logger.info(f"Circuit-tracer model saved to {self.save_dir.as_posix()}")

    def on_train_start(self, trainer, pl_module):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving circuit-tracer model to {self.save_dir.as_posix()}")

        self._convert_model(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._convert_model(pl_module)

    def on_train_end(self, trainer, pl_module):
        self._convert_model(pl_module)
