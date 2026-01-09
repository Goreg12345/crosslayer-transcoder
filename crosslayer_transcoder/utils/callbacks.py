"""
Simple Lightning callbacks for CrossLayer Transcoder training.
"""

from functools import partial
import logging
from pathlib import Path
from typing import List

import lightning as L
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)

from crosslayer_transcoder.model import CrossLayerTranscoder
from crosslayer_transcoder.model.serializable_module import SerializableModule

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


class SaveModelCallback(L.Callback):
    """Save checkpoint only at end of training."""
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        fold_standardizers: bool = True,
        on_events: List[str] = ["on_train_end"],
    ):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.fold_standardizers = fold_standardizers
        self.on_events = on_events
        self._setup_callbacks()

    def _setup_callbacks(self):
        for event in self.on_events:
            setattr(self, event, partial(self._save_model))

    def _save_model(self, trainer, pl_module, **kwargs):
        logger.info("Saving model...")
        pl_module.model.save_pretrained(
            self.checkpoint_dir, fold_standardizers=self.fold_standardizers
        )
        logger.info("Model saved")

