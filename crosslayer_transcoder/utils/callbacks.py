"""
Simple Lightning callbacks for CrossLayer Transcoder training.
"""

import logging
from functools import partial
from pathlib import Path
from typing import List

import lightning as L
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

from crosslayer_transcoder.model.clt_lightning import CrossLayerTranscoderModule

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
    """Save checkpoint only at end of training with WandB run name."""

    def __init__(self, checkpoint_dir: str = "local/checkpoints"):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_name = None

    def on_train_start(self, trainer, pl_module):
        """Get run name from WandB logger."""
        # Get run name from WandB logger if available
        if trainer.loggers:
            for trainer_logger in trainer.loggers:
                if hasattr(trainer_logger, "name") and trainer_logger.name:
                    self.run_name = trainer_logger.name
                    break

        # Fallback to a default name if no WandB logger
        if not self.run_name:
            self.run_name = "clt-training"

    def on_train_end(self, trainer, pl_module):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create a more descriptive filename with step and epoch info
        step = trainer.global_step
        epoch = trainer.current_epoch

        # Get project name from WandB logger if available
        project_name = "clt"
        if trainer.loggers:
            for trainer_logger in trainer.loggers:
                if hasattr(trainer_logger, "project") and trainer_logger.project:
                    project_name = trainer_logger.project
                    break

        checkpoint_name = f"{project_name}-{self.run_name}-epoch{epoch}-step{step}-final.ckpt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        trainer.save_checkpoint(checkpoint_path)
        logger.info(f"Saved final checkpoint: {checkpoint_path}")



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

    def _save_model(self, trainer, pl_module: CrossLayerTranscoderModule, **kwargs):
        logger.info("Saving model...")
        pl_module.model.save_pretrained(self.checkpoint_dir, fold_standardizers=self.fold_standardizers)
        logger.info("Model saved")


class FoldAndSaveModelCallback(L.Callback):
    """Fold and save model at end of training."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)

    def on_train_end(self, trainer, pl_module):
        pl_module.model.fold()
        pl_module.model.save_pretrained(self.checkpoint_dir)
