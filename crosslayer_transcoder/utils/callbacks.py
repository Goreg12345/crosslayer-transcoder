"""
Simple Lightning callbacks for CrossLayer Transcoder training.
"""

from ast import List
import logging
from pathlib import Path

from huggingface_hub import upload_folder
import lightning as L
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)

from crosslayer_transcoder.utils.model_converters.circuit_tracer import (
    CircuitTracerConverter,
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

class ModelConversionCallback(L.Callback):
    """Callback to convert the model to a circuit-tracer model."""

    # Note: you can't type these with List
    def __init__(self, save_dir: str = "clt_module", kinds=["circuit_tracer"], on_event=["on_train_batch_end"]):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.kinds = kinds
        self.on_event = on_event
        self.converters = {
            "circuit_tracer": CircuitTracerConverter,
        }
        self._setup_hooks()

    def _setup_hooks(self):
        for event in self.on_event:
            setattr(self, event, lambda *args, **kwargs: self._convert_model(*args, **kwargs))

    # Note: this should have the signature as all Lightning callbacks
    def _convert_model(self, trainer, pl_module, **kwargs):
        for kind in self.kinds:
            converter = self.converters[kind](save_dir=self.save_dir.as_posix())
            converter.convert_and_save(pl_module)
            logger.info(
                f"{kind} model converted and saved to {self.save_dir.as_posix()}"
            )


class HuggingFaceCallback(L.Callback):
    """Callback to upload the model to Hugging Face."""

    def __init__(
        self, repo_id: str, repo_type: str = "model", save_dir: str = "clt_module"
    ):
        super().__init__()
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.save_dir = Path(save_dir)

    def on_train_end(self, trainer, pl_module):
        upload_folder(
            folder_path=self.save_dir.as_posix(),
            repo_id=self.repo_id,
            repo_type=self.repo_type,
        )
