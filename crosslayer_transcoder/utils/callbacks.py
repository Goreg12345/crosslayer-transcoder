"""
Simple Lightning callbacks for CrossLayer Transcoder training.
"""

import logging
from functools import partial
from pathlib import Path
from typing import List, Optional

import lightning as L
import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

from crosslayer_transcoder.model import CrossLayerTranscoder, MultiLayerMolt
from crosslayer_transcoder.model.clt_lightning import CrossLayerTranscoderModule
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


class MoltPerLayerCheckpointCallback(L.Callback):
    """Save each `MultiLayerMolt` layer's transforms as a separate `.pt`.

    Files are written to `{checkpoint_dir}/{run_name}_layer_{layer}.pt`, where
    `run_name` is taken from the active wandb logger. If no wandb logger is
    attached, falls back to "molt".

    Optional HF offload: when `hf_repo_id` is set, periodic checkpoints are
    uploaded to that repo under `hf_repo_path/` and deleted locally to free
    disk. The final on_train_end checkpoint is always kept locally.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        every_n_train_steps: Optional[int] = None,
        hf_repo_id: Optional[str] = None,
        hf_repo_path: Optional[str] = None,
        delete_local_after_upload: bool = False,
        tokens_per_step: Optional[int] = None,
    ):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.every_n_train_steps = every_n_train_steps
        self.hf_repo_id = hf_repo_id
        self.hf_repo_path = (hf_repo_path or "").strip("/")
        self.delete_local_after_upload = delete_local_after_upload
        # If set, periodic-checkpoint filenames use processed-tokens instead
        # of optimizer-step. Pass effective batch (per_rank_batch * world_size
        # * accumulate_grad_batches) — the callback can't infer it reliably.
        self.tokens_per_step = tokens_per_step

    @staticmethod
    def _format_tokens(n: int) -> str:
        # Zero-padded millions for lexicographic sortability up to 9999M;
        # add K remainder only when non-zero so clean 10M-multiples stay tidy.
        m, rem = divmod(n, 1_000_000)
        if rem == 0:
            return f"{m:04d}M"
        return f"{m:04d}M{rem // 1_000:03d}K"

    @staticmethod
    def _wandb_run_name(trainer) -> Optional[str]:
        loggers = getattr(trainer, "loggers", None) or [trainer.logger]
        for lg in loggers:
            if lg is None:
                continue
            # Prefer init-arg lookups — they're the same on every rank, unlike
            # `.experiment` which is a _DummyExperiment on non-zero ranks
            # (whose __getattr__ returns its `nop` method, stringifying into
            # the filename if used directly).
            for attr in ("_name", "_wandb_init"):
                v = getattr(lg, attr, None)
                if isinstance(v, dict):
                    v = v.get("name")
                if isinstance(v, str) and v:
                    return v
            exp = getattr(lg, "experiment", None)
            name = getattr(exp, "name", None) if exp is not None else None
            if isinstance(name, str) and name:
                return name
        return None

    def _save(self, trainer, pl_module, suffix: str = "") -> List[Path]:
        model = pl_module.model
        if not isinstance(model, MultiLayerMolt):
            logger.warning(
                "MoltPerLayerCheckpointCallback expected MultiLayerMolt, got %s; skipping",
                type(model).__name__,
            )
            return []

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run_name = self._wandb_run_name(trainer) or "molt"
        saved: List[Path] = []
        for layer, molt in enumerate(model.molts):
            path = self.checkpoint_dir / f"{run_name}_layer_{layer}{suffix}.pt"
            torch.save(molt.state_dict(), path)
            logger.info("Saved MoLT layer %d to %s", layer, path)
            saved.append(path)
        return saved

    def _offload_to_hf(self, paths: List[Path]) -> None:
        """Upload paths to HF repo (creating it if needed) and optionally delete them."""
        if not self.hf_repo_id or not paths:
            return
        try:
            from huggingface_hub import HfApi
        except ImportError:
            logger.warning("huggingface_hub not installed; skipping HF offload")
            return

        api = HfApi()
        # Idempotent — exist_ok=True so this is safe to call every time.
        try:
            api.create_repo(repo_id=self.hf_repo_id, exist_ok=True)
        except Exception as e:
            logger.warning("HF create_repo failed (continuing): %s", e)

        for p in paths:
            remote = f"{self.hf_repo_path}/{p.name}" if self.hf_repo_path else p.name
            try:
                api.upload_file(
                    path_or_fileobj=str(p),
                    path_in_repo=remote,
                    repo_id=self.hf_repo_id,
                )
                logger.info("Uploaded %s to %s:%s", p, self.hf_repo_id, remote)
                if self.delete_local_after_upload:
                    p.unlink(missing_ok=True)
                    logger.info("Deleted local %s", p)
            except Exception as e:
                logger.error("HF upload of %s failed: %s", p, e)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_train_steps is None:
            return
        # DDP: only the global-zero rank saves & uploads. Other ranks have a
        # _DummyExperiment as their wandb logger, so attribute access (e.g.
        # `.name`) returns a bound `nop` method that would corrupt filenames.
        if not trainer.is_global_zero:
            return
        step = trainer.global_step
        if step > 0 and step % self.every_n_train_steps == 0:
            if self.tokens_per_step is not None:
                suffix = f"_tokens{self._format_tokens(step * self.tokens_per_step)}"
            else:
                suffix = f"_step{step}"
            saved = self._save(trainer, pl_module, suffix=suffix)
            self._offload_to_hf(saved)

    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        # Final checkpoint stays local — don't offload/delete.
        self._save(trainer, pl_module, suffix="")
