"""
Simple Lightning callbacks for CrossLayer Transcoder training.
"""

import logging
import re
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
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        every_n_train_steps: Optional[int] = None,
    ):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.every_n_train_steps = every_n_train_steps

    @staticmethod
    def _wandb_run_name(trainer) -> Optional[str]:
        loggers = getattr(trainer, "loggers", None) or [trainer.logger]
        for lg in loggers:
            if lg is None:
                continue
            exp = getattr(lg, "experiment", None)
            name = getattr(exp, "name", None) if exp is not None else None
            if name:
                return name
        return None

    def _save(self, trainer, pl_module, suffix: str = ""):
        model = pl_module.model
        if not isinstance(model, MultiLayerMolt):
            logger.warning(
                "MoltPerLayerCheckpointCallback expected MultiLayerMolt, got %s; skipping",
                type(model).__name__,
            )
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        run_name = self._wandb_run_name(trainer) or "molt"
        for layer, molt in enumerate(model.molts):
            path = self.checkpoint_dir / f"{run_name}_layer_{layer}{suffix}.pt"
            torch.save(molt.state_dict(), path)
            logger.info("Saved MoLT layer %d to %s", layer, path)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_train_steps is None:
            return
        step = trainer.global_step
        if step > 0 and step % self.every_n_train_steps == 0:
            self._save(trainer, pl_module, suffix=f"_step{step}")

    def on_train_end(self, trainer, pl_module):
        self._save(trainer, pl_module, suffix="")


def _slug(text: str, max_len: int = 40) -> str:
    """A wandb-metric-key-safe slug for a prompt."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", text.strip()).strip("_").lower()
    return s[:max_len] or "prompt"


class MoltEvalPromptCallback(L.Callback):
    """Log MoLT-spliced (and optional vanilla) completions on fixed prompts.

    At each checkpoint step (and at train end) this runs the *current* MoLT
    weights spliced into a base LM on a handful of probe prompts — e.g.
    "3 days after Tuesday is" or "3+47=" — and logs the greedy completions,
    top-1 token, and KL-vs-vanilla to the active wandb logger. This makes it
    possible to watch, over training, whether the MoLT preserves the base
    model's manifold/arithmetic behaviour.

    The base LM lives in the data-generator process during training, so this
    callback loads its *own* copy in the trainer process. Control the memory
    cost with `eval_device` / `eval_dtype` (e.g. put it on a second GPU or CPU).
    The model is lazily loaded on the first eval and freed at train end.

    Prompts are wrapped with the tokenizer's chat template by default
    (`chat_template=True`), matching IT-model training.
    """

    def __init__(
        self,
        prompts: List[str],
        base_model: str,
        *,
        chat_template: bool = True,
        system_prompt: Optional[str] = None,
        mode: str = "both",  # "vanilla" | "molt" | "both"
        splice_layers: Optional[List[int]] = None,
        max_new_tokens: int = 8,
        topk: int = 10,
        every_n_train_steps: Optional[int] = None,
        eval_on_train_end: bool = True,
        eval_device: Optional[str] = None,  # default: pl_module.device
        eval_dtype: str = "bfloat16",
    ):
        super().__init__()
        self.prompts = list(prompts)
        self.base_model = base_model
        self.chat_template = chat_template
        self.system_prompt = system_prompt
        self.mode = mode
        self.splice_layers = splice_layers
        self.max_new_tokens = max_new_tokens
        self.topk = topk
        self.every_n_train_steps = every_n_train_steps
        self.eval_on_train_end = eval_on_train_end
        self.eval_device = eval_device
        self.eval_dtype = eval_dtype

        self._model = None
        self._tokenizer = None
        self._arch = None

    # -- base LM lifecycle ---------------------------------------------------

    def _ensure_base_model(self, pl_module) -> bool:
        """Lazily load the base LM + tokenizer. Returns False on failure."""
        if self._model is not None:
            return True
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from crosslayer_transcoder.utils.molt_splice import _DTYPE, resolve_blocks

        device = self.eval_device or str(pl_module.device)
        dtype = _DTYPE[self.eval_dtype]
        logger.info(
            "MoltEvalPromptCallback: loading base LM %s (%s) on %s …",
            self.base_model, self.eval_dtype, device,
        )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model, torch_dtype=dtype
            ).to(device).eval()
            self._arch, _ = resolve_blocks(model)
            self._model = model
            return True
        except Exception as e:  # noqa: BLE001 - eval must never crash training
            logger.warning("MoltEvalPromptCallback: failed to load base LM: %s", e)
            return False

    def _free_base_model(self):
        self._model = None
        self._tokenizer = None
        self._arch = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- wandb logging -------------------------------------------------------

    @staticmethod
    def _wandb_experiment(trainer):
        loggers = getattr(trainer, "loggers", None) or [trainer.logger]
        for lg in loggers:
            exp = getattr(lg, "experiment", None)
            # WandB run objects expose `.log`.
            if exp is not None and hasattr(exp, "log") and hasattr(exp, "name"):
                return exp
        return None

    # -- eval driver ---------------------------------------------------------

    @torch.no_grad()
    def _run_eval(self, trainer, pl_module, step: int):
        model = pl_module.model
        if not isinstance(model, MultiLayerMolt):
            logger.warning(
                "MoltEvalPromptCallback expected MultiLayerMolt, got %s; skipping",
                type(model).__name__,
            )
            return
        if not self._ensure_base_model(pl_module):
            return

        from crosslayer_transcoder.utils.molt_splice import (
            apply_chat_template,
            compare_prompt,
        )

        was_training = model.training
        model.eval()
        try:
            rows = []
            metrics = {}
            for raw in self.prompts:
                prompt_text = (
                    apply_chat_template(self._tokenizer, raw, self.system_prompt)
                    if self.chat_template
                    else raw
                )
                r = compare_prompt(
                    model=self._model,
                    tokenizer=self._tokenizer,
                    prompt_text=prompt_text,
                    arch=self._arch,
                    molt=model,
                    mode=self.mode,
                    splice_layers=self.splice_layers,
                    topk=self.topk,
                    max_new_tokens=self.max_new_tokens,
                )
                slug = _slug(raw)
                vanilla = r.get("vanilla", {})
                molt = r.get("molt", {})
                rows.append([
                    step,
                    raw,
                    vanilla.get("completion", ""),
                    molt.get("completion", ""),
                    molt.get("top", [("", 0.0)])[0][0] if molt.get("top") else "",
                    r.get("kl_vanilla_molt", float("nan")),
                    r.get("agree_top1", None),
                ])
                if "kl_vanilla_molt" in r:
                    metrics[f"eval/{slug}/kl_vanilla_molt"] = r["kl_vanilla_molt"]
                    metrics[f"eval/{slug}/agree_top1"] = float(r["agree_top1"])
        finally:
            if was_training:
                model.train()

        exp = self._wandb_experiment(trainer)
        if exp is None:
            logger.info("MoltEvalPromptCallback: no wandb logger; skipping log")
            return
        try:
            import wandb

            table = wandb.Table(
                columns=[
                    "step", "prompt", "vanilla_completion", "molt_completion",
                    "molt_top1", "kl_vanilla_molt", "agree_top1",
                ],
                data=rows,
            )
            exp.log({"eval/prompts": table, **metrics}, step=step)
            logger.info("MoltEvalPromptCallback: logged %d prompts @ step %d", len(rows), step)
        except Exception as e:  # noqa: BLE001
            logger.warning("MoltEvalPromptCallback: wandb log failed: %s", e)

    # -- hooks ---------------------------------------------------------------

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_train_steps is None:
            return
        step = trainer.global_step
        if step > 0 and step % self.every_n_train_steps == 0:
            self._run_eval(trainer, pl_module, step)

    def on_train_end(self, trainer, pl_module):
        if self.eval_on_train_end:
            self._run_eval(trainer, pl_module, trainer.global_step)
        self._free_base_model()
