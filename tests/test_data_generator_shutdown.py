"""Exercise shutdown in isolated processes, without models, GPUs or network access."""

import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from crosslayer_transcoder.data.data_generator import DataGeneratorProcess
from crosslayer_transcoder.data.datamodule import ActivationDataModule
from crosslayer_transcoder.data.generation_loop import DataGenerationLoop
from crosslayer_transcoder.utils.callbacks import TimedCheckpointCallback


class _TinyGenerator(DataGeneratorProcess):
    def __init__(self):
        super().__init__(
            shared_buffer=None, buffer_size=2, n_in_out=2, n_layers=1,
            activation_dim=1, dtype=torch.float32, max_batch_size=1,
            model_name="unused", model_dtype=torch.float32, dataset_name="unused",
            dataset_split="train", dataset_text_field="text", max_sequence_length=1,
            generation_batch_size=1, refresh_interval=0.01,
        )
        self.ready = mp.Event()

    def setup(self):
        # Use real spawned DataLoader workers and the production cleanup method.
        self.generation_loop = DataGenerationLoop.__new__(DataGenerationLoop)
        self.generation_loop.running = True
        self.generation_loop.text_dataset_loader = iter(DataLoader(
            TensorDataset(torch.ones(8, 1)), num_workers=1,
        ))
        next(self.generation_loop.text_dataset_loader)
        self.ready.set()
        self.stop_event.wait()


class _TinyDataModule(ActivationDataModule):
    def setup(self, stage=None):
        self.data_generator = _TinyGenerator()
        self.data_generator.start()
        if not self.data_generator.ready.wait(timeout=20):
            self.teardown(stage)
            raise RuntimeError("Test generator did not start")
        self.data_loader = DataLoader(TensorDataset(torch.ones(8, 1)), num_workers=1)


class _TinyModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def training_step(self, batch, batch_idx):
        return (batch[0] * self.weight).sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _smoke_consecutive_fits(directory):
    """Check interpreter exit as well as fit() returning, with offline W&B."""
    import psutil
    import wandb
    from lightning.pytorch.loggers import WandbLogger

    mp.set_start_method("spawn", force=True)
    for run in range(2):
        datamodule = _TinyDataModule(buffer_size=2, batch_size=1)
        checkpoint_dir = directory / str(run)
        trainer = L.Trainer(
            accelerator="cpu", devices=1, max_steps=2, limit_val_batches=0,
            enable_progress_bar=False, enable_model_summary=False,
            enable_checkpointing=False, num_sanity_val_steps=0,
            logger=WandbLogger(project="shutdown-test", save_dir=str(directory), offline=True),
            callbacks=[TimedCheckpointCallback(str(checkpoint_dir))],
        )
        trainer.fit(_TinyModel(), datamodule=datamodule)
        assert datamodule.data_generator.exitcode == 0
        assert torch.load(checkpoint_dir / "clt.ckpt", weights_only=False)["global_step"] == 2
        # This requires the trainer's W&B service to have survived teardown.
        wandb.finish()
        datamodule.teardown("fit")  # Repeated cleanup must be harmless.
        assert not mp.active_children()
        assert not any("spawn_main" in " ".join(p.cmdline())
                       for p in psutil.Process().children(recursive=True))
        print(f"Run {run} exited cleanly", flush=True)


def test_consecutive_fits_exit_cleanly(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "tests.test_data_generator_shutdown", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=90,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "WANDB_MODE": "offline",
             "WANDB_DIR": str(tmp_path), "WANDB_CACHE_DIR": str(tmp_path / "cache")},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Run 0 exited cleanly" in result.stdout
    assert "Run 1 exited cleanly" in result.stdout
    assert "Traceback" not in result.stdout + result.stderr


def test_force_terminate_does_not_clean_up_in_parent(monkeypatch):
    generator = _TinyGenerator()
    cleanup = Mock()
    terminate = Mock()
    monkeypatch.setattr(generator, "cleanup", cleanup)
    monkeypatch.setattr(mp.Process, "terminate", terminate)

    generator.terminate()

    cleanup.assert_not_called()
    terminate.assert_called_once()


def test_generation_loop_observes_stop_event():
    loop = DataGenerationLoop.__new__(DataGenerationLoop)
    loop.running = True
    loop.stop_event = mp.Event()
    loop.stop_event.set()
    loop.deployment_policy = Mock()
    loop.model_setup_params = {}
    loop._setup_text_dataset_loader = Mock()
    loop.shared_buffer = SimpleNamespace(_get_invalid_indices=Mock())

    loop.generation_loop("unused", torch.float32, Mock())

    loop.shared_buffer._get_invalid_indices.assert_not_called()


@pytest.mark.parametrize("alive", [[True, False, False], [True, True, False], [True, True, True]])
def test_teardown_escalates_only_for_unresponsive_generator(alive):
    datamodule = ActivationDataModule(buffer_size=1, batch_size=1)
    generator = Mock(is_alive=Mock(side_effect=alive))
    datamodule.data_generator = generator

    datamodule.teardown("fit")

    generator.request_stop.assert_called_once()
    assert generator.terminate.call_count == int(alive[1])
    assert generator.kill.call_count == int(alive[2])


if __name__ == "__main__":
    _smoke_consecutive_fits(Path(sys.argv[1]))
