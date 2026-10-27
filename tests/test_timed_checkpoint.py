from pathlib import Path
from unittest.mock import Mock

import pytest

from crosslayer_transcoder.utils import callbacks
from crosslayer_transcoder.utils.callbacks import TimedCheckpointCallback


def test_timed_checkpoint_overwrites_atomically(tmp_path, monkeypatch):
    clock = iter([0.0, 5399.0, 5400.0, 5400.0, 5500.0])
    monkeypatch.setattr(callbacks.time, "monotonic", lambda: next(clock))
    trainer = Mock(is_global_zero=True)

    def save_checkpoint(path):
        Path(path).write_text("new checkpoint")

    trainer.save_checkpoint.side_effect = save_checkpoint
    callback = TimedCheckpointCallback(tmp_path, interval_minutes=90)
    (tmp_path / "clt.ckpt").write_text("old checkpoint")

    callback.on_train_start(trainer, None)
    callback.on_train_batch_end(trainer, None, None, None, 0)
    trainer.save_checkpoint.assert_not_called()
    callback.on_train_batch_end(trainer, None, None, None, 1)

    trainer.save_checkpoint.assert_called_once_with(tmp_path / "clt.ckpt.tmp")
    assert (tmp_path / "clt.ckpt").read_text() == "new checkpoint"
    assert not (tmp_path / "clt.ckpt.tmp").exists()


def test_timed_checkpoint_validates_interval():
    with pytest.raises(ValueError, match="interval_minutes must be positive"):
        TimedCheckpointCallback(interval_minutes=0)


def test_timed_checkpoint_only_saves_on_global_zero(tmp_path):
    trainer = Mock(is_global_zero=False)
    callback = TimedCheckpointCallback(tmp_path)

    callback.on_train_end(trainer, None)

    trainer.save_checkpoint.assert_not_called()
