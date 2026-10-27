import json
from unittest.mock import Mock

import pytest
import torch

from tools import launch_queued_molt as queue


@pytest.mark.parametrize(
    "finished_step,should_launch,oom",
    [(99999, False, False), (100000, True, False), (100000, True, True)],
)
def test_launch_requires_completed_predecessor(tmp_path, monkeypatch, finished_step, should_launch, oom):
    state_path = tmp_path / "queue.json"
    state_path.write_text(json.dumps({
        "phase": "waiting",
        "repo": str(tmp_path),
        "predecessor_processes": {"123": "old-start-time"},
        "predecessor_checkpoint": str(tmp_path / "old.ckpt"),
        "required_steps": 100000,
        "new_checkpoint": str(tmp_path / "new.ckpt"),
        "config": "config/queued.yaml",
        "training_log": str(tmp_path / "training.log"),
    }))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(queue.sys, "argv", ["launch_queued_molt.py", str(state_path)])
    # A reused PID must not keep the queue waiting for an unrelated process.
    monkeypatch.setattr(queue, "process_identity", lambda pid: "different-start-time")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"global_step": finished_step})
    check = Mock()
    launch = Mock(return_value=Mock(pid=456, poll=Mock(return_value=0), wait=Mock(return_value=0)))
    monkeypatch.setattr(queue.subprocess, "run", check)
    monkeypatch.setattr(queue.subprocess, "Popen", launch)
    kill = Mock()
    monkeypatch.setattr(queue.os, "killpg", kill)
    if oom:
        def start_failed_process(*args, **kwargs):
            (tmp_path / "training.log").write_text("torch.OutOfMemoryError: CUDA out of memory\n")
            return Mock(pid=456, poll=Mock(return_value=None), wait=Mock(return_value=-15))

        launch.side_effect = start_failed_process

    if oom:
        with pytest.raises(RuntimeError, match="out of GPU memory"):
            queue.main()
        assert json.loads(state_path.read_text())["phase"] == "failed"
        kill.assert_called_once_with(456, queue.signal.SIGTERM)
    elif should_launch:
        queue.main()
        assert json.loads(state_path.read_text())["phase"] == "complete"
        assert check.call_count == 2  # Tokenization regression and config validation.
        launch.assert_called_once()
        assert launch.call_args.args[0][-1] == "config/queued.yaml"
        with pytest.raises(RuntimeError, match="duplicate launch"):
            queue.main()
        launch.assert_called_once()
    else:
        with pytest.raises(RuntimeError, match="Predecessor stopped"):
            queue.main()
        assert json.loads(state_path.read_text())["phase"] == "failed"
        launch.assert_not_called()


def test_process_identity_tracks_live_process():
    assert queue.process_identity(queue.os.getpid()) is not None
    assert queue.process_identity(-1) is None
