import atexit
import multiprocessing as mp
from multiprocessing import shared_memory
import os
from pathlib import Path

import pytest
import torch

from crosslayer_transcoder.data.shared_memory import SharedActivationBuffer


@pytest.fixture
def buffer(monkeypatch):
    # Spawn uses named semaphores, as in training; fork unlinks them immediately.
    context = mp.get_context("spawn")
    monkeypatch.setattr(mp, "RLock", context.RLock)
    # Keep real shared memory but avoid allocating CUDA pinned memory in this test.
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    buffer = SharedActivationBuffer(
        2, 2, 1, 1, batch_size=1, generation_batch_size=1, max_sequence_length=1,
    )
    yield buffer
    buffer.cleanup()


@pytest.mark.skipif(not Path("/dev/shm").is_dir(), reason="Requires Linux named semaphores")
def test_cleanup_releases_semaphores_while_buffer_is_still_referenced(buffer):
    paths = [
        Path("/dev/shm") / f"sem.{lock._semlock.name.lstrip('/')}"
        for lock in (buffer.buffer_lock, buffer.validity_lock)
    ]
    assert all(path.exists() for path in paths)

    buffer.cleanup()
    buffer.cleanup()

    assert not any(path.exists() for path in paths)


def test_teardown_and_atexit_can_both_clean_up(buffer):
    name = buffer.shm_name
    buffer.cleanup()
    buffer.cleanup()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)


def test_atexit_uses_idempotent_cleanup(monkeypatch, buffer):
    # The registered callback must be safe after explicit datamodule teardown.
    callbacks = []
    monkeypatch.setattr(atexit, "register", callbacks.append)
    other = SharedActivationBuffer(
        2, 2, 1, 1, batch_size=1, generation_batch_size=1, max_sequence_length=1,
    )
    other.cleanup()
    callbacks[0]()


def test_attached_worker_cleanup_leaves_owner_memory_available(buffer):
    attached = SharedActivationBuffer.__new__(SharedActivationBuffer)
    attached.__setstate__(buffer.__getstate__())
    attached._owner_pid = os.getpid() + 1  # Simulate attachment from a worker PID.
    attached.cleanup()

    still_available = shared_memory.SharedMemory(name=buffer.shm_name)
    still_available.close()
    assert buffer.buffer_tensor.shape == (2, 2, 1, 1)
    assert buffer.get_stats()["valid_samples"] == 0
