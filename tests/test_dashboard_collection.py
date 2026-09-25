import numpy as np
import pytest

from crosslayer_transcoder.dashboard.collection import GateCollection


def record(tokens, positions, features, values):
    return {
        key: np.asarray(value)
        for key, value in dict(
            tokens=tokens,
            positions=positions,
            features=features,
            values=values,
        ).items()
    }


def test_resume_ignores_uncommitted_shard_and_keeps_conversation_boundaries(tmp_path):
    settings = {"checkpoint": "fixed", "target_tokens": 7}
    store = GateCollection(tmp_path, settings)
    store.append([record([1, 2, 3], [0, 2], [2, 1], [0.5, 2.0])])
    (store.parts / "part-000001.npz.tmp").write_bytes(b"interrupted write")
    resumed = GateCollection(tmp_path, settings)
    assert (resumed.tokens, resumed.conversations, resumed.entries) == (3, 1, 2)
    resumed.append(
        [
            record([4, 5], [], [], []),
            record([6, 7], [0, 0, 1], [0, 1, 2], [1.0, 3.0, 2.0]),
        ]
    )
    np.testing.assert_array_equal(resumed.finish(), np.arange(1, 8))
    with np.load(tmp_path / "activations.npz") as data:
        np.testing.assert_array_equal(data["offsets"], [0, 3, 5, 7])
        np.testing.assert_array_equal(data["positions"], [0, 2, 5, 5, 6])
        np.testing.assert_array_equal(data["features"], [2, 1, 0, 1, 2])
        np.testing.assert_array_equal(data["values"], [0.5, 2.0, 1.0, 3.0, 2.0])
    assert GateCollection(tmp_path, settings).tokens == 7


def test_resume_rejects_different_checkpoint_or_sampling(tmp_path):
    GateCollection(tmp_path, {"checkpoint": "fixed", "target_tokens": 7})
    with pytest.raises(ValueError, match="settings changed"):
        GateCollection(tmp_path, {"checkpoint": "other", "target_tokens": 7})


def test_refuses_to_overwrite_existing_unsharded_sample(tmp_path):
    np.savez(tmp_path / "activations.npz", tokens=np.arange(3))
    with pytest.raises(ValueError, match="existing, unsharded"):
        GateCollection(tmp_path, {})
    assert not (tmp_path / "collection" / "settings.json").exists()
