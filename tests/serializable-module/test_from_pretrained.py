from pathlib import Path

import torch

from crosslayer_transcoder.model.clt import CrossLayerTranscoder


def test_round_trip(config_dict, tmp_model_dir):
    original = CrossLayerTranscoder.from_config(config_dict)
    original.save_pretrained(Path(tmp_model_dir))

    loaded = CrossLayerTranscoder.from_pretrained(tmp_model_dir)

    assert isinstance(loaded, CrossLayerTranscoder)
    original_state = original.state_dict()
    loaded_state = loaded.state_dict()
    assert original_state.keys() == loaded_state.keys()
    for key in original_state:
        assert torch.allclose(original_state[key], loaded_state[key], equal_nan=True)


def test_accepts_string_path(config_dict, tmp_model_dir):
    original = CrossLayerTranscoder.from_config(config_dict)
    original.save_pretrained(Path(tmp_model_dir))

    loaded = CrossLayerTranscoder.from_pretrained(str(tmp_model_dir))

    assert isinstance(loaded, CrossLayerTranscoder)
