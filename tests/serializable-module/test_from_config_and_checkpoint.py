from pathlib import Path

import pytest
import yaml
from safetensors.torch import save_file

from crosslayer_transcoder.model.clt import CrossLayerTranscoder


def test_missing_checkpoint(config_dict, tmp_model_dir):
    config_path = Path(tmp_model_dir) / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"model": config_dict}, f)

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        CrossLayerTranscoder.from_config_and_checkpoint(
            config_path, Path(tmp_model_dir) / "checkpoint.safetensors"
        )


def test_missing_config(config_dict, tmp_model_dir):
    model = CrossLayerTranscoder.from_config(config_dict)
    checkpoint_path = Path(tmp_model_dir) / "checkpoint.safetensors"
    save_file(model.state_dict(), checkpoint_path)

    with pytest.raises(FileNotFoundError, match="Config file not found"):
        CrossLayerTranscoder.from_config_and_checkpoint(
            Path(tmp_model_dir) / "config.yaml", checkpoint_path
        )


def test_missing_model_key_in_config(config_dict, tmp_model_dir):
    model = CrossLayerTranscoder.from_config(config_dict)
    dir_path = Path(tmp_model_dir)
    config_path = dir_path / "config.yaml"
    checkpoint_path = dir_path / "checkpoint.safetensors"

    with open(config_path, "w") as f:
        yaml.dump({"wrong_key": config_dict}, f)
    save_file(model.state_dict(), checkpoint_path)

    with pytest.raises(ValueError, match="Model config not found"):
        CrossLayerTranscoder.from_config_and_checkpoint(config_path, checkpoint_path)


def test_is_folded_true(config_dict, tmp_model_dir):
    model = CrossLayerTranscoder.from_config(config_dict)
    dir_path = Path(tmp_model_dir)
    config_path = dir_path / "config.yaml"
    checkpoint_path = dir_path / "checkpoint.safetensors"

    config_with_folded = config_dict.copy()
    config_with_folded["is_folded"] = True
    with open(config_path, "w") as f:
        yaml.dump({"model": config_with_folded}, f)
    save_file(model.state_dict(), checkpoint_path)

    loaded = CrossLayerTranscoder.from_config_and_checkpoint(
        config_path, checkpoint_path
    )

    assert loaded._is_folded is True


def test_is_folded_false(config_dict, tmp_model_dir):
    model = CrossLayerTranscoder.from_config(config_dict)
    dir_path = Path(tmp_model_dir)
    config_path = dir_path / "config.yaml"
    checkpoint_path = dir_path / "checkpoint.safetensors"

    config_with_folded = config_dict.copy()
    config_with_folded["is_folded"] = False
    with open(config_path, "w") as f:
        yaml.dump({"model": config_with_folded}, f)
    save_file(model.state_dict(), checkpoint_path)

    loaded = CrossLayerTranscoder.from_config_and_checkpoint(
        config_path, checkpoint_path
    )

    assert loaded._is_folded is False


def test_is_folded_defaults_false_when_missing(config_dict, tmp_model_dir):
    model = CrossLayerTranscoder.from_config(config_dict)
    dir_path = Path(tmp_model_dir)
    config_path = dir_path / "config.yaml"
    checkpoint_path = dir_path / "checkpoint.safetensors"

    with open(config_path, "w") as f:
        yaml.dump({"model": config_dict}, f)
    save_file(model.state_dict(), checkpoint_path)

    loaded = CrossLayerTranscoder.from_config_and_checkpoint(
        config_path, checkpoint_path
    )

    assert loaded._is_folded is False
