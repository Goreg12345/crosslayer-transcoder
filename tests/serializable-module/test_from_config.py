# Tests needed
# 1. from config
# 2. save pretained
# 3. from pretrained
# 4. from config and checkpoint

import pytest

from crosslayer_transcoder.model.clt import (
    CrossLayerTranscoder,
    Encoder,
)


def test_from_config_success(config_dict):
    module = CrossLayerTranscoder.from_config(config_dict)

    assert isinstance(module, CrossLayerTranscoder)


def test_from_config_bad_class_path(config_dict):
    with pytest.raises(ValueError):
        Encoder.from_config(config_dict)


def test_from_config_class_path_dne(config_dict):
    config_dict["init_args"]["encoder"]["class_path"] = "path/that/doesnt/exist.py"

    with pytest.raises(Exception):
        CrossLayerTranscoder.from_config(config_dict)


def test_init_args_missing(config_dict):
    del config_dict["init_args"]["encoder"]

    with pytest.raises(Exception):
        CrossLayerTranscoder.from_config(config_dict)
