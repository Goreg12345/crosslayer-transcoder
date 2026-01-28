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
from crosslayer_transcoder.model.serializable_module import (
    ConfigDict,
)


@pytest.fixture
def config() -> ConfigDict:
    return {
        "class_path": "crosslayer_transcoder.model.clt.CrossLayerTranscoder",
        "init_args": {
            "encoder": {
                "class_path": "crosslayer_transcoder.model.clt.Encoder",
                "init_args": {"d_acts": 110, "d_features": 32, "n_layers": 2},
            },
            "decoder": {
                "class_path": "crosslayer_transcoder.model.clt.CrosslayerDecoder",
                "init_args": {"d_acts": 110, "d_features": 32, "n_layers": 2},
            },
            "nonlinearity": {
                "class_path": "crosslayer_transcoder.model.jumprelu.JumpReLU",
                "init_args": {
                    "theta": 0.03,
                    "bandwidth": 0.01,
                    "n_layers": 2,
                    "d_features": 32,
                },
            },
            "input_standardizer": {
                "class_path": "crosslayer_transcoder.model.standardize.DimensionwiseInputStandardizer",
                "init_args": {"n_layers": 2, "activation_dim": 110},
            },
            "output_standardizer": {
                "class_path": "crosslayer_transcoder.model.standardize.DimensionwiseOutputStandardizer",
                "init_args": {"n_layers": 2, "activation_dim": 110},
            },
        },
    }


def test_from_config_success(config):
    module = CrossLayerTranscoder.from_config(config)

    assert isinstance(module, CrossLayerTranscoder)


def test_from_config_bad_class_path(config):
    with pytest.raises(ValueError):
        Encoder.from_config(config)


def test_from_config_class_path_dne(config):
    config["init_args"]["encoder"]["class_path"] = "path/that/doesnt/exist.py"

    with pytest.raises(Exception):
        CrossLayerTranscoder.from_config(config)


def test_init_args_missing(config):
    del config["init_args"]["encoder"]

    with pytest.raises(Exception):
        CrossLayerTranscoder.from_config(config)
