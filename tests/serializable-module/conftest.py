import tempfile
import pytest

from crosslayer_transcoder.model.serializable_module import ConfigDict


@pytest.fixture
def config_dict() -> ConfigDict:
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


@pytest.fixture
def tmp_model_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def tmp_config_path():
    with tempfile.NamedTemporaryFile(suffix=".yaml") as tmpfile:
        yield tmpfile.name


@pytest.fixture
def tmp_checkpoint_path():
    with tempfile.NamedTemporaryFile(suffix=".safetensors") as tmpfile:
        yield tmpfile.name
