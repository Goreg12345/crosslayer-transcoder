import tempfile
from pathlib import Path

import pytest

from crosslayer_transcoder.model.clt import CrosslayerDecoder, CrossLayerTranscoder, Decoder, Encoder
from crosslayer_transcoder.model.jumprelu import JumpReLU, ReLU
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)
from crosslayer_transcoder.model.topk import BatchTopK, PerLayerBatchTopK, PerLayerTopK

N_LAYERS = 2
D_ACTS = 110
D_FEATURES = 32


@pytest.fixture
def standardizers():
    return (
        DimensionwiseInputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS),
        DimensionwiseOutputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS),
    )


@pytest.fixture
def encoder():
    return Encoder(d_acts=D_ACTS, d_features=D_FEATURES, n_layers=N_LAYERS)


DECODER_CLASSES = [Decoder, CrosslayerDecoder]

NONLINEARITIES = [
    JumpReLU(theta=0.03, bandwidth=0.01, n_layers=N_LAYERS, d_features=D_FEATURES),
    ReLU(),
    PerLayerTopK(k=8, n_layers=N_LAYERS),
    BatchTopK(k=8, e=0.01, n_layers=N_LAYERS),
    PerLayerBatchTopK(k=8, e=0.01, n_layers=N_LAYERS),
]


@pytest.mark.parametrize("decoder_cls", DECODER_CLASSES)
@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
def test_save_to_pretrained_runs(decoder_cls, nonlinearity, encoder, standardizers):
    model = CrossLayerTranscoder(
        nonlinearity=nonlinearity,
        input_standardizer=standardizers[0],
        output_standardizer=standardizers[1],
        encoder=encoder,
        decoder=decoder_cls(d_acts=D_ACTS, d_features=D_FEATURES, n_layers=N_LAYERS),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        pretrained_dir = Path(tmpdir)
        model.save_pretrained(pretrained_dir)

        assert (pretrained_dir / "config.yaml").exists()
        assert (pretrained_dir / "checkpoint.safetensors").exists()


@pytest.mark.parametrize("decoder_cls", DECODER_CLASSES)
@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
def test_save_to_pretrained_config_structure(decoder_cls, nonlinearity, encoder, standardizers):
    model = CrossLayerTranscoder(
        nonlinearity=nonlinearity,
        input_standardizer=standardizers[0],
        output_standardizer=standardizers[1],
        encoder=encoder,
        decoder=decoder_cls(d_acts=D_ACTS, d_features=D_FEATURES, n_layers=N_LAYERS),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        pretrained_dir = Path(tmpdir)
        model.save_pretrained(pretrained_dir)

        import yaml

        with open(pretrained_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        model_config = config["model"]
        assert model_config["class_path"] == "crosslayer_transcoder.model.clt.CrossLayerTranscoder"

        # Verify decoder class path
        decoder_config = model_config["init_args"]["decoder"]
        assert decoder_config["class_path"].endswith(decoder_cls.__name__)

        # Verify nonlinearity class path
        nonlinearity_config = model_config["init_args"]["nonlinearity"]
        assert nonlinearity_config["class_path"].endswith(nonlinearity.__class__.__name__)
