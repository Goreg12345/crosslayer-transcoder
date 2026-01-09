from pathlib import Path
import tempfile

from numpy import allclose

from crosslayer_transcoder.model.clt import CrossLayerTranscoder
from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)
from crosslayer_transcoder.model.clt import Encoder
from crosslayer_transcoder.model.clt import CrosslayerDecoder


def test_save_to_pretrained_runs():
    model = CrossLayerTranscoder(
        nonlinearity=JumpReLU(theta=0.03, bandwidth=0.01, n_layers=2, d_features=32),
        input_standardizer=DimensionwiseInputStandardizer(
            n_layers=2, activation_dim=110
        ),
        output_standardizer=DimensionwiseOutputStandardizer(
            n_layers=2, activation_dim=110
        ),
        encoder=Encoder(d_acts=110, d_features=32, n_layers=2),
        decoder=CrosslayerDecoder(d_acts=110, d_features=32, n_layers=2),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        pretrained_dir = Path(tmpdir)

        model.save_pretrained(pretrained_dir)

        assert (pretrained_dir / "config.yaml").exists()
        assert (pretrained_dir / "checkpoint.safetensors").exists()


def test_load_from_pretrained_config():
    model = CrossLayerTranscoder(
        nonlinearity=JumpReLU(theta=0.03, bandwidth=0.01, n_layers=2, d_features=32),
        input_standardizer=DimensionwiseInputStandardizer(
            n_layers=2, activation_dim=110
        ),
        output_standardizer=DimensionwiseOutputStandardizer(
            n_layers=2, activation_dim=110
        ),
        encoder=Encoder(d_acts=110, d_features=32, n_layers=2),
        decoder=CrosslayerDecoder(d_acts=110, d_features=32, n_layers=2),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        pretrained_dir = Path(tmpdir)

        model.save_pretrained(pretrained_dir)

        assert (pretrained_dir / "config.yaml").exists()
        assert (pretrained_dir / "checkpoint.safetensors").exists()

        import yaml

        with open(pretrained_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        model_config = config["model"]
        assert (
            model_config["class_path"]
            == "crosslayer_transcoder.model.clt.CrossLayerTranscoder"
        )

        decoder = model_config["init_args"]["decoder"]
        assert (
            decoder["class_path"] == "crosslayer_transcoder.model.clt.CrosslayerDecoder"
        )
        assert decoder["init_args"]["d_acts"] == 110
        assert decoder["init_args"]["d_features"] == 32
        assert decoder["init_args"]["n_layers"] == 2

        encoder = model_config["init_args"]["encoder"]
        assert encoder["class_path"] == "crosslayer_transcoder.model.clt.Encoder"
        assert encoder["init_args"]["d_acts"] == 110
        assert encoder["init_args"]["d_features"] == 32
        assert encoder["init_args"]["n_layers"] == 2

        nonlinearity = model_config["init_args"]["nonlinearity"]
        assert (
            nonlinearity["class_path"]
            == "crosslayer_transcoder.model.jumprelu.JumpReLU"
        )
        assert nonlinearity["init_args"]["d_features"] == 32
        assert nonlinearity["init_args"]["n_layers"] == 2
        assert allclose(nonlinearity["init_args"]["theta"], 0.03)
        assert allclose(nonlinearity["init_args"]["bandwidth"], 0.01)
