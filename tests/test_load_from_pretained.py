import tempfile
from pathlib import Path

import pytest

from crosslayer_transcoder.model.clt import (
    CrossLayerTranscoder,
    CrosslayerDecoder,
    Encoder,
)
from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)


@pytest.fixture
def model_and_pretrained_dir():
    """Create a model, save to temp dir, yield both for testing."""
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

        yield model, pretrained_dir


def test_load_from_pretrained(model_and_pretrained_dir):
    original_model, pretrained_dir = model_and_pretrained_dir

    loaded_model = CrossLayerTranscoder.from_pretrained(pretrained_dir)

    assert isinstance(loaded_model, CrossLayerTranscoder)
    assert loaded_model.encoder.d_acts == 110
    assert loaded_model.encoder.d_features == 32
    assert loaded_model.decoder.n_layers == 2

    for (name1, p1), (name2, p2) in zip(
        original_model.named_parameters(), loaded_model.named_parameters()
    ):
        assert name1 == name2
        # NOTE: saving folds so we can't check equality rn
        assert p1.shape == p2.shape, f"Parameter {name1} shape doesn't match"
