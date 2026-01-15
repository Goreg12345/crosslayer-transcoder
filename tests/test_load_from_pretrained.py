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
def test_load_from_pretrained(decoder_cls, nonlinearity):
    model = CrossLayerTranscoder(
        nonlinearity=nonlinearity,
        input_standardizer=DimensionwiseInputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS),
        output_standardizer=DimensionwiseOutputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS),
        encoder=Encoder(d_acts=D_ACTS, d_features=D_FEATURES, n_layers=N_LAYERS),
        decoder=decoder_cls(d_acts=D_ACTS, d_features=D_FEATURES, n_layers=N_LAYERS),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        pretrained_dir = Path(tmpdir)
        model.save_pretrained(pretrained_dir)

        loaded_model = CrossLayerTranscoder.from_pretrained(pretrained_dir)

        assert isinstance(loaded_model, CrossLayerTranscoder)
        assert loaded_model.encoder.d_acts == D_ACTS
        assert loaded_model.encoder.d_features == D_FEATURES
        assert loaded_model.decoder.n_layers == N_LAYERS

        # Verify weights have same shapes
        for (name1, p1), (name2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
            assert name1 == name2
            assert p1.shape == p2.shape, f"Parameter {name1} shape doesn't match"
