import pytest


@pytest.fixture(scope="session")
def jumprelu_clt():
    from crosslayer_transcoder.model.clt import CrossLayerTranscoder

    model_config = {
        "class_path": "crosslayer_transcoder.model.clt.CrossLayerTranscoder",
        "init_args": {
            "encoder": {
                "class_path": "crosslayer_transcoder.model.clt.Encoder",
                "init_args": {"d_acts": 768, "d_features": 10_000, "n_layers": 12},
            },
            "decoder": {
                "class_path": "crosslayer_transcoder.model.clt.CrosslayerDecoder",
                "init_args": {"d_acts": 768, "d_features": 10_000, "n_layers": 12},
            },
            "nonlinearity": {
                "class_path": "crosslayer_transcoder.model.jumprelu.JumpReLU",
                "init_args": {
                    "theta": 0.03,
                    "bandwidth": 0.01,
                    "n_layers": 12,
                    "d_features": 10_000,
                },
            },
            "input_standardizer": {
                "class_path": "crosslayer_transcoder.model.standardize.DimensionwiseInputStandardizer",
                "init_args": {"n_layers": 12, "activation_dim": 768},
            },
            "output_standardizer": {
                "class_path": "crosslayer_transcoder.model.standardize.DimensionwiseOutputStandardizer",
                "init_args": {"n_layers": 12, "activation_dim": 768},
            },
        },
    }

    model = CrossLayerTranscoder.from_config(model_config)

    return model
