import pytest


@pytest.fixture(scope="session")
def jumprelu_clt_module():
    from crosslayer_transcoder.metrics.dead_features import DeadFeatures
    from crosslayer_transcoder.model.clt import CrossLayerTranscoder
    from crosslayer_transcoder.model.clt_lightning import CrossLayerTranscoderModule

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

    dead_features = DeadFeatures(
        n_features=10_000,
        n_layers=12,
        return_per_layer=True,
        return_log_freqs=True,
        return_neuron_indices=True,
    )

    clt_module = CrossLayerTranscoderModule(
        model=model,
        dead_features=dead_features,
        learning_rate=1e-4,
        compile=True,
        lr_decay_step=80_000,
        lr_decay_factor=0.1,
        compute_dead_features=True,
        compute_dead_features_every=500,
    )

    return clt_module
