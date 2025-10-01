from crosslayer_transcoder.utils.convert_to_circuit_tracer import (
    convert_model_to_circuit_tracer,
)


def test_convert_to_circuit_tracer():
    from crosslayer_transcoder.metrics.dead_features import DeadFeatures
    from crosslayer_transcoder.model.clt import (
        CrosslayerDecoder,
        CrossLayerTranscoder,
        Encoder,
    )
    from crosslayer_transcoder.model.clt_lightning import CrossLayerTranscoderModule
    from crosslayer_transcoder.model.jumprelu import JumpReLU
    from crosslayer_transcoder.model.standardize import (
        DimensionwiseInputStandardizer,
        DimensionwiseOutputStandardizer,
    )

    # Create components based on default.yaml config
    encoder = Encoder(d_acts=768, d_features=10_000, n_layers=12)

    decoder = CrosslayerDecoder(d_acts=768, d_features=10_000, n_layers=12)

    nonlinearity = JumpReLU(theta=0.03, bandwidth=0.01, n_layers=12, d_features=10_000)

    input_standardizer = DimensionwiseInputStandardizer(n_layers=12, activation_dim=768)

    output_standardizer = DimensionwiseOutputStandardizer(
        n_layers=12, activation_dim=768
    )

    model = CrossLayerTranscoder(
        encoder=encoder,
        decoder=decoder,
        nonlinearity=nonlinearity,
        input_standardizer=input_standardizer,
        output_standardizer=output_standardizer,
    )

    # replacement_model = ReplacementModelAccuracy(
    #     model_name="openai-community/gpt2", device_map="mps", loader_batch_size=2
    # )

    dead_features = DeadFeatures(
        n_features=10_000,
        n_layers=12,
        return_per_layer=True,
        return_log_freqs=True,
        return_neuron_indices=True,
    )

    clt_module = CrossLayerTranscoderModule(
        model=model,
        # replacement_model=replacement_model,
        dead_features=dead_features,
        learning_rate=1e-4,
        compile=True,
        lr_decay_step=80_000,
        lr_decay_factor=0.1,
        compute_dead_features=True,
        compute_dead_features_every=500,
    )

    convert_model_to_circuit_tracer(clt_module, "clt_module")


if __name__ == "__main__":
    test_convert_to_circuit_tracer()
