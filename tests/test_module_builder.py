def test_build_module_from_config(clt_module):
    from crosslayer_transcoder.utils.module_builder import build_module_from_config, yaml_to_config
    config = yaml_to_config("config/default.yaml")
    model = build_module_from_config(config)

    assert model.model.encoder.n_layers == clt_module.model.encoder.n_layers
    assert model.model.decoder.n_layers == clt_module.model.decoder.n_layers


