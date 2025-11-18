def test_build_module_from_config():
    from crosslayer_transcoder.utils.module_builder import build_module_from_config, yaml_to_config
    config = yaml_to_config("config/default.yaml")
    build_module_from_config(config)


