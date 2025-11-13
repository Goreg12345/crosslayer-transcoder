import pathlib
import shutil
from circuit_tracer.transcoder.cross_layer_transcoder import load_clt

from crosslayer_transcoder.utils.model_converters.circuit_tracer import (
    CircuitTracerConverter,
)


def test_circuit_tracer_integration():
    """Verify uploaded model loads correctly."""
    from crosslayer_transcoder.utils.module_builder import build_module_from_config, yaml_to_config
    config = yaml_to_config("config/topk-clt-debug.yaml")
    clt_module = build_module_from_config(config)

    save_dir = pathlib.Path("clt_module_test")
    feature_input_hook = "blocks.{layer}.hook_resid_pre"
    feature_output_hook = "blocks.{layer}.hook_mlp_out"

    converter = CircuitTracerConverter(
        save_dir=save_dir,
        feature_input_hook=feature_input_hook,
        feature_output_hook=feature_output_hook,
    )
    converter.convert_and_save(clt_module)


    transcoder = load_clt(
        clt_path=save_dir.as_posix(),
        lazy_decoder=False,
        lazy_encoder=False,
        feature_input_hook=feature_input_hook,
        feature_output_hook=feature_output_hook,
    )

    assert transcoder.n_layers == clt_module.model.encoder.n_layers
    assert transcoder.d_transcoder == clt_module.model.encoder.d_features
    assert transcoder.d_model == clt_module.model.encoder.d_acts
    assert transcoder.feature_input_hook == feature_input_hook
    assert transcoder.feature_output_hook == feature_output_hook

    # cleanup
    shutil.rmtree(save_dir)
