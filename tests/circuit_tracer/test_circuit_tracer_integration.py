import pathlib
import shutil

import yaml
from circuit_tracer.transcoder.cross_layer_transcoder import load_clt

from crosslayer_transcoder.model.clt import CrossLayerTranscoder
from crosslayer_transcoder.utils.model_converters.circuit_tracer import (
    CircuitTracerConverter,
)


def test_circuit_tracer_integration():
    with open("config/circuit-tracer.yaml", "r") as f:
        config = yaml.safe_load(f)
        model_config = config["model"]["init_args"]["model"]
        clt = CrossLayerTranscoder.from_config(model_config)

    save_dir = pathlib.Path("clt_module_test")
    feature_input_hook = "hook_resid_mid"
    feature_output_hook = "hook_mlp_out"

    converter = CircuitTracerConverter(
        save_dir=save_dir,
        feature_input_hook=feature_input_hook,
        feature_output_hook=feature_output_hook,
    )
    converter.export(clt)

    transcoder = load_clt(
        clt_path=save_dir.as_posix(),
        lazy_decoder=False,
        lazy_encoder=False,
        feature_input_hook=feature_input_hook,
        feature_output_hook=feature_output_hook,
    )

    assert transcoder.n_layers == clt.encoder.n_layers
    assert transcoder.d_transcoder == clt.encoder.d_features
    assert transcoder.d_model == clt.encoder.d_acts
    assert transcoder.feature_input_hook == feature_input_hook
    assert transcoder.feature_output_hook == feature_output_hook

    # cleanup
    shutil.rmtree(save_dir)
