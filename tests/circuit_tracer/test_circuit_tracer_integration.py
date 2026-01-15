import pathlib
import shutil
from unittest.mock import MagicMock
import torch
import yaml
from circuit_tracer.transcoder.cross_layer_transcoder import load_clt
import pytest

from crosslayer_transcoder.model.clt import CrossLayerTranscoder
from crosslayer_transcoder.utils.model_converters.circuit_tracer import (
    CircuitTracerConverter,
)


def test_circuit_tracer_integration():
    """Verify uploaded model loads correctly."""
    with open("config/circuit-tracer.yaml", "r") as f:
        config = yaml.safe_load(f)
        model_config = config["model"]["init_args"]["model"]
        clt_model = CrossLayerTranscoder.from_config(model_config)

    clt_module = MagicMock()
    clt_module.model = clt_model

    save_dir = pathlib.Path("clt_module_test")
    feature_input_hook = "hook_resid_mid"
    feature_output_hook = "hook_mlp_out"

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


def test_topk_nonlinearity():
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
                "class_path": "crosslayer_transcoder.model.topk.PerLayerTopK",
                "init_args": {
                    "k": 8,
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
    clt_model = CrossLayerTranscoder.from_config(model_config)
    clt_module = MagicMock()
    clt_module.model = clt_model
    save_dir = pathlib.Path("topk_clt_module_test")
    feature_input_hook = "hook_resid_mid"
    feature_output_hook = "hook_mlp_out"

    with pytest.raises(
        ValueError,
        match="TopK nonlinearity is not supported by circuit-tracer.",
    ):
        converter = CircuitTracerConverter(
            save_dir=save_dir,
            feature_input_hook=feature_input_hook,
            feature_output_hook=feature_output_hook,
        )
        converter.convert_and_save(clt_module)
