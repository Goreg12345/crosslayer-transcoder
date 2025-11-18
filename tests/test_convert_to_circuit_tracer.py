from crosslayer_transcoder.utils.model_converters.circuit_tracer import (
    CircuitTracerConverter,
)

import pathlib
import shutil


def test_circuit_tracer_converter(jumprelu_clt_module):
    save_dir = pathlib.Path("clt_module")
    converter = CircuitTracerConverter(save_dir=save_dir)
    converter.convert_and_save(jumprelu_clt_module)

    assert save_dir.exists()
    assert (
        len(list(save_dir.glob("*.safetensors")))
        == jumprelu_clt_module.model.encoder.n_layers * 2
    )

    assert (save_dir / "config.yaml").exists()

    for layer in range(12):
        assert (save_dir / f"W_enc_{layer}.safetensors").exists()
        assert (save_dir / f"W_dec_{layer}.safetensors").exists()

    # cleanup
    shutil.rmtree(save_dir)
