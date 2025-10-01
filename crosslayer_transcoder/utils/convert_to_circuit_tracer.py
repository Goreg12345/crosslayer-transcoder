# simple file to convert the lightning model to a circuit-tracer model
import logging
from pathlib import Path
from typing import List, Union

import torch
import yaml
from safetensors.torch import save_file

from crosslayer_transcoder.model.clt import (
    CrosslayerDecoder,
)
from crosslayer_transcoder.model.clt_lightning import (
    CrossLayerTranscoderModule,
    JumpReLUCrossLayerTranscoderModule,
    TopKCrossLayerTranscoderModule,
)

logger = logging.getLogger(__name__)

CLTModule = Union[
    CrossLayerTranscoderModule,
    JumpReLUCrossLayerTranscoderModule,
    TopKCrossLayerTranscoderModule,
]


def add_decoder_bias(d: List[dict], decoder: CrosslayerDecoder):
    for i in range(decoder.n_layers):
        d[i][f"b_dec_{i}"] = (
            decoder.b[i].cpu() if hasattr(decoder, "b") else torch.zeros(decoder.d_acts)
        )
    return d


def save_decoder_dict(decoder_dict: dict, path: str):
    for key, value in decoder_dict.items():
        save_file({key: value}, f"{path}/{key}.safetensors")


def convert_model_to_circuit_tracer(
    lightning_module: CLTModule,
    save_dir: str,
    # TODO: check the hooks
    feature_input_hook: str = "blocks.{layer}.hook_resid_pre",
    feature_output_hook: str = "blocks.{layer}.hook_mlp_out",
):
    # convert the lightning model to the circuit-tracer compatible shape
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    encoder = lightning_module.model.encoder
    decoder = lightning_module.model.decoder
    nonlinearity = lightning_module.model.nonlinearity
    n_layers = encoder.n_layers
    d_acts = encoder.d_acts  # -> d_model
    d_features = encoder.d_features  # -> d_transcoder

    for source_layer in range(encoder.n_layers):
        # encoder
        layer_encoder_dict = {
            f"W_enc_{source_layer}": encoder.W[source_layer]
            .T.contiguous()
            .cpu(),  # Transpose!
            f"b_enc_{source_layer}": (
                encoder.b[source_layer].cpu()
                if hasattr(encoder, "b")
                else torch.zeros(encoder.d_features)
            ),
            f"b_dec_{source_layer}": (
                decoder.b[source_layer].cpu()
                if hasattr(decoder, "b")
                else torch.zeros(d_acts)
            ),
            f"threshold_{source_layer}": (
                nonlinearity.theta.cpu()
                if hasattr(nonlinearity, "theta")
                else torch.zeros(d_features)
            ),
        }

        # TODO: check names
        save_file(layer_encoder_dict, f"{save_path}/W_enc_{source_layer}.safetensors")

        # decoder
        output_dec_i = torch.zeros([d_features, n_layers - source_layer, d_acts])

        for k in range(source_layer, n_layers):
            # get decoder mat for layer i --> k
            decoder_w_k = decoder.get_parameter(f"W_{k}")
            dec_i_k = decoder_w_k[source_layer, ...]
            assert dec_i_k.shape == (
                d_features,
                d_acts,
            )
            output_dec_i[:, k - source_layer, ...] = dec_i_k.cpu()

        decoder_dict = {f"W_dec_{source_layer}": output_dec_i}

        save_file(decoder_dict, f"{save_path}/W_dec_{source_layer}.safetensors")

    # Create config
    config = {
        "model_kind": "cross_layer_transcoder",
        "feature_input_hook": feature_input_hook,
        "feature_output_hook": feature_output_hook,
    }

    with open(save_path / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
