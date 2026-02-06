import logging
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file

from crosslayer_transcoder.model import (
    CrossLayerTranscoder,
    PerLayerTopK,
)
from crosslayer_transcoder.model.clt import Decoder
from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.utils.model_converters.model_converter import (
    ModelConverter,
)

logger = logging.getLogger(__name__)


class CircuitTracerConverter(ModelConverter):
    def __init__(
        self,
        save_dir: str,
        feature_input_hook: str = "hook_resid_mid",
        feature_output_hook: str = "hook_mlp_out",
    ):
        self.save_dir = Path(save_dir)
        self.feature_input_hook = feature_input_hook
        self.feature_output_hook = feature_output_hook

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _create_plt_files(self, encoder, decoder, nonlinearity, n_layers):
        # decoder shape: d_transcoder, d_model

        file_dicts = []

        for source_layer in range(n_layers):
            layer_dict = {
                "W_dec": decoder["W"][source_layer],
                "W_enc": encoder["W"][source_layer],
                "b_dec": decoder["b"][source_layer],
                "b_enc": encoder["b"][source_layer],
            }
            if isinstance(nonlinearity, JumpReLU):
                layer_dict["activation_function.threshold"] = nonlinearity.theta[
                    :, source_layer, :
                ]
            elif isinstance(nonlinearity, PerLayerTopK):
                layer_dict["activation_function.k"] = torch.tensor(nonlinearity.k)
            file_dicts.append(
                (layer_dict, f"{self.save_dir}/layer_{source_layer}.safetensors")
            )

        return file_dicts

    def _create_clt_files(
        self,
        encoder,
        decoder,
        nonlinearity,
        n_layers,
    ):
        file_dicts = []

        for source_layer in range(n_layers):
            layer_encoder_dict = {
                f"W_enc_{source_layer}": encoder["W"][source_layer],
                f"b_enc_{source_layer}": encoder["b"][source_layer],
                f"b_dec_{source_layer}": decoder["b"][source_layer],
            }

            if isinstance(nonlinearity, JumpReLU):
                layer_encoder_dict[f"threshold_{source_layer}"] = nonlinearity.theta[
                    :, source_layer, :
                ]
            if isinstance(nonlinearity, PerLayerTopK):
                layer_encoder_dict[f"k_{source_layer}"] = torch.tensor(nonlinearity.k)

            file_dicts.append(
                (
                    layer_encoder_dict,
                    f"{self.save_dir}/W_enc_{source_layer}.safetensors",
                )
            )

            decoder_dict = {f"W_dec_{source_layer}": decoder["W"][source_layer]}

            file_dicts.append(
                (decoder_dict, f"{self.save_dir}/W_dec_{source_layer}.safetensors")
            )

        return file_dicts

    def export(self, model: CrossLayerTranscoder, dtype: torch.dtype = torch.bfloat16):
        is_per_layer_decoder = isinstance(model.decoder, Decoder)
        config = {
            "model_name": "PLACEHOLDER MODEL_NAME",
            "feature_input_hook": self.feature_input_hook,
            "feature_output_hook": self.feature_output_hook,
            "model_kind": "transcoder_set"
            if is_per_layer_decoder
            else "cross_layer_transcoder",
        }

        model_dict = model.to_circuit_tracer()

        n_layers = model.encoder.n_layers

        if model_dict["is_per_layer_decoder"]:
            file_dicts = self._create_plt_files(
                model_dict["encoder"],
                model_dict["decoder"],
                model.nonlinearity,
                n_layers,
            )
        else:
            file_dicts = self._create_clt_files(
                model_dict["encoder"],
                model_dict["decoder"],
                model.nonlinearity,
                n_layers,
            )

        # TODO: convert to dtype and CPU (?)

        for file_dict, file_path in file_dicts:
            save_file(file_dict, file_path)

        with open(self.save_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
