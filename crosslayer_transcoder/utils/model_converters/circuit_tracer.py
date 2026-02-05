from pathlib import Path

import einops
from tqdm import tqdm
import torch
import yaml
from safetensors.torch import save_file

from crosslayer_transcoder.model import (
    BatchTopK,
    CrossLayerTranscoder,
    PerLayerBatchTopK,
    PerLayerTopK,
)
from crosslayer_transcoder.utils.model_converters.model_converter import (
    ModelConverter,
)
from crosslayer_transcoder.model.jumprelu import JumpReLU

import logging

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

    def convert_and_save(
        self, model: CrossLayerTranscoder, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        if isinstance(model.nonlinearity, (PerLayerTopK, BatchTopK, PerLayerBatchTopK)):
            logger.warning(
                "TopK nonlinearity is not supported by circuit-tracer. Skipping conversion."
            )
            raise ValueError("TopK nonlinearity is not supported by circuit-tracer.")

        # NOTE: this mutates the model in-place. Potentially bad, but a tradeoff for copying a huge model.
        model.fold()

        encoder = model.encoder
        decoder = model.decoder
        nonlinearity = model.nonlinearity
        n_layers = encoder.n_layers
        d_acts = encoder.d_acts  # -> circuit-tracer.d_model
        d_features = encoder.d_features  # -> circuit-tracer.d_transcoder

        rearranged_W_enc = einops.rearrange(
            encoder.get_parameter("W").to(dtype),
            "n_layers d_acts d_features -> n_layers d_features d_acts",
        ).contiguous()

        for source_layer in tqdm(
            range(encoder.n_layers), desc="Converting CLT encoder"
        ):
            layer_encoder_dict = {
                f"W_enc_{source_layer}": rearranged_W_enc[source_layer].cpu().to(dtype),
                f"b_enc_{source_layer}": encoder.get_parameter("b")[source_layer]
                .cpu()
                .to(dtype),
                f"b_dec_{source_layer}": decoder.get_parameter("b")[source_layer]
                .cpu()
                .to(dtype),
            }
            if isinstance(nonlinearity, JumpReLU):
                layer_encoder_dict[f"threshold_{source_layer}"] = (
                    nonlinearity.theta[:, source_layer, :].cpu().to(dtype)
                )

            save_file(
                layer_encoder_dict, f"{self.save_dir}/W_enc_{source_layer}.safetensors"
            )

            output_dec_i = torch.zeros([d_features, n_layers - source_layer, d_acts])

            for k in range(source_layer, n_layers):
                # get decoder mat for layer i --> k
                decoder_w_k = decoder.get_parameter(f"W_{k}")

                dec_i_k = decoder_w_k.to(dtype)[source_layer, ...]
                assert dec_i_k.shape == (
                    d_features,
                    d_acts,
                )
                output_dec_i[:, k - source_layer, ...] = dec_i_k.cpu().to(dtype)

            decoder_dict = {f"W_dec_{source_layer}": output_dec_i}

            save_file(
                decoder_dict,
                f"{self.save_dir}/W_dec_{source_layer}.safetensors",
            )

        config = {
            "model_kind": "cross_layer_transcoder",
            "feature_input_hook": self.feature_input_hook,
            "feature_output_hook": self.feature_output_hook,
        }

        with open(self.save_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
