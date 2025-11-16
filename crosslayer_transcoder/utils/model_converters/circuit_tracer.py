from pathlib import Path

import einops
import torch
import yaml
from safetensors.torch import save_file

from crosslayer_transcoder.utils.model_converters.model_converter import (
    ModelConverter,
)
from crosslayer_transcoder.model.jumprelu import JumpReLU


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

    def convert_and_save(self, model, dtype=torch.bfloat16):
        encoder = model.model.encoder
        input_standardizer = model.model.input_standardizer
        output_standardizer = model.model.output_standardizer
        decoder = model.model.decoder
        nonlinearity = model.model.nonlinearity
        n_layers = encoder.n_layers
        d_acts = encoder.d_acts  # -> circuit-tracer.d_model
        d_features = encoder.d_features  # -> circuit-tracer.d_transcoder

        # TODO: maybe pass in dtype
        W_enc_folded, b_enc_folded = input_standardizer.fold_in_encoder(
            encoder.W.to(dtype), encoder.b.to(dtype)
        )

        b_dec_folded = output_standardizer.fold_in_decoder_bias(decoder.b.to(dtype))

        # encoder
        for source_layer in range(encoder.n_layers):
            rearranged_W_enc = einops.rearrange(
                W_enc_folded[source_layer],
                "d_acts d_features -> d_features d_acts",
            ).contiguous()
            layer_encoder_dict = {
                f"W_enc_{source_layer}": rearranged_W_enc.cpu().to(dtype),
                f"b_enc_{source_layer}": (b_enc_folded[source_layer].cpu().to(dtype)),
                f"b_dec_{source_layer}": (b_dec_folded[source_layer].cpu().to(dtype)),
            }
            # TODO: double check non-linearity compatibility
            print(nonlinearity)
            if isinstance(nonlinearity, JumpReLU):
                print(nonlinearity.theta.shape)
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

                # fold in output standardization for decoder weights using standardizer method
                decoder_w_k_folded = output_standardizer.fold_in_decoder_weights_layer(
                    decoder_w_k.to(dtype), k
                )
                dec_i_k = decoder_w_k_folded[source_layer, ...]
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
