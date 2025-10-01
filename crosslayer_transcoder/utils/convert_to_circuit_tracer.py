# simple file to convert the lightning model to a circuit-tracer model
import logging
import os
from pathlib import Path
from typing import List, Union

import torch
import yaml
from huggingface_hub import upload_folder
from safetensors.torch import save_file

from crosslayer_transcoder.model.clt import (
    CrosslayerDecoder,
    Encoder,
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


def upload_to_hub(save_dir: str, repo_id: str, repo_type: str = "model"):
    from huggingface_hub import upload_folder

    print(f"Uploading {save_dir} to {repo_id} ({repo_type})")
    upload_folder(folder_path=save_dir, repo_id=repo_id, repo_type=repo_type)


if __name__ == "__main__":
    from crosslayer_transcoder.metrics.dead_features import DeadFeatures
    from crosslayer_transcoder.metrics.replacement_model_accuracy import (
        ReplacementModelAccuracy,
    )
    from crosslayer_transcoder.model.clt import (
        CrosslayerDecoder,
        CrossLayerTranscoder,
        Encoder,
    )
    from crosslayer_transcoder.model.clt_lightning import CrossLayerTranscoderModule
    from crosslayer_transcoder.model.jumprelu import JumpReLU
    from crosslayer_transcoder.model.standardize import (
        DimensionwiseInputStandardizer,
        DimensionwiseOutputStandardizer,
    )

    # Create components based on default.yaml config
    encoder = Encoder(d_acts=768, d_features=10_000, n_layers=12)

    decoder = CrosslayerDecoder(d_acts=768, d_features=10_000, n_layers=12)

    nonlinearity = JumpReLU(theta=0.03, bandwidth=0.01, n_layers=12, d_features=10_000)

    input_standardizer = DimensionwiseInputStandardizer(n_layers=12, activation_dim=768)

    output_standardizer = DimensionwiseOutputStandardizer(
        n_layers=12, activation_dim=768
    )

    model = CrossLayerTranscoder(
        encoder=encoder,
        decoder=decoder,
        nonlinearity=nonlinearity,
        input_standardizer=input_standardizer,
        output_standardizer=output_standardizer,
    )

    # replacement_model = ReplacementModelAccuracy(
    #     model_name="openai-community/gpt2", device_map="mps", loader_batch_size=2
    # )

    dead_features = DeadFeatures(
        n_features=10_000,
        n_layers=12,
        return_per_layer=True,
        return_log_freqs=True,
        return_neuron_indices=True,
    )

    clt_module = CrossLayerTranscoderModule(
        model=model,
        # replacement_model=replacement_model,
        dead_features=dead_features,
        learning_rate=1e-4,
        compile=True,
        lr_decay_step=80_000,
        lr_decay_factor=0.1,
        compute_dead_features=True,
        compute_dead_features_every=500,
    )

    convert_model_to_circuit_tracer(clt_module, "clt_module")

    upload_folder(
        folder_path="clt_module", repo_id="jiito/clt_test_gpt2_zero", repo_type="model"
    )
    upload_folder(
        folder_path="clt_module/encoder",
        path_in_repo="encoder",
        repo_id="jiito/clt_test_gpt2_zero",
        repo_type="model",
    )
    upload_folder(
        folder_path="clt_module/decoder",
        path_in_repo="decoder",
        repo_id="jiito/clt_test_gpt2_zero",
        repo_type="model",
    )
