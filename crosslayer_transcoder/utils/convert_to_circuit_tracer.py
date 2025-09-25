# simple file to convert the lightning model to a circuit-tracer model
from typing import Union

from einops import einops
import torch
from torch.export.graph_signature import OutputKind

from crosslayer_transcoder.model.clt import CrosslayerDecoder
from crosslayer_transcoder.model.clt_lightning import (
    CrossLayerTranscoderModule,
    JumpReLUCrossLayerTranscoderModule,
    TopKCrossLayerTranscoderModule,
)


CLTModule = Union[
    CrossLayerTranscoderModule,
    JumpReLUCrossLayerTranscoderModule,
    TopKCrossLayerTranscoderModule,
]


def convert_decoder(decoder: CrosslayerDecoder):
    # TODO: convert the CLT to a circuit tracing CLT shape
    # our decoder shape: [i+1, d_features, d_acts]
    # expected W_{i} shape: [d_features, n_layers - i, d_acts]
    # From [here](https://github.com/safety-research/circuit-tracer/blob/2eff952df66400eeb1066595dc5567a7203c6acd/circuit_tracer/transcoder/cross_layer_transcoder.py#L103C38-L103C73)
    #

    d_feats = decoder.d_features
    d_acts = decoder.d_acts
    n_layers = decoder.n_layers

    # layer i decoder mat of shape [d_feats, n_layers - i, d_acts]
    decoder_dict = {}

    for i in range(n_layers):
        output_dec_i = torch.zeros([d_feats, n_layers - i, d_acts])

        for k in range(i, n_layers):
            # get decoder mat for layer i --> k
            decoder_w_k = decoder.get_parameter(f"W_{k}")
            dec_i_k = decoder_w_k[i, ...]
            assert dec_i_k.shape == (
                d_feats,
                d_acts,
            )
            output_dec_i[:, k - i, ...] = dec_i_k

        decoder_dict[f"W_dec_{i}"] = output_dec_i

    return decoder_dict


def convert_encoder():
    pass


# def convert_model_to_circuit_tracer(lightning_module: CLTModule):
def convert_model_to_circuit_tracer():
    # convert the lightning model to the circuit-tracer compatible shape

    mock_decoder = CrosslayerDecoder(728, 1023, 12)

    ct_decoder_dict = convert_decoder(mock_decoder)

    print(len(ct_decoder_dict))
    print(ct_decoder_dict["W_dec_0"].shape)
    pass


if __name__ == "__main__":
    convert_model_to_circuit_tracer()
