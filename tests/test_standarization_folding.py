import torch
from einops import einsum
from torch.func import functional_call

from crosslayer_transcoder.model.clt import CrosslayerDecoder, Encoder
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)


def test_math_sanity_check():
    batch_size, d_acts, d_feats, n_layers = 100, 768, 32, 12
    torch.manual_seed(42)
    dtype = torch.float64
    batch = torch.randn((batch_size, 2, n_layers, d_acts), dtype=dtype)
    encoder = Encoder(d_acts=d_acts, d_features=d_feats, n_layers=n_layers).to(dtype)

    input_std = DimensionwiseInputStandardizer(
        n_layers=n_layers, activation_dim=d_acts
    ).to(dtype)
    input_std.initialize_from_batch(batch=batch)

    resid = batch[:, 0]
    W = encoder.W
    b = encoder.b
    mean, std = input_std.mean, input_std.std

    W_div_std = W / std.unsqueeze(-1)

    lhs = (
        einsum(
            (resid - mean) / std,
            W,
            "batch n_layers d_acts, n_layers d_acts d_features -> batch n_layers d_features",
        )
        + b
    )
    rhs = (
        einsum(
            resid,
            W_div_std,
            "batch n_layers d_acts, n_layers d_acts d_features -> batch n_layers d_features",
        )
        + b
        - einsum(
            mean,
            W_div_std,
            "n_layers d_acts, n_layers d_acts d_features -> n_layers d_features",
        )
    )

    assert torch.allclose(lhs, rhs, rtol=1e-7, atol=1e-9)


def test_encoder_standarization_folding():
    # create test tensors

    batch_size = 100
    d_acts = 768
    d_feats = 32
    n_layers = 12

    torch.manual_seed(42)

    dtype = torch.float64

    batch = torch.randn((batch_size, 2, n_layers, d_acts), dtype=dtype)

    encoder = Encoder(d_acts=d_acts, d_features=d_feats, n_layers=n_layers).to(dtype)
    input_std = DimensionwiseInputStandardizer(
        n_layers=n_layers, activation_dim=d_acts
    ).to(dtype)

    input_std.initialize_from_batch(batch=batch)

    # run tensors through the orignal forward passes

    resid = batch[:, 0]

    acts = input_std(resid)

    pre_actvs = encoder(acts)

    # fold in

    w_enc_folded, b_enc_folded = input_std.fold_in_encoder(encoder.W, encoder.b)
    # run inputs with folded

    folded_params = {
        "W": torch.nn.Parameter(w_enc_folded.clone().detach()),
        "b": torch.nn.Parameter(b_enc_folded.clone().detach()),
    }
    pre_actvs_folded = functional_call(encoder, folded_params, resid, {"layer": "all"})

    diff = pre_actvs - pre_actvs_folded
    print("max diff:", diff.max().item(), "mean diff:", diff.mean().item())

    # test equality
    assert torch.allclose(pre_actvs, pre_actvs_folded, rtol=1e-7, atol=1e-9)


def test_decoder_standarization_folding():
    # create test tensors

    batch_size = 100
    d_acts = 768
    d_feats = 32
    n_layers = 12

    torch.manual_seed(42)

    dtype = torch.float64

    encoder_batch = torch.randn((batch_size, 2, n_layers, d_acts), dtype=dtype)

    decoder = CrosslayerDecoder(
        d_acts=d_acts, d_features=d_feats, n_layers=n_layers
    ).to(dtype)
    output_std = DimensionwiseOutputStandardizer(
        n_layers=n_layers, activation_dim=d_acts
    ).to(dtype)

    output_std.initialize_from_batch(batch=encoder_batch)

    # run tensors through the orignal forward passes

    test_features = torch.randn((batch_size, n_layers, d_feats), dtype=dtype)

    recons_norm = decoder(test_features)

    recons = output_std(recons_norm)

    # fold in

    folded_params = {}
    for layer in range(n_layers):
        w_dec_folded = output_std.fold_in_decoder_weights_layer(
            decoder.get_parameter(f"W_{layer}"),
            layer,
        )
        assert w_dec_folded.shape == decoder.get_parameter(f"W_{layer}").shape
        folded_params[f"W_{layer}"] = torch.nn.Parameter(w_dec_folded.clone().detach())

    # fold bias
    b_dec_folded = decoder.b * output_std.std + output_std.mean
    folded_params["b"] = torch.nn.Parameter(b_dec_folded.clone().detach())

    print("folded_params:", folded_params.keys())

    recons_folded = functional_call(
        decoder, folded_params, test_features, {"layer": "all"}
    )

    diff = recons - recons_folded
    print("max diff:", diff.max().item(), "mean diff:", diff.mean().item())

    # test equality
    assert torch.allclose(recons, recons_folded, rtol=1e-7, atol=1e-9)
