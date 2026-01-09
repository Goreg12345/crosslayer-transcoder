import torch
from einops import einsum
from torch.func import functional_call

from crosslayer_transcoder.model.clt import CrosslayerDecoder, Encoder
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)

torch.manual_seed(42)
DTYPE = torch.float64
BATCH_SIZE, D_ACTS, D_FEATS, N_LAYERS = 100, 768, 32, 12
BATCH = torch.randn((BATCH_SIZE, 2, N_LAYERS, D_ACTS), dtype=DTYPE)


def assert_allclose(lhs, rhs, rtol=1e-7, atol=1e-9):
    try:
        assert torch.allclose(lhs, rhs, rtol=rtol, atol=atol)
    except AssertionError as e:
        print("lhs shape:", lhs.shape)
        print("rhs shape:", rhs.shape)
        print("max diff:", (lhs - rhs).max().item())
        print("mean diff:", (lhs - rhs).mean().item())
        raise e from e


def test_math_sanity_check():
    encoder = Encoder(d_acts=D_ACTS, d_features=D_FEATS, n_layers=N_LAYERS).to(DTYPE)

    input_std = DimensionwiseInputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS).to(DTYPE)
    input_std.initialize_from_batch(batch=BATCH)

    resid = BATCH[:, 0]
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

    assert_allclose(lhs, rhs)


def test_encoder_standarization_folding():
    encoder = Encoder(d_acts=D_ACTS, d_features=D_FEATS, n_layers=N_LAYERS).to(DTYPE)
    input_std = DimensionwiseInputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS).to(DTYPE)

    input_std.initialize_from_batch(batch=BATCH)

    resid = BATCH[:, 0]

    acts = input_std(resid)

    pre_actvs = encoder(acts)

    w_enc_folded, b_enc_folded = input_std.fold_in_encoder(encoder.W, encoder.b)

    folded_params = {
        "W": torch.nn.Parameter(w_enc_folded.clone().detach()),
        "b": torch.nn.Parameter(b_enc_folded.clone().detach()),
    }
    pre_actvs_folded = functional_call(encoder, folded_params, resid, {"layer": "all"})

    assert_allclose(pre_actvs, pre_actvs_folded)


def test_decoder_standarization_folding():
    decoder = CrosslayerDecoder(d_acts=D_ACTS, d_features=D_FEATS, n_layers=N_LAYERS).to(DTYPE)
    output_std = DimensionwiseOutputStandardizer(n_layers=N_LAYERS, activation_dim=D_ACTS).to(DTYPE)

    output_std.initialize_from_batch(batch=BATCH)

    test_features = torch.randn((BATCH_SIZE, N_LAYERS, D_FEATS), dtype=DTYPE)

    recons_norm = decoder(test_features)

    recons = output_std(recons_norm)

    folded_params = {}
    for layer in range(N_LAYERS):
        w_dec_folded = output_std.fold_in_decoder_weights_layer(
            decoder.get_parameter(f"W_{layer}"),
            layer,
        )
        assert w_dec_folded.shape == decoder.get_parameter(f"W_{layer}").shape
        folded_params[f"W_{layer}"] = torch.nn.Parameter(w_dec_folded.clone().detach())

    b_dec_folded = output_std.fold_in_decoder_bias(decoder.b)
    folded_params["b"] = torch.nn.Parameter(b_dec_folded.clone().detach())

    recons_folded = functional_call(decoder, folded_params, test_features, {"layer": "all"})

    assert_allclose(recons, recons_folded)
