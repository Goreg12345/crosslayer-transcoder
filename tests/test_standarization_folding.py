import torch
from einops import einsum

from crosslayer_transcoder.model.clt import Encoder
from crosslayer_transcoder.model.standardize import DimensionwiseInputStandardizer


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

    pre_actvs_folded = einsum(
        resid,
        w_enc_folded,
        "batch_size n_layers d_acts, n_layers d_acts d_features -> batch_size n_layers d_features",
    )
    pre_actvs_folded = pre_actvs_folded.contiguous()

    pre_actvs_folded = pre_actvs_folded + b_enc_folded

    diff = pre_actvs - pre_actvs_folded
    print("max diff:", diff.max().item(), "mean diff:", diff.mean().item())

    # test equality
    assert torch.allclose(pre_actvs, pre_actvs_folded, rtol=1e-7, atol=1e-9)
