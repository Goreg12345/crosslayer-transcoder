import torch
from einops import einsum
from torch.func import functional_call

from crosslayer_transcoder.model.clt import CrossLayerTranscoder, CrosslayerDecoder, Decoder, Encoder
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

    input_std = DimensionwiseInputStandardizer(
        n_layers=N_LAYERS, activation_dim=D_ACTS
    ).to(DTYPE)
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
    input_std = DimensionwiseInputStandardizer(
        n_layers=N_LAYERS, activation_dim=D_ACTS
    ).to(DTYPE)

    input_std.initialize_from_batch(batch=BATCH)

    resid = BATCH[:, 0]

    acts = input_std(resid)

    pre_actvs = encoder(acts)

    encoder.fold(input_std)
    assert encoder._is_folded

    pre_actvs_folded = encoder(resid)

    assert_allclose(pre_actvs, pre_actvs_folded)

def test_decoder_standarization_folding():
    decoder = Decoder(
        d_acts=D_ACTS, d_features=D_FEATS, n_layers=N_LAYERS
    ).to(DTYPE)
    output_std = DimensionwiseOutputStandardizer(
        n_layers=N_LAYERS, activation_dim=D_ACTS
    ).to(DTYPE)

    output_std.initialize_from_batch(batch=BATCH)

    test_features = torch.randn((BATCH_SIZE, N_LAYERS, D_FEATS), dtype=DTYPE)

    recons_norm = decoder(test_features)

    recons = output_std(recons_norm)

    decoder.fold(output_std)
    assert decoder._is_folded

    recons_folded = decoder(test_features)

    assert_allclose(recons, recons_folded)

def test_crosslayer_decoder_standarization_folding():
    decoder = CrosslayerDecoder(
        d_acts=D_ACTS, d_features=D_FEATS, n_layers=N_LAYERS
    ).to(DTYPE)
    output_std = DimensionwiseOutputStandardizer(
        n_layers=N_LAYERS, activation_dim=D_ACTS
    ).to(DTYPE)

    output_std.initialize_from_batch(batch=BATCH)

    test_features = torch.randn((BATCH_SIZE, N_LAYERS, D_FEATS), dtype=DTYPE)

    recons_norm = decoder(test_features)

    recons = output_std(recons_norm)

    decoder.fold(output_std)
    assert decoder._is_folded

    recons_folded = decoder(test_features)
    assert_allclose(recons, recons_folded)

def test_crosslayer_transcoder_standarization_folding():
    input_std = DimensionwiseInputStandardizer(
        n_layers=N_LAYERS, activation_dim=D_ACTS
    ).to(DTYPE)
    output_std = DimensionwiseOutputStandardizer(
        n_layers=N_LAYERS, activation_dim=D_ACTS
    ).to(DTYPE)
    transcoder = CrossLayerTranscoder(
        nonlinearity=torch.nn.Identity(),
        encoder=Encoder(d_acts=D_ACTS, d_features=D_FEATS, n_layers=N_LAYERS),
        decoder=CrosslayerDecoder(d_acts=D_ACTS, d_features=D_FEATS, n_layers=N_LAYERS),
        input_standardizer=input_std,
        output_standardizer=output_std,
    ).to(DTYPE)
    transcoder.initialize_standardizers(batch=BATCH)

    (_, _, recons_norm, recons) = transcoder(BATCH[:, 0])

    transcoder.fold()
    assert transcoder._is_folded

    (_, _, recons_norm_folded, recons_folded) = transcoder(BATCH[:, 0])

    assert_allclose(recons, recons_folded)