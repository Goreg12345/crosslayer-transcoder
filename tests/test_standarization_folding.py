import torch
from einops import einsum

from crosslayer_transcoder.model.clt import Encoder
from crosslayer_transcoder.model.standardize import DimensionwiseInputStandardizer


def test_encoder_standarization_folding():
    # create test tensors

    batch_size = 100
    d_acts = 768
    d_feats = 32
    n_layers = 12

    torch.manual_seed(42)

    batch = torch.randn((batch_size, 2, n_layers, d_acts))

    encoder = Encoder(d_acts=d_acts, d_features=d_feats, n_layers=n_layers)
    input_std = DimensionwiseInputStandardizer(n_layers=n_layers, activation_dim=d_acts)

    input_std.initialize_from_batch(batch=batch)

    assert (input_std.std > 1e-8).all(), "Std values too small"

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

    # test equality

    assert torch.allclose(pre_actvs, pre_actvs_folded)
