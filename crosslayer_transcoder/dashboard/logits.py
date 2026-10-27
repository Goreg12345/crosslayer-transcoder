"""Input-dependent MOLT output and frozen-final-RMS direct logit attribution."""

import torch


def transform_contribution(normalized_inputs, gates, v, u, output_std):
    """Marginal raw-space output; the shared output mean cancels in on-minus-off."""
    return ((normalized_inputs @ v) @ u) * gates[:, None] * output_std


def frozen_rms_readout(contribution, final_rms, final_norm_weight):
    """Hold the baseline final RMS denominator fixed; ignore downstream layers."""
    return contribution * final_norm_weight / final_rms[:, None]


def signed_topk(logits, k):
    """Return only strictly positive / negative entries (never mislabel a sign)."""
    top_values, top_ids = torch.topk(logits, min(k, logits.numel()))
    bottom_values, bottom_ids = torch.topk(
        logits, min(k, logits.numel()), largest=False
    )
    return {
        "top_token_ids": top_ids[top_values > 0].tolist(),
        "top_logits": top_values[top_values > 0].tolist(),
        "bottom_token_ids": bottom_ids[bottom_values < 0].tolist(),
        "bottom_logits": bottom_values[bottom_values < 0].tolist(),
    }
