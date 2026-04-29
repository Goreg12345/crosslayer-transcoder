"""Forward-hook plumbing to inject a TransformIntervention into a base model.

We need two coupled hooks: one to capture the input to ln_2 (which is what
MoLT was trained on), and one to add the intervention's delta to the MLP
output. They're coupled because the MLP-output hook needs the value the ln_2
hook saw — recovering it from inside the MLP forward would require unwinding
the layernorm.

Currently only GPT-2-family attribute paths (`transformer.h[L].ln_2`,
`transformer.h[L].mlp`) are supported, matching the rest of this codebase.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn

from crosslayer_transcoder.steering.intervention import TransformIntervention


def _resolve_block(model: nn.Module, layer: int) -> nn.Module:
    """Locate transformer block `layer` on a HF causal LM.

    Tries the GPT-2 path first (model.transformer.h), then a few common
    alternatives. Raises with a clear message if the layout is unrecognized.
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        # Llama / Mistral layout — left in for future support, but the rest of
        # this module assumes ln_2/mlp naming, so we still error below.
        raise NotImplementedError(
            "Llama-style layouts (model.model.layers[*]) are not supported yet — "
            "MoLT in this repo trains on GPT-2-style ln_2.input / mlp.output."
        )
    raise NotImplementedError(
        f"Could not find transformer blocks on {type(model).__name__}. "
        "Steering currently supports GPT-2-family models only."
    )


@contextmanager
def steered(
    model: nn.Module,
    intervention: TransformIntervention,
) -> Iterator[None]:
    """Patch `model` so that during forward passes within this context, the
    output of block `intervention.layer`'s MLP gets `intervention(mlp_in)`
    added to it.

    `mlp_in` is the input to ln_2 — the pre-LN residual that MoLT was trained
    on. The hooks fire in this order on each forward: pre-ln_2 stashes the
    residual, then post-mlp reads it and adds the delta to the MLP output.
    The stash is per-context and not thread-safe.
    """
    block = _resolve_block(model, intervention.layer)
    if not (hasattr(block, "ln_2") and hasattr(block, "mlp")):
        raise NotImplementedError(
            f"Block at layer {intervention.layer} lacks ln_2/mlp attributes — "
            "this hook implementation is GPT-2-specific."
        )

    stash: dict[str, torch.Tensor | None] = {"mlp_in": None}

    def pre_ln2(_module: nn.Module, args: tuple) -> None:
        # ln_2.forward(x) — args[0] is the pre-LN residual.
        stash["mlp_in"] = args[0]
        return None

    def post_mlp(_module: nn.Module, _args, output):
        mlp_in = stash["mlp_in"]
        if mlp_in is None:
            return output
        delta = intervention(mlp_in)
        # Some HF MLPs return a tuple; GPT-2 returns a tensor. Be defensive.
        if isinstance(output, tuple):
            return (output[0] + delta,) + output[1:]
        return output + delta

    h_pre = block.ln_2.register_forward_pre_hook(pre_ln2)
    h_post = block.mlp.register_forward_hook(post_mlp)
    try:
        yield
    finally:
        h_pre.remove()
        h_post.remove()
