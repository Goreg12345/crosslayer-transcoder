"""Shared MoLT splicing + chat-template helpers.

These primitives are used in two places:

  * ``scripts/verify_molt.py`` — CLI sanity-checks of a base LM vs. its
    MoLT-spliced counterpart.
  * ``crosslayer_transcoder.utils.callbacks.MoltEvalPromptCallback`` — logging
    eval-prompt completions to wandb at training checkpoints.

What "spliced" means
--------------------
For Gemma 3, MoLT maps ``pre_feedforward_layernorm.input`` ->
``post_feedforward_layernorm.output``. For GPT-2 it maps ``ln_2.input`` ->
``mlp.output``. To splice we register, per block, a forward-pre-hook on the
capture module (reads MoLT's input residual) and a forward-hook on the splice
module (overwrites its output with ``MoLT(captured_residual)``). The splice
points mirror the activation-source detection in
``crosslayer_transcoder/data/activation_sources.py`` so they line up with what
MoLT was trained on.
"""

from __future__ import annotations

from contextlib import contextmanager

import einops
import torch

_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# Architecture-specific block resolution.
# ---------------------------------------------------------------------------


def resolve_blocks(model) -> tuple[str, list]:
    """Return ``("gpt2"|"gemma3", per-block module list)``."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return "gpt2", list(model.transformer.h)
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        layers = model.model.language_model.layers
    if layers is None:
        raise ValueError(
            f"Don't know how to find transformer blocks on {type(model).__name__}"
        )
    sample = layers[0]
    if not hasattr(sample, "pre_feedforward_layernorm"):
        raise ValueError(
            f"Block type {type(sample).__name__} has no pre_feedforward_layernorm; "
            "this helper only handles GPT-2 and Gemma 3 layouts."
        )
    return "gemma3", list(layers)


def splice_targets(arch: str, block):
    """Return ``(capture_module, splice_module)`` for one transformer block.

    capture_module: forward-pre-hook reads MoLT's input residual from its arg.
    splice_module:  forward-hook overwrites the module's output with MoLT's recon.
    """
    if arch == "gpt2":
        return block.ln_2, block.mlp
    return block.pre_feedforward_layernorm, block.post_feedforward_layernorm


# ---------------------------------------------------------------------------
# MoLT splice forward (matches Molt.forward, but exposes per-batch shapes).
# ---------------------------------------------------------------------------


def molt_layer_forward(inner_molt, resid: torch.Tensor, layer: int) -> torch.Tensor:
    """Run one per-layer ``Molt`` and return its standardized reconstruction.

    ``resid`` may be ``(B, T, D)`` at prefill or ``(B, 1, D)`` during
    generation; both flow through input/output standardizers without special
    casing.
    """
    acts = inner_molt.input_standardizer(resid, layer)
    pre = inner_molt.e(acts)
    gate = inner_molt.nonlinearity(pre)  # (..., n_features)
    raw = []
    for U, V in zip(inner_molt.Us, inner_molt.Vs):
        latents = einops.einsum(acts, V, "... d, n d r -> ... n r")
        raw.append(einops.einsum(latents, U, "... n r, n r d -> ... n d"))
    raw = torch.cat(raw, dim=-2)  # (..., n_features, d_acts)
    weighted = gate.unsqueeze(-1) * raw
    recons = weighted.sum(dim=-2)
    return inner_molt.output_standardizer(recons, layer)


@contextmanager
def splice_hooks(model, molt, arch: str, layers: list[int]):
    """Splice MoLT into ``model`` at the listed block indices for the duration."""
    blocks = resolve_blocks(model)[1]
    captured: dict[int, torch.Tensor] = {}
    handles = []
    molt_dtype = next(molt.parameters()).dtype

    for L in layers:
        capture_mod, splice_mod = splice_targets(arch, blocks[L])

        def make_pre(L=L):
            def pre(_m, args):
                captured[L] = args[0]
            return pre

        def make_post(L=L):
            def post(_m, _args, out):
                resid = captured[L]
                resid_in = resid.to(molt_dtype)
                rec = molt_layer_forward(molt.molts[L], resid_in, L)
                # Match the dtype/device the model expects on the residual stream.
                rec = rec.to(dtype=out.dtype, device=out.device)
                return rec
            return post

        handles.append(capture_mod.register_forward_pre_hook(make_pre()))
        handles.append(splice_mod.register_forward_hook(make_post()))

    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Prompt prep + comparison primitives.
# ---------------------------------------------------------------------------


def apply_chat_template(tokenizer, user_text: str, system: str | None = None) -> str:
    """Wrap ``user_text`` in the chat template, stripping a duplicate BOS.

    The leading BOS is removed because downstream tokenization re-adds special
    tokens; keeping it would double the BOS.
    """
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_text})
    s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    bos = getattr(tokenizer, "bos_token", None)
    if bos and s.startswith(bos):
        s = s[len(bos):]
    return s


def topk_str(probs: torch.Tensor, tokenizer, k: int) -> list[tuple[str, float]]:
    vals, idxs = torch.topk(probs, k)
    return [(tokenizer.decode([int(i)]), float(p)) for p, i in zip(vals, idxs)]


def kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    return float((p * (p.add(eps).log() - q.add(eps).log())).sum().item())


@torch.no_grad()
def next_token_probs(model, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids)
    return out.logits[0, -1].float().softmax(-1).cpu()


@torch.no_grad()
def greedy_complete(model, tokenizer, input_ids: torch.Tensor, n: int) -> str:
    """Append ``n`` greedy tokens. Returns only the newly generated text."""
    if n <= 0:
        return ""
    eos = tokenizer.eos_token_id
    cur = input_ids
    new_ids: list[int] = []
    for _ in range(n):
        logits = model(input_ids=cur).logits[0, -1]
        nxt = int(logits.argmax().item())
        new_ids.append(nxt)
        if eos is not None and nxt == eos:
            break
        cur = torch.cat([cur, torch.tensor([[nxt]], device=cur.device)], dim=1)
    return tokenizer.decode(new_ids)


@torch.no_grad()
def compare_prompt(
    model,
    tokenizer,
    prompt_text: str,
    arch: str,
    molt,
    *,
    mode: str = "both",
    splice_layers: list[int] | None = None,
    topk: int = 10,
    max_new_tokens: int = 8,
) -> dict:
    """Run one prompt under selected mode(s) and return a comparison dict.

    ``mode`` is one of ``{"vanilla", "molt", "both"}``. When MoLT is requested
    but ``molt`` is None, that side is skipped.
    """
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    input_ids = enc.input_ids.to(next(model.parameters()).device)

    out: dict = {
        "prompt": prompt_text,
        "n_tokens": int(input_ids.shape[1]),
        "tokens": [tokenizer.decode([int(t)]) for t in input_ids[0]],
    }

    do_vanilla = mode in ("vanilla", "both")
    do_molt = mode in ("molt", "both") and molt is not None

    probs_v = probs_m = None
    if do_vanilla:
        probs_v = next_token_probs(model, input_ids)
        out["vanilla"] = {
            "top": topk_str(probs_v, tokenizer, topk),
            "completion": greedy_complete(model, tokenizer, input_ids, max_new_tokens),
        }

    if do_molt:
        layers = splice_layers if splice_layers else list(range(molt.n_layers))
        with splice_hooks(model, molt, arch, layers):
            probs_m = next_token_probs(model, input_ids)
            comp_m = greedy_complete(model, tokenizer, input_ids, max_new_tokens)
        out["molt"] = {
            "top": topk_str(probs_m, tokenizer, topk),
            "completion": comp_m,
            "splice_layers": layers,
        }

    if probs_v is not None and probs_m is not None:
        out["kl_vanilla_molt"] = kl(probs_v, probs_m)
        out["agree_top1"] = out["vanilla"]["top"][0][0] == out["molt"]["top"][0][0]

    return out
