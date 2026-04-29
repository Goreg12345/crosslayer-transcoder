"""Free-generation helpers for MoLT steering."""

from __future__ import annotations

from typing import Optional

import torch

from crosslayer_transcoder.steering.hooks import steered
from crosslayer_transcoder.steering.intervention import TransformIntervention


def load_base_model(name: str, device: str | torch.device = "cpu"):
    """Load an HF causal LM + tokenizer for use with steering.

    Returns (model, tokenizer). Imports transformers lazily so the rest of the
    steering API stays usable in test environments without it.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    return model, tok


@torch.no_grad()
def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    intervention: Optional[TransformIntervention],
    *,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 0,
) -> str:
    """Generate text under an optional steering intervention.

    If `intervention` is None (or has alpha=0), no hooks are registered and
    this is a plain `model.generate()` call — useful for baselining.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    torch.manual_seed(seed)

    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    if intervention is None or intervention.alpha == 0.0:
        out = model.generate(**inputs, **gen_kwargs)
    else:
        with steered(model, intervention):
            out = model.generate(**inputs, **gen_kwargs)

    return tokenizer.decode(out[0], skip_special_tokens=True)
