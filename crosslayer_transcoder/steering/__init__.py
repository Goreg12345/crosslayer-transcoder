"""Steering with MoLT transforms — causal validation of transform interpretations.

Given a MoLT checkpoint, a transform index, and a steering strength alpha,
inject `alpha` units of that transform's contribution into the base model's
residual stream during generation, and read off the resulting text. Useful for
checking whether a transform that *correlates* with a concept (per the feature
dashboard) actually *causes* the model to talk about it.

Mechanism: a MoLT transform `i` writes `gate_i * U_i^T @ V_i^T @ acts_std` into
the standardized MLP-out at training time, where the gate is computed by a
separate detector `e`. Steering bypasses `e` and forces the gate to alpha, so
the injected contribution is `alpha * U_i^T @ V_i^T @ acts_std`, unstandardized
back into real activation space. The intervention is recomputed per token from
the actual residual — it isn't a static direction.
"""

from crosslayer_transcoder.steering.intervention import TransformIntervention
from crosslayer_transcoder.steering.hooks import steered
from crosslayer_transcoder.steering.generate import (
    generate_with_steering,
    load_base_model,
)

__all__ = [
    "TransformIntervention",
    "steered",
    "generate_with_steering",
    "load_base_model",
]
