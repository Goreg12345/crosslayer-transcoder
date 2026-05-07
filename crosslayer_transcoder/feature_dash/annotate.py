"""LLM-based feature annotation via OpenRouter.

Given a (layer, feature_id) -> list of windowed top-activating examples, ask
a Claude (or any) model — through OpenRouter's OpenAI-compatible API — for a
one-line description of what activates the feature.

Design notes:

* OpenRouter's chat completions endpoint mirrors OpenAI's, but for Anthropic
  models it passes through `cache_control` markers — so the constant system
  prompt (the bulk of the input tokens) is cached after the first call.
* We dispatch annotation requests concurrently with a thread pool. Each call
  is independent, short, and idempotent so a thread pool over `requests` is
  enough.
* Failures (rate limits, timeouts, transient errors) are caught per-feature
  so a single bad feature doesn't break the whole pass — those features just
  end up with no description.
* The function consumes pre-windowed examples (the same `examples` dict that
  ends up in the dashboard bundle) so the same data is shown to the LLM as
  is shown to the user.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, Optional

import requests

logger = logging.getLogger(__name__)


# A short description is the goal — explicit limit keeps cost down and the
# dashboard table tidy.
_SYSTEM_PROMPT = """\
You are an interpretability assistant analysing features from a sparse
transcoder (MoLT) trained on the residual stream of a transformer language
model. For each feature you will be shown several text excerpts that strongly
activate it, with the peak token marked in **bold**.

Write a one-line description (under 20 words) of the linguistic, semantic, or
syntactic pattern that activates this feature. Be specific — describe the
shared *pattern across examples*, not any one example. If the examples don't
share an obvious pattern, say so briefly.

Examples of good descriptions:
  - "Words/punctuation immediately after the start of a quoted dialogue."
  - "Numerals in the context of years (e.g. '1995', '2023')."
  - "The token 'the' when it begins a noun phrase referring to a US president."
  - "No clear unifying pattern — fires sparsely on punctuation."

Output ONLY the description sentence. No preamble, no markdown headers, no
trailing notes."""


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class FeatureAnnotation:
    layer: int
    feature_id: int
    description: str  # empty string if annotation failed


def _format_examples(examples: list[dict], max_examples: int = 5) -> str:
    """Render a feature's top examples as a plain-text prompt.

    Each example becomes a numbered block with the windowed tokens joined,
    and the peak token wrapped in `**...**` so the LLM can see what fired.
    """
    parts = []
    for i, ex in enumerate(examples[:max_examples]):
        tokens = list(ex["tokens"])
        peak = ex.get("peak_token_pos", 0)
        if 0 <= peak < len(tokens):
            tokens[peak] = f"**{tokens[peak]}**"
        text = "".join(tokens).replace("\n", " ").strip()
        if len(text) > 800:
            text = text[:800] + "…"
        parts.append(f"Example {i + 1} (peak activation {ex['peak_activation']:.2f}):\n{text}")
    return "\n\n".join(parts)


def _build_user_message(layer: int, feature_id: int, examples: list[dict]) -> str:
    return (
        f"Feature L{layer}#{feature_id} — top {min(5, len(examples))} activating excerpts:\n\n"
        + _format_examples(examples)
    )


def _post_openrouter(
    session: requests.Session,
    api_key: str,
    model: str,
    user_msg: str,
    max_tokens: int,
    timeout: float,
    use_cache: bool,
) -> str:
    """Single OpenRouter chat-completions call. Returns description text."""
    if use_cache:
        # Anthropic-style structured system message — OpenRouter passes the
        # `cache_control` flag through to the upstream provider for caching.
        system_content = [
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system_content = _SYSTEM_PROMPT

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_msg},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # OpenRouter recommends a referrer + title for routing, but they're optional.
        "HTTP-Referer": "https://github.com/jiito/crosslayer-transcoder",
        "X-Title": "MoLT feature dashboard",
    }
    resp = session.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if "choices" not in data:
        raise RuntimeError(f"unexpected response: {data}")
    text = data["choices"][0]["message"]["content"]
    if isinstance(text, list):
        # Some providers return content as a list of blocks; take the text ones.
        text = "".join(b.get("text", "") for b in text if b.get("type") == "text")
    return (text or "").strip()


def _annotate_one(
    session: requests.Session,
    api_key: str,
    model: str,
    layer: int,
    feature_id: int,
    examples: list[dict],
    max_tokens: int,
    timeout: float,
    use_cache: bool,
) -> FeatureAnnotation:
    if not examples:
        return FeatureAnnotation(layer=layer, feature_id=feature_id, description="")
    user_msg = _build_user_message(layer, feature_id, examples)
    try:
        text = _post_openrouter(
            session, api_key, model, user_msg, max_tokens, timeout, use_cache
        )
        # Strip surrounding quotes if added.
        if text.startswith('"') and text.endswith('"') and len(text) > 1:
            text = text[1:-1]
        return FeatureAnnotation(layer=layer, feature_id=feature_id, description=text)
    except Exception as exc:  # noqa: BLE001 — log per-feature, keep going
        logger.warning("annotation failed for L%dF%d: %s", layer, feature_id, exc)
        return FeatureAnnotation(layer=layer, feature_id=feature_id, description="")


def annotate_features(
    features: Iterable[tuple[int, int, list[dict]]],
    model: str = "anthropic/claude-haiku-4.5",
    max_workers: int = 20,
    max_tokens: int = 80,
    timeout: float = 30.0,
    log_every: int = 50,
    api_key: Optional[str] = None,
    use_cache: bool = True,
) -> dict[tuple[int, int], str]:
    """Annotate many (layer, feature_id, examples) tuples concurrently.

    Returns a dict mapping (layer, feature_id) -> description (str). Features
    with no examples or that hit an error get an empty string.

    `model` follows OpenRouter naming (e.g. ``anthropic/claude-haiku-4.5``,
    ``anthropic/claude-sonnet-4.5``, ``meta-llama/llama-3-70b-instruct``).
    """
    api_key = (
        api_key
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPEN_ROUTER_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY (or OPEN_ROUTER_API_KEY) not set — "
            "pass api_key= or export the env var"
        )

    items = list(features)
    out: dict[tuple[int, int], str] = {}
    if not items:
        return out

    logger.info(
        "annotating %d features with %s via OpenRouter (max_workers=%d)…",
        len(items),
        model,
        max_workers,
    )

    session = requests.Session()
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                _annotate_one,
                session,
                api_key,
                model,
                layer,
                feature_id,
                examples,
                max_tokens,
                timeout,
                use_cache,
            )
            for (layer, feature_id, examples) in items
        ]
        for fut in as_completed(futures):
            ann = fut.result()
            out[(ann.layer, ann.feature_id)] = ann.description
            completed += 1
            if log_every and completed % log_every == 0:
                logger.info("annotated %d/%d features", completed, len(items))

    return out


def select_features_for_annotation(
    collectors: list,
    feature_summaries_per_layer: list[list[dict]],
    *,
    min_max_activation: float = 0.0,
    require_examples: bool = True,
    max_per_layer: Optional[int] = None,
) -> list[tuple[int, int, list[dict]]]:
    """Pick which (layer, feature_id) pairs are worth annotating.

    `feature_summaries_per_layer[layer][feature_id]` is the windowed-summary
    dict that the bundle would write for that feature. We use it directly so
    the LLM sees the same examples the human will.

    Filters:
      * `require_examples`  : drop features whose `examples` list is empty.
      * `min_max_activation`: drop features with peak activation below this.
      * `max_per_layer`     : keep only the top-K-by-max_activation per layer
                               (useful when annotation budget is tight).
    """
    out: list[tuple[int, int, list[dict]]] = []
    for layer, summaries in enumerate(feature_summaries_per_layer):
        candidates = []
        for feature_id, body in enumerate(summaries):
            if require_examples and not body.get("examples"):
                continue
            if body.get("max_activation", 0.0) < min_max_activation:
                continue
            candidates.append((feature_id, body))
        if max_per_layer is not None:
            candidates.sort(key=lambda x: x[1].get("max_activation", 0.0), reverse=True)
            candidates = candidates[:max_per_layer]
        for feature_id, body in candidates:
            out.append((layer, feature_id, body["examples"]))
    return out
