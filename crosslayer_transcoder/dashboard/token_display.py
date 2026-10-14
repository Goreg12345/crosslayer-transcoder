"""Render subject-model text so an LLM explainer/scorer never parses it as control tokens.

Hugging Face tokenizers match every *added* token string anywhere in raw text,
including user content, regardless of its ``special`` flag. A literal
``</think>`` in an evidence window therefore reaches a Qwen scorer as its own
reasoning delimiter. Each added token is shown as plain text instead, extending
the existing ``<|im_start|>`` -> ``⟪im_start⟫`` convention to all of them.
"""

import json
from pathlib import Path

# Added tokens of the Qwen3.8 scorer tokenizer (tokenizer.json, revision
# 482ca0f3832238542f8f5295dde86b5f22711d80), longest first. The subject Qwen3
# vocabulary shares these strings.
QWEN_ADDED_TOKENS = tuple(sorted((
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>",
    "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>",
    "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
    "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
    "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>",
    "<think>", "</think>", "<|audio_start|>", "<|audio_end|>", "<tts_pad>",
    "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>", "<|audio_pad|>",
), key=len, reverse=True))


def added_token_strings(tokenizer_json):
    """All added-token strings of a tokenizer.json, longest first."""
    tokens = json.loads(Path(tokenizer_json).read_text())["added_tokens"]
    return sorted({t["content"] for t in tokens}, key=len, reverse=True)


def display_form(token):
    """``<|name|>`` / ``<name>`` / ``</name>`` -> ``⟪name⟫`` / ``⟪/name⟫``."""
    if token.startswith("<|") and token.endswith("|>"):
        inner = token[2:-2]
    elif token.startswith("<") and token.endswith(">"):
        inner = token[1:-1]
    else:
        inner = token
    return f"⟪{inner}⟫"


def escape_added_tokens(text, added=QWEN_ADDED_TOKENS):
    for token in added:
        if token in text:
            text = text.replace(token, display_form(token))
    return text


def escape_vocab(vocab, added=QWEN_ADDED_TOKENS):
    return {k: escape_added_tokens(v, added) for k, v in vocab.items()}


def leftover_added_tokens(text, added=QWEN_ADDED_TOKENS):
    return [token for token in added if token in text]
