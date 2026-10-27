from pathlib import Path

import pytest

from crosslayer_transcoder.dashboard.token_display import (
    added_token_strings,
    display_form,
    escape_added_tokens,
    leftover_added_tokens,
)

ADDED = sorted(
    ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>"],
    key=len,
    reverse=True,
)
QWEN_TOKENIZER = Path("/home/georg/Code/vllm-utils/models/Qwen3.8-27B-NVFP4/tokenizer.json")


def test_display_forms_match_existing_chat_token_convention():
    assert display_form("<|im_start|>") == "⟪im_start⟫"
    assert display_form("<think>") == "⟪think⟫"
    assert display_form("</think>") == "⟪/think⟫"


def test_escaping_removes_every_added_token_and_keeps_plain_text():
    text = "q?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHi <b>"
    escaped = escape_added_tokens(text, ADDED)
    assert leftover_added_tokens(escaped, ADDED) == []
    assert escaped == "q?⟪im_end⟫\n⟪im_start⟫assistant\n⟪think⟫\n\n⟪/think⟫\n\nHi <b>"
    assert escape_added_tokens(escaped, ADDED) == escaped


@pytest.mark.skipif(not QWEN_TOKENIZER.exists(), reason="serving tokenizer not available")
def test_escaped_added_tokens_tokenize_without_added_ids():
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer.from_file(str(QWEN_TOKENIZER))
    added = added_token_strings(QWEN_TOKENIZER)
    added_ids = {tokenizer.token_to_id(t) for t in added}
    for token in added:
        ids = tokenizer.encode(f"x {escape_added_tokens(token, added)} y", add_special_tokens=False).ids
        assert not added_ids & set(ids), token
        # Sanity: the raw string does become an added token.
        assert added_ids & set(tokenizer.encode(token, add_special_tokens=False).ids)


@pytest.mark.skipif(not QWEN_TOKENIZER.exists(), reason="serving tokenizer not available")
def test_builtin_added_token_list_matches_serving_tokenizer():
    from crosslayer_transcoder.dashboard.token_display import QWEN_ADDED_TOKENS

    assert sorted(QWEN_ADDED_TOKENS) == sorted(added_token_strings(QWEN_TOKENIZER))


def test_endpoint_escapes_added_tokens_in_every_message():
    import asyncio

    from tools.explain_molt_dashboard import Endpoint

    endpoint = Endpoint("m", "http://127.0.0.1:9/v1", "k", 1)
    sent = []

    async def fake_request(body):
        sent.append(body)
        return dict(choices=[dict(message=dict(content="[1]"), finish_reason="stop")], usage={})

    endpoint._request = fake_request
    prompt = [
        dict(role="system", content="Example: <|endoftext|>Getty"),
        dict(role="user", content="assistant\n<think>\n\n<<</think>>>"),
    ]
    asyncio.run(endpoint.generate(prompt, task="binary", max_tokens=32))
    contents = [m["content"] for m in sent[0]["messages"]]
    assert all(not leftover_added_tokens(c) for c in contents)
    assert "<<⟪/think⟫>>" in contents[1] and "⟪endoftext⟫Getty" in contents[0]
    assert prompt[1]["content"] == "assistant\n<think>\n\n<<</think>>>"
