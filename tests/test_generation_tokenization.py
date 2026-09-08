"""The extractor must see the tokenizer's exact tokens, including on restart."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from crosslayer_transcoder.data import generation_loop
from crosslayer_transcoder.data.generation_loop import DataGenerationLoop
from crosslayer_transcoder.data.text_dataset import TextDataset


def make_loop(monkeypatch, tokenizer, dataset, field, length):
    # Keep the real TextDataset path, but avoid worker processes and GPU models.
    monkeypatch.setattr(generation_loop, "DataLoader", lambda dataset, **kwargs: dataset)
    model = SimpleNamespace(tokenizer=tokenizer)  # No config/BOS lookup is needed.
    loop = DataGenerationLoop.__new__(DataGenerationLoop)
    loop.dataset = dataset
    loop.dataset_text_field = field
    loop.generation_batch_size = 2
    loop.max_sequence_length = length
    loop.shared_buffer = SimpleNamespace(get_stats=lambda: {})
    loop.deployment_policy = SimpleNamespace(
        get_current_model=lambda: model, select_device=lambda stats: "cpu"
    )
    loop.monitor = Mock()
    loop.activation_computer = SimpleNamespace(
        get_next_batch=lambda model, tokens, mask: (tokens.clone(), mask.clone())
    )
    loop._setup_text_dataset_loader()
    return loop


@pytest.mark.parametrize("template_tokens", [[151644, 872, 198, 42, 151645], [2, 105, 2364, 107, 42]])
@pytest.mark.parametrize("length", [3, 5, 8])
def test_template_tokens_reach_extractor_unchanged(monkeypatch, template_tokens, length):
    messages = [{"role": "user", "content": "hello"}]
    tokenizer = SimpleNamespace(apply_chat_template=Mock(return_value=template_tokens))
    loop = make_loop(monkeypatch, tokenizer, [{"messages": messages}], "messages", length)
    expected = template_tokens[:length]

    for _ in range(2):  # The second pass exercises dataset exhaustion/recreation.
        tokens, mask = loop._generate_activations()
        assert tokens.shape == mask.shape == (2, length)
        assert tokens[0, mask[0]].tolist() == expected
        assert mask[0].tolist() == [True] * len(expected) + [False] * (length - len(expected))
        assert not mask[1].any()
        assert not tokens[~mask].any()

    tokenizer.apply_chat_template.assert_called_with(
        messages, tokenize=True, add_generation_prompt=False
    )
    loop.monitor.log_dataset_exhausted.assert_called_once()


def test_plain_text_also_preserves_tokenizer_special_tokens(monkeypatch):
    tokenizer = Mock(return_value={"input_ids": [2, 12, 13]})
    loop = make_loop(monkeypatch, tokenizer, [{"text": "hello"}], "text", 5)
    tokens, mask = loop._generate_activations()
    assert tokens[0, mask[0]].tolist() == [2, 12, 13]
    tokenizer.assert_called_once_with("hello")


def test_conversation_requires_callable_chat_formatter():
    dataset = TextDataset(
        [{"messages": [{"role": "user", "content": "hello"}]}],
        SimpleNamespace(apply_chat_template=None),
        batch_size=1,
        hf_text_accessor="messages",
    )
    with pytest.raises(TypeError, match="apply_chat_template"):
        next(dataset)


@pytest.mark.parametrize("model", ["Qwen/Qwen3-4B", "google/gemma-3-4b-it"])
def test_cached_model_template_reaches_extractor_unchanged(monkeypatch, model):
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    except OSError:
        pytest.skip(f"Tokenizer is not cached locally: {model}")
    messages = [{"role": "user", "content": "Hello"}]
    expected = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    loop = make_loop(
        monkeypatch, tokenizer, [{"messages": messages}], "messages", len(expected) + 3
    )
    tokens, mask = loop._generate_activations()
    assert tokens[0, mask[0]].tolist() == expected
    if model.startswith("Qwen/"):
        assert tokenizer.decode([tokens[0, 0]]) == "<|im_start|>"
    else:
        assert tokens[0, 0].item() == tokenizer.bos_token_id
        assert tokens[0, 1].item() != tokenizer.bos_token_id
