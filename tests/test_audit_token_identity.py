"""Audit labels identify the exact consumed token without changing inference."""

import threading

import pytest
import torch

from tests.test_dashboard_steering import config, engine


def test_trace_labels_prompt_and_generated_positions(engine):
    settings = config(
        assignments={1: 0.0},
        generated={"start": 0, "end": 1, "value": 0.0},
        max_new_tokens=4,
    )
    baseline, baseline_logits = engine.generate(
        settings, threading.Event(), lambda *args: None, "baseline"
    )
    steered, steered_logits = engine.generate(
        settings, threading.Event(), lambda *args: None, "steered"
    )
    torch.testing.assert_close(baseline_logits, steered_logits, atol=0, rtol=0)
    assert baseline["token_ids"] == steered["token_ids"]
    assert baseline["trace"] == []
    assert [row["position"] for row in steered["trace"]] == [1, 3, 4]
    consumed = settings["ids"] + steered["token_ids"]
    for row in steered["trace"]:
        token = consumed[row["position"]]
        assert row["token_id"] == token
        assert row["token_text"] == engine.tokenizer.decode([token])


@pytest.mark.parametrize("max_new_tokens", [1, 2, 4])
def test_audit_does_not_label_unconsumed_final_token(engine, max_new_tokens):
    settings = config(
        assignments={},
        generated={"start": 0, "end": max_new_tokens - 1, "value": 0.0},
        max_new_tokens=max_new_tokens,
    )
    result, _ = engine.generate(
        settings, threading.Event(), lambda *args: None, "steered"
    )
    assert len(result["trace"]) == max_new_tokens - 1
    for index, row in enumerate(result["trace"]):
        assert row["source"] == "generated"
        assert row["token_id"] == result["token_ids"][index]
    layer = engine.model.model.layers[0]
    assert not layer.mlp._forward_hooks
    assert not layer.post_attention_layernorm._forward_pre_hooks
