import threading

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from crosslayer_transcoder.dashboard.gates import MoltGates
from crosslayer_transcoder.dashboard.steering import (
    SteeringEngine,
    requested_amount,
    target_gate,
)


@pytest.fixture
def engine():
    torch.manual_seed(3)
    e = SteeringEngine.__new__(SteeringEngine)
    e.device, e.layer = "cpu", 0
    e.model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=20,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )
    ).eval()
    e.state = {
        "model.e.weight": torch.randn(1, 8),
        "model.e.bias": torch.tensor([0.2]),
        "model.nonlinearity.theta": torch.tensor([[0.1]]),
        "model.nonlinearity.bandwidth": torch.tensor(1.0),
        "model.input_standardizer.mean": torch.zeros(1, 8),
        "model.input_standardizer.std": torch.ones(1, 8),
        "model.output_standardizer.std": torch.rand(1, 8),
        "model.Us.0": torch.randn(1, 2, 8),
        "model.Vs.0": torch.randn(1, 8, 2),
    }
    e.gates = MoltGates(e.state, 0).eval()
    e.eos_ids = set()

    class Tokenizer:
        def decode(self, ids, **kwargs):
            return ",".join(map(str, ids))

    e.tokenizer = Tokenizer()
    return e


def config(**kwargs):
    c = dict(
        ids=[1, 2, 3],
        feature=0,
        mode="add",
        assignments={2: 0.0},
        generated=None,
        max_new_tokens=3,
        temperature=0,
        top_p=0.95,
        seed=42,
    )
    c.update(kwargs)
    return c


def test_gate_modes_and_generated_position_boundaries():
    natural = torch.tensor([0.0, 2.0])
    torch.testing.assert_close(target_gate(natural, 3, "set"), torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(
        target_gate(natural, -1, "add"), torch.tensor([0.0, 1.0])
    )
    torch.testing.assert_close(
        target_gate(natural, 2, "multiply"), torch.tensor([0.0, 4.0])
    )
    generated = {"start": 1, "end": 2, "value": 4}
    assert requested_amount(2, 3, {2: 7}, generated) == 7
    assert requested_amount(3, 3, {}, generated) is None
    assert requested_amount(4, 3, {}, generated) == 4
    assert requested_amount(5, 3, {}, generated) == 4
    assert requested_amount(6, 3, {}, generated) is None


def test_zero_addition_is_exactly_baseline_with_cache(engine):
    c = config()
    baseline, b = engine.generate(c, threading.Event(), lambda *a: None, "baseline")
    steered, s = engine.generate(c, threading.Event(), lambda *a: None, "steered")
    torch.testing.assert_close(b, s, atol=0, rtol=0)
    assert baseline["token_ids"] == steered["token_ids"]
    assert steered["trace"][0]["delta_l2"] == 0
    assert not engine.model.model.layers[0].mlp._forward_hooks


def test_only_selected_mlp_position_is_changed(engine):
    layer = engine.model.model.layers[0]
    before, after = [], []
    h1 = layer.mlp.register_forward_hook(
        lambda m, a, o: before.append(o.detach().clone())
    )
    trace = []
    try:
        with (
            torch.inference_mode(),
            engine.intervention(config(mode="set", assignments={1: 3.0}), trace, [0]),
        ):
            h2 = layer.mlp.register_forward_hook(
                lambda m, a, o: after.append(o.detach().clone())
            )
            try:
                engine.model.model(input_ids=torch.tensor([[1, 2, 3]]))
            finally:
                h2.remove()
    finally:
        h1.remove()
    torch.testing.assert_close(
        before[0][:, [0, 2]], after[0][:, [0, 2]], atol=0, rtol=0
    )
    assert not torch.equal(before[0][:, 1], after[0][:, 1])
    assert len(trace) == 1 and trace[0]["position"] == 1 and trace[0]["after"] == 3.0


def test_generated_range_uses_consumed_token_indices_and_cleans_up(engine):
    c = config(assignments={}, generated={"start": 0, "end": 0, "value": 2.0})
    result, _ = engine.generate(c, threading.Event(), lambda *a: None, "steered")
    assert [t["position"] for t in result["trace"]] == [3]
    layer = engine.model.model.layers[0]
    with pytest.raises(RuntimeError), engine.intervention(c, [], [0]):
        raise RuntimeError("cancelled operation")
    assert not layer.mlp._forward_hooks
    assert not layer.post_attention_layernorm._forward_pre_hooks


def test_reference_scale_distinguishes_mean_raw_and_natural():
    from crosslayer_transcoder.dashboard.steering import reference_scale

    assert reference_scale("set", "mean_active", 1.5) == 1.5
    assert reference_scale("add", "mean_active", 1.5) * 2 == 3
    assert reference_scale("set", "raw", 1.5) == 1
    assert reference_scale("multiply", "mean_active", 1.5) == 1
    with pytest.raises(ValueError, match="No active-gate mean"):
        reference_scale("set", "mean_active", 0)
