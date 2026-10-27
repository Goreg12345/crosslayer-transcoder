import json

import pytest

from crosslayer_transcoder.dashboard.pilot import (
    parse_simulation,
    simulation_metrics,
    unique_examples,
    wilson_interval,
)


def test_simulation_rejects_misaligned_tokens_and_non_numeric_values():
    tokens = ["a", " b"]
    correct = {
        "activations": [
            {"token": "a", "activation": 2},
            {"token": " b", "activation": 0},
        ]
    }
    assert parse_simulation(json.dumps(correct), tokens) == [2, 0]
    for wrong in [
        {"activations": [{"token": "a", "activation": 2}]},
        {
            "activations": [
                {"token": "a", "activation": 2},
                {"token": "b", "activation": 0},
            ]
        },
        {
            "activations": [
                {"token": "a", "activation": True},
                {"token": " b", "activation": 0},
            ]
        },
        {
            "activations": [
                {"token": "a", "activation": 11},
                {"token": " b", "activation": 0},
            ]
        },
    ]:
        with pytest.raises(ValueError):
            parse_simulation(json.dumps(wrong), tokens)


def test_parse_failures_are_excluded_and_zero_predictor_is_not_good_simulation():
    rows = [
        {"actual": [0, 1, 0, 2], "prediction": [0, 0, 0, 0]},
        {"actual": [0, 1], "prediction": None},
    ]
    score = simulation_metrics(rows)
    assert score["parse_failures"] == 1
    assert score["valid"] == 1
    assert score["pearson"] is None
    assert score["mse_skill_vs_zero"] == 0
    assert score["r_squared"] < 0


def test_confidence_interval_and_exact_duplicate_exclusion():
    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high
    assert high - low < 0.2
    assert wilson_interval(0, 0) is None
    example = {"tokens": list(range(20)), "activations": [0] * 19 + [1]}
    other = {"tokens": list(range(20, 40)), "activations": [0] * 19 + [1]}
    assert unique_examples([example, dict(example), other], positive=True) == [
        example,
        other,
    ]
    assert unique_examples(
        [example, other], {tuple(example["tokens"])}, positive=True
    ) == [other]


def test_fuzz_distractors_are_true_zero_tokens_and_keep_original_text():
    pytest.importorskip("delphi")
    import torch
    from delphi.latents import (
        ActivatingExample,
        NonActivatingExample,
        Latent,
        LatentRecord,
    )
    from crosslayer_transcoder.dashboard.delphi_pilot import AuditedFuzzingScorer

    ex = ActivatingExample(
        tokens=torch.arange(16),
        str_tokens=[f"t{i}" for i in range(16)],
        activations=torch.tensor([0.0, 0.1, 1.0, 0.0] * 4),
    )
    zero = NonActivatingExample(
        tokens=torch.arange(16),
        str_tokens=[f"n{i}" for i in range(16)],
        activations=torch.zeros(16),
    )
    record = LatentRecord(latent=Latent("test", 1), test=[ex], not_active=[zero])
    for mode in ["default", "active"]:
        scorer = AuditedFuzzingScorer(
            None, n_examples_shown=1, fuzz_type=mode, sample_seed=7
        )
        samples = scorer._prepare(record)
        assert len(samples) == 2
        bad = samples[1]
        expected = zero if mode == "default" else ex
        assert bad.text.replace("<<", "").replace(">>", "") == "".join(
            expected.str_tokens
        )
        assert all(expected.activations[i] == 0 for i in bad.data.highlighted_positions)
        assert len(bad.data.highlighted_positions) == 4
        assert (
            scorer._prepare(record)[1].data.highlighted_positions
            == bad.data.highlighted_positions
        )


def test_fenced_json_is_accepted_but_misaligned_audit_responses_are_not_reused(
    tmp_path,
):
    from crosslayer_transcoder.dashboard.pilot import load_response_cache

    tokens = [" a"]
    good = json.dumps({"activations": [{"token": " a", "activation": 3}]})
    assert parse_simulation("```json\n" + good + "\n```", tokens) == [3]
    prompt = [
        {
            "role": "user",
            "content": json.dumps(
                {"activations": [{"token": " a", "activation": None}]}
            )
            + "\nReturn JSON.",
        }
    ]
    bad = json.dumps({"activations": [{"token": "a", "activation": 3}]})
    path = tmp_path / "audit.jsonl"
    records = [
        dict(
            task="simulation",
            messages=prompt,
            max_tokens=3072,
            response=response,
            finish_reason="stop",
        )
        for response in [bad, "```json\n" + good + "\n```"]
    ]
    path.write_text("\n".join(json.dumps(row) for row in records))
    cached = load_response_cache(path)
    assert len(cached) == 1
    assert next(iter(cached.values())) == "```json\n" + good + "\n```"


def test_simulator_preserves_thinking_tokens_as_literal_evidence():
    pytest.importorskip("delphi")
    import asyncio
    from delphi.clients.client import Response
    from crosslayer_transcoder.dashboard.delphi_pilot import StrictNeuronSimulator

    class Client:
        async def generate(self, prompt, **kwargs):
            data = json.loads(prompt[-1]["content"])
            assert [r["token"] for r in data["activations"]] == [
                "⟪think⟫",
                "text",
                "⟪/think⟫",
            ]
            for entry in data["activations"]:
                entry["activation"] = 1
            return Response(text=json.dumps(data))

    simulation = asyncio.run(
        StrictNeuronSimulator(Client(), "test", use_logprobs=False).simulate(
            ["<think>", "text", "</think>"]
        )
    )
    assert simulation.expected_activations == [1, 1, 1]
