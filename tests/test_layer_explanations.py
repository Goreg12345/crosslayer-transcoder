import copy

from crosslayer_transcoder.dashboard.layer_explanations import (
    METHODS,
    compact_result,
    evidence_counts,
    layer_samples,
    result_status,
)


def example(conv, length=20, active=True, offset=None):
    offset = conv * 100 if offset is None else offset
    return dict(
        conversation=conv,
        tokens=list(range(offset, offset + length)),
        str_tokens=[f"t{i}" for i in range(length)],
        activations=[0.0] * (length - 1) + [float(active)],
    )


def test_rare_evidence_is_preserved_and_reassignment_never_leaks_to_scoring():
    only = example(7, length=1)
    raw = dict(train=[], test=[only], negatives=[example(8, active=False)])
    samples = layer_samples(raw)
    assert samples["train"] == [only]
    assert samples["test"] == []
    assert samples["sampling_exceptions"]
    assert evidence_counts(samples)["full_budget"] is False
    assert evidence_counts(samples)["short_windows"] == 1
    assert raw["train"] == []  # Does not mutate the stored source data.


def test_full_sample_keeps_dedup_and_conversation_split_with_preferred_windows():
    raw = dict(
        train=[example(i) for i in range(64)],
        test=[example(100 + i) for i in range(100)],
        negatives=[example(300 + i, active=False) for i in range(100)],
    )
    # A held-out duplicate of explanation evidence must be excluded.
    raw["test"][0] = example(100, offset=0)
    # Preserve enough good windows without adding short fallback windows.
    raw["train"][-1] = example(63, length=1)
    samples = layer_samples(raw)
    assert evidence_counts(samples)["full_budget"]
    assert len(samples["train"]) == 32
    all_windows = [e for key in ("train", "test", "negatives") for e in samples[key]]
    assert len({tuple(e["tokens"]) for e in all_windows}) == 132
    assert not (
        {e["conversation"] for e in samples["train"]}
        & {e["conversation"] for e in samples["test"] + samples["negatives"]}
    )


def test_compact_progress_omits_cases_and_simulation_and_keeps_incomplete_visible():
    full = dict(
        id=4,
        status="scoring",
        title="Test",
        explanation="Test explanation",
        evidence_counts=dict(full_budget=True),
        methods={
            m: dict(
                status="complete",
                metrics=dict(accuracy=0.5),
                rows=[dict(tokens=list(range(64)))],
            )
            for m in METHODS
        },
    )
    full["methods"]["simulation"] = dict(status="complete", rows=[1])
    compact = compact_result(full)
    assert set(compact["methods"]) == set(METHODS)
    assert all("rows" not in m for m in compact["methods"].values())
    assert "rows" in full["methods"]["detection"]
    assert result_status(full) == "complete"
    full["methods"]["intruder"] = dict(status="unavailable", reason="too few cases")
    assert result_status(full) == "limited_evidence"
    full["methods"]["fuzz"] = dict(status="incomplete")
    assert result_status(full) == "incomplete"


def test_rare_scoring_uses_balanced_cases_and_intruder_stays_five_way():
    from tools.run_molt_pilot import make_record
    from tools.run_molt_layer_explanations import scoring_record

    samples = dict(
        train=[example(0)],
        test=[example(i) for i in range(1, 14)],
        negatives=[example(100 + i, active=False) for i in range(5)],
    )
    record = make_record(22, 5, samples)
    record.explanation = "Test"
    trial, reason = scoring_record(record, "detection")
    assert reason is None and len(trial.test) == len(trial.not_active) == 5
    trial, reason = scoring_record(record, "intruder")
    assert reason is None
    from collections import Counter

    assert min(Counter(e.quantile for e in trial.test).values()) >= 4
    scarce = copy.deepcopy(record)
    scarce.test = scarce.test[:3]
    assert scoring_record(scarce, "intruder")[0] is None
    for e in scarce.test:
        e.activations[:] = 1
    assert scoring_record(scarce, "active_fuzz")[0] is None
