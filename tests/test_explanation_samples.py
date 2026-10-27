import numpy as np
import pytest

from crosslayer_transcoder.dashboard.explanation_samples import (
    detection_metrics,
    feature_examples,
    feature_order,
    split_conversations,
)


def test_queue_visits_every_observed_feature_once_with_uneven_rank_counts():
    stats = [dict(id=i, rank=512 if i < 6 else 32, count=20 - i) for i in range(15)]
    stats.append(dict(id=99, rank=128, count=0))
    order = feature_order(stats)
    assert len(order) == 15
    assert set(order) == set(range(15))
    assert order[:4] == [0, 6, 1, 7]
    assert feature_order([]) == []


def test_examples_keep_conversations_disjoint_and_do_not_show_future_tokens():
    tokens = np.arange(12 * 80)
    positions = np.array([c * 80 + 30 for c in range(12)])
    cache = dict(
        tokens=tokens,
        offsets=np.arange(13) * 80,
        positions=positions,
        features=np.zeros(12, dtype=int),
        values=np.arange(1, 13, dtype=float),
    )
    train, test = split_conversations(12)
    data = feature_examples(
        cache, 0, {str(t): str(t) for t in tokens}, train, test, n_test=1, context=16
    )
    assert {x["conversation"] for x in data["train"]} <= train
    assert {x["conversation"] for x in data["test"] + data["negatives"]} <= test
    assert data["negatives"]
    for ex in data["train"] + data["test"]:
        assert ex["stop"] == ex["conversation"] * 80 + 31
        assert ex["activations"][-1] > 0
        assert len(ex["tokens"]) == len(ex["activations"]) == 16
    assert all(not any(ex["activations"]) for ex in data["negatives"])


def test_unobserved_feature_has_no_positive_evidence():
    cache = dict(
        tokens=np.arange(80),
        offsets=np.array([0, 40, 80]),
        positions=np.array([], dtype=int),
        features=np.array([], dtype=int),
        values=np.array([], dtype=float),
    )
    data = feature_examples(cache, 0, {str(i): str(i) for i in range(80)}, {0}, {1})
    assert data["train"] == data["test"] == []
    assert data["active_conversations"] == 0


def test_overlapping_splits_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        feature_examples({}, 0, {}, {0}, {0})


def test_parse_failures_are_not_counted_as_correct_negatives():
    metrics = detection_metrics(
        [
            dict(activating=True, prediction=True),
            dict(activating=True, prediction=False),
            dict(activating=False, prediction=False),
            dict(activating=False, prediction=None),
        ]
    )
    assert metrics["parse_failures"] == 1
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == 0.75


def test_indexed_large_cache_selects_before_decoding_and_preserves_examples():
    from crosslayer_transcoder.dashboard.explanation_samples import index_features

    conversations = 1000
    tokens = np.arange(conversations * 80)
    positions = np.repeat(np.arange(conversations) * 80 + 30, 2)
    cache = dict(
        tokens=tokens,
        offsets=np.arange(conversations + 1) * 80,
        positions=positions,
        features=np.tile([0, 1], conversations),
        values=np.repeat(np.arange(1, conversations + 1, dtype=float), 2),
    )

    class CountingVocabulary(dict):
        calls = 0

        def __getitem__(self, key):
            self.calls += 1
            return key

    vocab = CountingVocabulary()
    train, test = split_conversations(conversations)
    expected = feature_examples(cache, 1, vocab, train, test, context=16)
    assert vocab.calls == (12 + 5 + 5) * 16
    index_features(cache, 3)
    actual = feature_examples(cache, 1, vocab, train, test, context=16)
    assert actual == expected
    assert len(actual["train"]) == 12
    assert len(actual["test"]) == len(actual["negatives"]) == 5
    assert actual["active_conversations"] == conversations
    absent = feature_examples(cache, 2, vocab, train, test, context=16)
    assert absent["train"] == absent["test"] == []
