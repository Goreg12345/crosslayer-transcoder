"""Tests for the GateCollector accumulator (no LM, no dataset).

Synthetic gate batches go in, we check that firing counts, activation rates,
top-K ordering, and per-token activation traces come out right.
"""

from __future__ import annotations

import pytest
import torch

from crosslayer_transcoder.feature_dash.collect import GateCollector


F = 3       # features
K = 4       # top-K
T = 6       # seq_len


def _zero_gates(B: int) -> torch.Tensor:
    return torch.zeros(B, T, F)


def test_firing_count_and_total_tokens():
    coll = GateCollector(n_features=F, top_k=K, seq_len=T)
    tok = torch.arange(2 * T, dtype=torch.long).reshape(2, T)
    g = _zero_gates(2)
    g[0, 1, 0] = 0.5
    g[0, 4, 0] = 1.5
    g[1, 2, 0] = 0.0  # 0 doesn't count as active (gate > 0)
    g[1, 3, 1] = 0.7

    coll.update(tok, g)

    assert coll.total_tokens == 2 * T
    assert coll.firing_count.tolist() == [2, 1, 0]
    rates = coll.activation_rate()
    assert torch.allclose(rates, torch.tensor([2 / 12, 1 / 12, 0.0]))


def test_max_activation_tracks_global_max_across_batches():
    coll = GateCollector(n_features=F, top_k=K, seq_len=T)
    g1 = _zero_gates(1)
    g1[0, 0, 0] = 1.0
    coll.update(torch.zeros(1, T, dtype=torch.long), g1)

    g2 = _zero_gates(1)
    g2[0, 0, 0] = 0.5
    g2[0, 0, 1] = 4.2
    coll.update(torch.zeros(1, T, dtype=torch.long), g2)

    # Once a batch lands, max_activation rises from -inf to >=0 even for
    # non-firing features (max of -inf and 0 is 0). That's fine for the
    # dashboard — feature_summary surfaces 0.0 for dead features.
    assert torch.allclose(
        coll.max_activation, torch.tensor([1.0, 4.2, 0.0]), atol=1e-6
    )


def test_topk_orders_by_per_sequence_peak():
    coll = GateCollector(n_features=F, top_k=K, seq_len=T)
    # 5 sequences, all activate feature 0 with different peaks at different positions.
    B = 5
    tok = torch.arange(B * T, dtype=torch.long).reshape(B, T)
    g = _zero_gates(B)
    peaks_in = [0.1, 3.0, 2.0, 0.5, 5.0]
    pos_in = [0, 2, 5, 1, 4]
    for b, (p, pos) in enumerate(zip(peaks_in, pos_in)):
        g[b, pos, 0] = p
    coll.update(tok, g)

    summary = coll.feature_summary(0)

    # Top-K = 4: should drop the 0.1 peak.
    assert summary.top_peaks == pytest.approx([5.0, 3.0, 2.0, 0.5])
    assert summary.activation_rate == pytest.approx(5 / (B * T))
    assert summary.max_activation == 5.0
    # Token ids of top sequence should match seq with peak 5.0 (b=4).
    assert summary.top_token_ids[0] == tok[4].tolist()
    # The activation trace's argmax should match where we placed the peak.
    trace0 = summary.top_activations[0]
    assert max(range(T), key=lambda i: trace0[i]) == pos_in[4]


def test_topk_merges_across_multiple_batches():
    coll = GateCollector(n_features=F, top_k=K, seq_len=T)
    # Batch 1: peaks 1, 2, 3 for feature 0.
    tok1 = torch.full((3, T), 11, dtype=torch.long)
    g1 = _zero_gates(3)
    g1[0, 0, 0] = 1.0
    g1[1, 0, 0] = 2.0
    g1[2, 0, 0] = 3.0
    coll.update(tok1, g1)

    # Batch 2: peaks 0.5, 4.0 for feature 0. Top-4 across the two batches
    # should be {4.0, 3.0, 2.0, 1.0}, dropping 0.5.
    tok2 = torch.full((2, T), 22, dtype=torch.long)
    g2 = _zero_gates(2)
    g2[0, 0, 0] = 0.5
    g2[1, 0, 0] = 4.0
    coll.update(tok2, g2)

    summary = coll.feature_summary(0)
    assert summary.top_peaks == [4.0, 3.0, 2.0, 1.0]
    # Top sequence is from batch 2 (its tokens are all 22).
    assert summary.top_token_ids[0] == [22] * T
    # The 1.0-peak entry is from batch 1 (tokens all 11).
    assert summary.top_token_ids[3] == [11] * T


def test_dead_feature_summary_is_empty():
    coll = GateCollector(n_features=F, top_k=K, seq_len=T)
    g = _zero_gates(2)
    g[0, 0, 0] = 1.0  # only feature 0 fires
    coll.update(torch.zeros(2, T, dtype=torch.long), g)

    dead = coll.feature_summary(2)
    assert dead.activation_rate == 0.0
    assert dead.max_activation == 0.0
    assert dead.top_peaks == []
    assert dead.top_token_ids == []
    assert dead.top_activations == []


def test_per_token_trace_is_feature_specific():
    """The trace stored for feature i must be gates[..., i] for that sequence,
    not any other feature."""
    coll = GateCollector(n_features=F, top_k=K, seq_len=T)
    tok = torch.zeros(1, T, dtype=torch.long)
    g = _zero_gates(1)
    g[0, :, 0] = torch.arange(T, dtype=torch.float32)        # feat 0: 0..T-1
    g[0, :, 1] = torch.arange(T, dtype=torch.float32) * 10   # feat 1: 0,10,20,...
    coll.update(tok, g)

    s0 = coll.feature_summary(0)
    s1 = coll.feature_summary(1)
    assert s0.top_activations[0] == list(range(T))
    assert s1.top_activations[0] == [i * 10 for i in range(T)]


def test_update_rejects_wrong_shape():
    coll = GateCollector(n_features=F, top_k=K, seq_len=T)
    bad_g = torch.zeros(1, T + 1, F)
    try:
        coll.update(torch.zeros(1, T + 1, dtype=torch.long), bad_g)
    except ValueError:
        return
    raise AssertionError("expected ValueError for seq_len mismatch")


def test_update_rejects_wrong_feature_count():
    coll = GateCollector(n_features=F, top_k=K, seq_len=T)
    bad_g = torch.zeros(1, T, F + 1)
    try:
        coll.update(torch.zeros(1, T, dtype=torch.long), bad_g)
    except ValueError:
        return
    raise AssertionError("expected ValueError for feature count mismatch")


# ---- Activation histogram --------------------------------------------------


def test_histogram_starts_empty_and_has_correct_shape():
    coll = GateCollector(n_features=F, top_k=K, seq_len=T)
    assert coll.act_histogram.shape == (F, GateCollector.HIST_N_BINS)
    assert coll.act_histogram.sum() == 0
    edges = coll.hist_edges()
    assert len(edges) == GateCollector.HIST_N_BINS + 1
    # Edges are log-spaced over [10^LO, 10^HI].
    assert edges[0] == pytest.approx(10 ** GateCollector.HIST_LOG_LO)
    assert edges[-1] == pytest.approx(10 ** GateCollector.HIST_LOG_HI)


def test_histogram_counts_only_firing_values():
    coll = GateCollector(n_features=F, top_k=K, seq_len=T)
    g = _zero_gates(2)
    g[0, 0, 0] = 1.0     # feature 0: one firing
    g[1, 0, 0] = 5.0     # feature 0: another firing
    g[0, 1, 1] = 0.0     # feature 1: zero is not "firing"
    coll.update(torch.zeros(2, T, dtype=torch.long), g)

    # Feature 0 has 2 firings, feature 1 has none, feature 2 has none.
    assert coll.act_histogram[0].sum().item() == 2
    assert coll.act_histogram[1].sum().item() == 0
    assert coll.act_histogram[2].sum().item() == 0


def test_histogram_drops_out_of_range_values():
    coll = GateCollector(n_features=F, top_k=K, seq_len=T)
    g = _zero_gates(1)
    # 10^4 is above HI=10^3; should NOT be counted.
    g[0, 0, 0] = 1e4
    # 10^-5 is below LO=10^-3; should NOT be counted.
    g[0, 1, 0] = 1e-5
    # In-range value, should be counted.
    g[0, 2, 0] = 1.0
    coll.update(torch.zeros(1, T, dtype=torch.long), g)

    assert coll.act_histogram[0].sum().item() == 1


def test_histogram_accumulates_across_batches():
    coll = GateCollector(n_features=F, top_k=K, seq_len=T)
    for _ in range(3):
        g = _zero_gates(1)
        g[0, 0, 0] = 0.5
        coll.update(torch.zeros(1, T, dtype=torch.long), g)
    assert coll.act_histogram[0].sum().item() == 3


def test_histogram_bins_in_correct_log_bucket():
    """A value of 1.0 with edges spanning 10^-3..10^3 should fall in the
    middle of the histogram (bin index = HIST_N_BINS / 2)."""
    coll = GateCollector(n_features=F, top_k=K, seq_len=T)
    g = _zero_gates(1)
    g[0, 0, 0] = 1.0
    coll.update(torch.zeros(1, T, dtype=torch.long), g)
    nonzero_bins = (coll.act_histogram[0] > 0).nonzero().flatten().tolist()
    assert len(nonzero_bins) == 1
    # Bin index equals HIST_N_BINS / 2 since 1.0 is at the geometric midpoint
    # of [10^-3, 10^3].
    expected = GateCollector.HIST_N_BINS // 2
    assert nonzero_bins[0] == expected


def test_feature_summary_carries_histogram():
    coll = GateCollector(n_features=F, top_k=K, seq_len=T)
    g = _zero_gates(1)
    g[0, 0, 0] = 1.0
    coll.update(torch.zeros(1, T, dtype=torch.long), g)
    s = coll.feature_summary(0)
    assert len(s.act_histogram) == GateCollector.HIST_N_BINS
    assert sum(s.act_histogram) == 1
