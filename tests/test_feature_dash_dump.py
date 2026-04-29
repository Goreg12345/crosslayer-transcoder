"""Tests for windowing (Step 3) and JSON dump (Step 4).

The windowing helper is checked with a stub tokenizer so the test stays CPU
and offline. The dump test builds a tiny GateCollector + MoltCheckpointMetadata
by hand, runs `dump_dashboard`, and checks the file layout and JSON schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from crosslayer_transcoder.feature_dash.collect import (
    FeatureSummary,
    GateCollector,
    window_example,
    window_feature_summary,
)
from crosslayer_transcoder.feature_dash.dump import dump_dashboard
from crosslayer_transcoder.feature_dash.load import MoltCheckpointMetadata


class _StubTokenizer:
    """Minimal HF-tokenizer-like object: id -> string with a leading space."""

    def decode(self, ids):
        # Match HF behavior of single-id decode for byte-level BPE: returns
        # the raw token text including leading whitespace where applicable.
        return f" tok{ids[0]}"


# ---- Step 3 -----------------------------------------------------------------


def test_window_example_centers_on_peak():
    ids = list(range(10))                       # 0..9
    acts = [0.0] * 10
    acts[6] = 5.0                               # peak at index 6

    out = window_example(ids, acts, _StubTokenizer(), window=2)

    # window=2 -> indices [4, 5, 6, 7, 8] = 5 tokens
    assert out["tokens"] == [" tok4", " tok5", " tok6", " tok7", " tok8"]
    assert out["activations"] == [0.0, 0.0, 5.0, 0.0, 0.0]
    assert out["peak_token_pos"] == 2          # index of 6 inside the window
    assert out["peak_activation"] == 5.0


def test_window_example_clips_at_left_edge():
    ids = list(range(10))
    acts = [0.0] * 10
    acts[1] = 3.0                               # peak near the start

    out = window_example(ids, acts, _StubTokenizer(), window=4)

    # window=4 around pos 1 -> [0..5], length 6, peak at pos 1
    assert out["tokens"] == [f" tok{i}" for i in range(6)]
    assert out["peak_token_pos"] == 1
    assert out["peak_activation"] == 3.0


def test_window_example_clips_at_right_edge():
    ids = list(range(10))
    acts = [0.0] * 10
    acts[9] = 7.0                               # peak at the end

    out = window_example(ids, acts, _StubTokenizer(), window=3)

    # window=3 around pos 9 -> [6..9], length 4, peak at last pos within window
    assert out["tokens"] == [" tok6", " tok7", " tok8", " tok9"]
    assert out["peak_token_pos"] == 3
    assert out["peak_activation"] == 7.0


def test_window_example_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        window_example([1, 2, 3], [0.1, 0.2], _StubTokenizer(), window=1)


def test_window_feature_summary_passes_through_metadata():
    summary = FeatureSummary(
        feature_id=42,
        activation_rate=0.05,
        max_activation=2.5,
        top_peaks=[2.5, 1.0],
        top_token_ids=[[10, 11, 12], [20, 21, 22]],
        top_activations=[[0.0, 2.5, 0.0], [1.0, 0.0, 0.0]],
        act_histogram=[0, 1, 2, 0, 0],
    )
    out = window_feature_summary(summary, _StubTokenizer(), window=10)

    assert out["feature_id"] == 42
    assert out["activation_rate"] == 0.05
    assert out["max_activation"] == 2.5
    assert len(out["examples"]) == 2
    assert out["examples"][0]["peak_activation"] == 2.5
    assert out["examples"][1]["peak_activation"] == 1.0
    assert out["act_histogram"] == [0, 1, 2, 0, 0]


# ---- Step 4 -----------------------------------------------------------------


def _populate_collector(F: int, K: int, T: int) -> GateCollector:
    coll = GateCollector(n_features=F, top_k=K, seq_len=T)
    # feature 0 fires on two sequences with different peaks
    tok = torch.arange(2 * T, dtype=torch.long).reshape(2, T)
    g = torch.zeros(2, T, F)
    g[0, 1, 0] = 1.0
    g[1, 3, 0] = 2.5
    g[0, 0, 1] = 0.5     # feature 1 fires once
    coll.update(tok, g)
    return coll


def _meta(F: int, ranks: list[int], N: int) -> MoltCheckpointMetadata:
    feature_tier: list[int] = []
    feature_rank: list[int] = []
    for t, r in enumerate(ranks):
        n_in_tier = N * (2**t)
        feature_tier.extend([t] * n_in_tier)
        feature_rank.extend([r] * n_in_tier)
    return MoltCheckpointMetadata(
        ckpt_path="dummy.ckpt",
        d_acts=8,
        n_features=F,
        n_layers=2,
        ranks=ranks,
        N=N,
        feature_tier=feature_tier,
        feature_rank=feature_rank,
        base_model_name="stub",
        training_dataset="stub-ds",
        global_step=123,
        epoch=0,
    )


def test_dump_dashboard_writes_metadata_and_features(tmp_path: Path):
    F, K, T, N = 3, 2, 5, 1
    ranks = [4, 2]                               # tier 0: 1 feat, tier 1: 2 feats -> 3
    coll = _populate_collector(F, K, T)
    meta = _meta(F, ranks, N)

    data_dir = dump_dashboard(
        collector=coll,
        meta=meta,
        tokenizer=_StubTokenizer(),
        out_dir=tmp_path,
        layer=8,
        dataset_name="Skylion007/openwebtext",
        window=2,
    )

    assert data_dir == tmp_path / "data"
    assert (data_dir / "metadata.json").is_file()

    feat_dir = data_dir / "features"
    files = sorted(feat_dir.glob("*.json"))
    assert len(files) == F
    # Filenames are zero-padded to width=4 (max(4, len("2"))).
    assert [p.name for p in files] == ["0000.json", "0001.json", "0002.json"]

    md = json.loads((data_dir / "metadata.json").read_text())
    assert md["schema_version"] == 1
    assert md["n_features"] == F
    assert md["ranks"] == ranks
    assert md["layer"] == 8
    assert md["dashboard_dataset"] == "Skylion007/openwebtext"
    assert md["n_tokens_collected"] == 2 * T
    assert md["seq_len"] == T
    assert md["top_k"] == K
    assert md["window"] == 2
    assert md["feature_tier"] == [0, 1, 1]
    assert md["feature_rank"] == [4, 2, 2]
    # Per-feature aggregates so the index page renders from one fetch.
    assert len(md["feature_activation_rate"]) == F
    assert len(md["feature_max_activation"]) == F
    assert md["feature_activation_rate"][0] == pytest.approx(2 / (2 * T))
    assert md["feature_max_activation"][0] == pytest.approx(2.5)
    assert md["feature_activation_rate"][2] == 0.0  # dead feature
    assert md["feature_max_activation"][2] == 0.0   # max clamped from -inf to 0


def test_dump_dashboard_copies_render_assets(tmp_path: Path):
    F, K, T, N = 3, 2, 5, 1
    ranks = [4, 2]
    coll = _populate_collector(F, K, T)
    meta = _meta(F, ranks, N)

    dump_dashboard(
        collector=coll,
        meta=meta,
        tokenizer=_StubTokenizer(),
        out_dir=tmp_path,
        layer=8,
        window=2,
    )
    # Static assets land alongside data/.
    for rel in [
        "index.html",
        "dashboard.html",
        "assets/dashboard.css",
        "assets/dashboard.js",
        "assets/index.js",
    ]:
        assert (tmp_path / rel).is_file(), f"missing {rel}"


def test_dump_dashboard_skips_assets_when_disabled(tmp_path: Path):
    F, K, T, N = 3, 2, 5, 1
    coll = _populate_collector(F, K, T)
    meta = _meta(F, [4, 2], N)

    dump_dashboard(
        collector=coll,
        meta=meta,
        tokenizer=_StubTokenizer(),
        out_dir=tmp_path,
        layer=8,
        window=2,
        copy_assets=False,
    )
    assert not (tmp_path / "index.html").exists()
    assert not (tmp_path / "assets").exists()
    # data/ still written.
    assert (tmp_path / "data" / "metadata.json").is_file()


def test_dump_dashboard_feature_payload_schema(tmp_path: Path):
    F, K, T, N = 3, 2, 5, 1
    ranks = [4, 2]
    coll = _populate_collector(F, K, T)
    meta = _meta(F, ranks, N)

    dump_dashboard(
        collector=coll,
        meta=meta,
        tokenizer=_StubTokenizer(),
        out_dir=tmp_path,
        layer=8,
        window=2,
    )
    feat0 = json.loads((tmp_path / "data" / "features" / "0000.json").read_text())

    # Required schema keys.
    assert set(feat0.keys()) >= {
        "feature_id",
        "tier",
        "rank",
        "activation_rate",
        "max_activation",
        "examples",
    }
    assert feat0["feature_id"] == 0
    assert feat0["tier"] == 0
    assert feat0["rank"] == 4
    assert feat0["activation_rate"] == pytest.approx(2 / (2 * T))
    assert feat0["max_activation"] == pytest.approx(2.5)

    # Two examples, sorted desc by peak.
    assert len(feat0["examples"]) == 2
    assert feat0["examples"][0]["peak_activation"] == pytest.approx(2.5)
    assert feat0["examples"][1]["peak_activation"] == pytest.approx(1.0)

    ex0 = feat0["examples"][0]
    assert len(ex0["tokens"]) == len(ex0["activations"])
    # peak_token_pos must index the argmax of the windowed activations.
    assert (
        max(range(len(ex0["activations"])), key=lambda i: ex0["activations"][i])
        == ex0["peak_token_pos"]
    )


def test_dump_dashboard_dead_feature_yields_no_examples(tmp_path: Path):
    F, K, T, N = 3, 2, 5, 1
    ranks = [4, 2]
    coll = _populate_collector(F, K, T)        # feature 2 never fires
    meta = _meta(F, ranks, N)

    dump_dashboard(
        collector=coll,
        meta=meta,
        tokenizer=_StubTokenizer(),
        out_dir=tmp_path,
        layer=8,
        window=2,
    )
    feat2 = json.loads((tmp_path / "data" / "features" / "0002.json").read_text())
    assert feat2["activation_rate"] == 0.0
    assert feat2["examples"] == []
