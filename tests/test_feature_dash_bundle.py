"""Tests for the single-file `bundle.html` builder.

Two paths to the bundle:
  - `make_bundle_from_disk(out_dir)`: read an existing dump, write bundle.html.
  - `make_bundle(collector, meta, tokenizer, out_path, ...)`: build directly
    from in-memory state, no disk dump needed.

Both should produce a self-contained file (no `fetch(` calls, all data inlined)
that contains every feature's payload.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

from crosslayer_transcoder.feature_dash.bundle import (
    default_bundle_filename,
    make_bundle,
    make_bundle_from_disk,
)
from crosslayer_transcoder.feature_dash.collect import GateCollector
from crosslayer_transcoder.feature_dash.dump import dump_dashboard
from crosslayer_transcoder.feature_dash.load import MoltCheckpointMetadata


class _StubTokenizer:
    def decode(self, ids):
        return f" tok{ids[0]}"


def _populate_collector(F: int, K: int, T: int) -> GateCollector:
    coll = GateCollector(n_features=F, top_k=K, seq_len=T)
    tok = torch.arange(2 * T, dtype=torch.long).reshape(2, T)
    g = torch.zeros(2, T, F)
    g[0, 1, 0] = 1.0
    g[1, 3, 0] = 2.5
    g[0, 0, 1] = 0.5
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


def _extract_inlined_json(html: str, script_id: str) -> dict:
    """Pull a `<script id="X" type="application/json">{...}</script>` payload."""
    m = re.search(
        rf'<script id="{re.escape(script_id)}" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert m, f"missing inlined script #{script_id}"
    raw = m.group(1)
    # Reverse the `<` hardening before parsing.
    raw = raw.replace(r"<", "<")
    return json.loads(raw)


def test_bundle_contains_no_fetch_calls(tmp_path: Path):
    """Bundle must work from file:// — no network/disk fetches allowed."""
    F, K, T, N = 3, 2, 5, 1
    coll = _populate_collector(F, K, T)
    meta = _meta(F, [4, 2], N)

    out = make_bundle(
        collector=coll,
        meta=meta,
        tokenizer=_StubTokenizer(),
        out_path=tmp_path / "bundle.html",
        layer=8,
        window=2,
    )
    text = out.read_text()
    # No fetch(... call anywhere — neither real nor templated.
    assert "fetch(" not in text


def test_bundle_inlines_metadata_and_all_features(tmp_path: Path):
    F, K, T, N = 3, 2, 5, 1
    coll = _populate_collector(F, K, T)
    meta = _meta(F, [4, 2], N)

    bundle_path = make_bundle(
        collector=coll, meta=meta, tokenizer=_StubTokenizer(),
        out_path=tmp_path, layer=8, window=2,
    )
    # When out_path is a directory, the bundle is named after the checkpoint.
    # `_meta()` sets ckpt_path="dummy.ckpt" and no hf_filename, so stem="dummy".
    assert bundle_path == tmp_path / "bundle_dummy.html"

    html = bundle_path.read_text()
    md = _extract_inlined_json(html, "metadata")
    feats = _extract_inlined_json(html, "features")

    assert md["n_features"] == F
    assert md["layer"] == 8
    assert md["feature_tier"] == [0, 1, 1]
    assert md["feature_rank"] == [4, 2, 2]
    assert len(md["feature_activation_rate"]) == F
    assert len(md["feature_max_activation"]) == F

    assert set(feats.keys()) == {"0", "1", "2"}
    for k, v in feats.items():
        assert v["feature_id"] == int(k)
        assert {"tier", "rank", "activation_rate", "max_activation", "examples"} <= set(v.keys())

    # Feature 0 has two real examples and feature 2 is dead.
    assert len(feats["0"]["examples"]) == 2
    assert feats["2"]["examples"] == []


def test_bundle_css_is_inlined(tmp_path: Path):
    F, K, T, N = 3, 2, 5, 1
    coll = _populate_collector(F, K, T)
    meta = _meta(F, [4, 2], N)
    out = make_bundle(
        collector=coll, meta=meta, tokenizer=_StubTokenizer(),
        out_path=tmp_path / "bundle.html", layer=8, window=2,
    )
    html = out.read_text()
    # No external stylesheet link, and a non-empty inline <style> block.
    assert '<link rel="stylesheet"' not in html
    assert "<style>" in html and "</style>" in html
    assert "table.features" in html  # one of our CSS rules


def test_bundle_from_disk_reads_dumped_dashboard(tmp_path: Path):
    F, K, T, N = 3, 2, 5, 1
    coll = _populate_collector(F, K, T)
    meta = _meta(F, [4, 2], N)

    dump_dashboard(
        collector=coll, meta=meta, tokenizer=_StubTokenizer(),
        out_dir=tmp_path, layer=8, window=2, copy_assets=False,
    )
    bundle_path = make_bundle_from_disk(tmp_path)

    # ckpt_path="dummy.ckpt" -> bundle_dummy.html
    assert bundle_path == tmp_path / "bundle_dummy.html"
    html = bundle_path.read_text()
    md = _extract_inlined_json(html, "metadata")
    feats = _extract_inlined_json(html, "features")
    assert md["n_features"] == F
    assert len(feats) == F


def test_bundle_from_disk_errors_when_data_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        make_bundle_from_disk(tmp_path)


def test_default_bundle_filename_prefers_hf_filename():
    F, N = 3, 1
    meta = _meta(F, [4, 2], N)
    meta.hf_filename = "gpt2-molt-lam-0_00015-50M.ckpt"
    meta.ckpt_path = "/some/random/local/path.ckpt"
    assert default_bundle_filename(meta) == "bundle_gpt2-molt-lam-0_00015-50M.html"


def test_default_bundle_filename_falls_back_to_ckpt_path():
    F, N = 3, 1
    meta = _meta(F, [4, 2], N)
    meta.hf_filename = None
    meta.ckpt_path = "/runs/exp42/checkpoint-final.ckpt"
    assert default_bundle_filename(meta) == "bundle_checkpoint-final.html"


def test_default_bundle_filename_handles_unsafe_chars():
    F, N = 3, 1
    meta = _meta(F, [4, 2], N)
    meta.hf_filename = "weird/name with spaces.ckpt"
    # The stem is "name with spaces" (Path.stem already drops the directory).
    # Spaces are sanitised to '-'.
    assert default_bundle_filename(meta) == "bundle_name-with-spaces.html"


def test_make_bundle_explicit_filepath_is_respected(tmp_path: Path):
    F, K, T, N = 3, 2, 5, 1
    coll = _populate_collector(F, K, T)
    meta = _meta(F, [4, 2], N)

    explicit = tmp_path / "my-custom-name.html"
    out = make_bundle(
        collector=coll, meta=meta, tokenizer=_StubTokenizer(),
        out_path=explicit, layer=8, window=2,
    )
    # When the user passes a file path, we don't override it.
    assert out == explicit


def test_bundle_handles_script_tag_in_token_text(tmp_path: Path):
    """A token string containing `</script>` must not break the bundle.

    The sanitiser replaces `<` with `\\u003c`; the JS in the bundle parses the
    JSON via `JSON.parse`, which decodes the unicode escape back. We just
    verify here that the raw HTML doesn't contain a literal `</script>` inside
    the data payload.
    """

    class _NastyTokenizer:
        def decode(self, ids):
            return "</script>BAD"

    F, K, T, N = 3, 2, 5, 1
    coll = _populate_collector(F, K, T)
    meta = _meta(F, [4, 2], N)

    out = make_bundle(
        collector=coll, meta=meta, tokenizer=_NastyTokenizer(),
        out_path=tmp_path / "bundle.html", layer=8, window=2,
    )
    html = out.read_text()
    # Find the data scripts and check their content is sanitised.
    m = re.search(
        r'<script id="features" type="application/json">(.*?)</script>',
        html, flags=re.DOTALL,
    )
    assert m
    payload = m.group(1)
    # The sanitised payload must not contain a literal "</" sequence.
    assert "</" not in payload, payload
    # And the bundle as a whole still has exactly the two data scripts and
    # one logic script — no premature closure.
    assert html.count('<script') == 3
    assert html.count('</script>') == 3
