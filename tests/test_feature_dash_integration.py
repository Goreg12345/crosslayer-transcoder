"""Integration smoke test for the feature-dash pipeline.

Wires together: build tiny MoLT -> save Lightning ckpt -> load_molt ->
GateCollector with synthetic gates -> dump_dashboard -> assert the full file
contract holds.

We bypass `collect_features` (which would need a real GPT-2 + HF dataset) by
feeding the collector hand-crafted batches. The integration we care about
here is the file pipeline: that the loaded MoLT's metadata flows correctly
into the JSON, and that all the per-feature contracts the renderer relies on
are satisfied.

CLI smoke (Step 6) is also exercised here via argparse `--help`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import lightning as L
import pytest
import torch
from transformers import GPT2TokenizerFast

from crosslayer_transcoder.feature_dash import (
    GateCollector,
    dump_dashboard,
    load_molt,
)
from crosslayer_transcoder.model.clt_lightning import MoltModule
from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.molt import Molt
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)


D_ACTS = 16
N_LAYERS_LM = 4
N = 2
RANKS = [4, 2]                 # tier 0: 2 feats, tier 1: 4 feats -> 6 features total
LAYER = 1
SEQ_LEN = 8
TOP_K = 3


def _build_molt() -> Molt:
    n_features = N * 1 + N * 2  # 6
    nonlin = JumpReLU(theta=0.05, bandwidth=1.0, n_layers=1, d_features=n_features)
    in_std = DimensionwiseInputStandardizer(n_layers=N_LAYERS_LM, activation_dim=D_ACTS)
    out_std = DimensionwiseOutputStandardizer(
        n_layers=N_LAYERS_LM, activation_dim=D_ACTS
    )
    fake_batch = torch.randn(8, 2, N_LAYERS_LM, D_ACTS)
    in_std.initialize_from_batch(fake_batch)
    out_std.initialize_from_batch(fake_batch)

    return Molt(
        d_acts=D_ACTS,
        N=N,
        ranks=RANKS,
        nonlinearity=nonlin,
        input_standardizer=in_std,
        output_standardizer=out_std,
    )


def _save_lightning_checkpoint(tmp_path: Path, molt: Molt) -> str:
    module = MoltModule(model=molt)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        max_steps=0,
    )
    trainer.strategy.connect(module)
    ckpt = tmp_path / "molt.ckpt"
    trainer.save_checkpoint(str(ckpt))
    return str(ckpt)


def _populate_collector_with_molt_gates(
    molt: Molt, collector: GateCollector, n_batches: int = 4, batch_size: int = 4
) -> None:
    """Run synthetic resid through the loaded MoLT to get realistic gates."""
    torch.manual_seed(0)
    for _ in range(n_batches):
        # Random residuals; the standardizer will re-center them.
        resid = torch.randn(batch_size, SEQ_LEN, D_ACTS)
        resid_std = molt.input_standardizer(resid, LAYER)
        gates = molt.nonlinearity(molt.e(resid_std))     # (B, T, F)
        # Synthetic token ids: just whatever, the renderer doesn't care.
        token_ids = torch.randint(
            0, 1000, (batch_size, SEQ_LEN), dtype=torch.long
        )
        collector.update(token_ids, gates.float())


def test_full_pipeline_contract(tmp_path: Path):
    """Load -> collect -> dump produces a directory the renderer can consume."""
    src_molt = _build_molt()
    ckpt_path = _save_lightning_checkpoint(tmp_path, src_molt)

    molt, meta = load_molt(ckpt_path, device="cpu")
    assert meta.n_features == src_molt.n_features

    collector = GateCollector(
        n_features=meta.n_features, top_k=TOP_K, seq_len=SEQ_LEN
    )
    _populate_collector_with_molt_gates(molt, collector)

    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")

    out_dir = tmp_path / "dash"
    dump_dashboard(
        collector=collector,
        meta=meta,
        tokenizer=tokenizer,
        out_dir=out_dir,
        layer=LAYER,
        dataset_name="synthetic",
        window=2,
    )

    # --- File layout ---
    for rel in [
        "data/metadata.json",
        "index.html",
        "dashboard.html",
        "assets/dashboard.css",
        "assets/dashboard.js",
        "assets/index.js",
    ]:
        assert (out_dir / rel).is_file(), f"missing {rel}"
    feat_files = sorted((out_dir / "data" / "features").glob("*.json"))
    assert len(feat_files) == meta.n_features

    # --- metadata.json invariants the renderer relies on ---
    md = json.loads((out_dir / "data" / "metadata.json").read_text())
    assert md["schema_version"] == 1
    assert md["n_features"] == meta.n_features
    assert len(md["feature_activation_rate"]) == meta.n_features
    assert len(md["feature_max_activation"]) == meta.n_features
    assert len(md["feature_tier"]) == meta.n_features
    assert len(md["feature_rank"]) == meta.n_features
    assert all(0.0 <= r <= 1.0 for r in md["feature_activation_rate"])
    assert all(m >= 0.0 for m in md["feature_max_activation"])

    # --- per-feature invariants ---
    for fp in feat_files:
        d = json.loads(fp.read_text())
        assert {"feature_id", "tier", "rank", "activation_rate",
                "max_activation", "examples"} <= set(d.keys())
        assert 0.0 <= d["activation_rate"] <= 1.0
        for ex in d["examples"]:
            assert {"peak_activation", "peak_token_pos", "tokens", "activations"} <= set(ex.keys())
            assert len(ex["tokens"]) == len(ex["activations"])
            assert 0 <= ex["peak_token_pos"] < len(ex["tokens"])
            # peak_token_pos must index the argmax of the windowed activations.
            argmax = max(
                range(len(ex["activations"])),
                key=lambda i: ex["activations"][i],
            )
            assert argmax == ex["peak_token_pos"]
            # Peak activation must be > 0 (dead-feature filter in feature_summary).
            assert ex["peak_activation"] > 0


def test_cli_help_runs():
    """Make sure argparse construction (and therefore imports) don't blow up."""
    result = subprocess.run(
        [sys.executable, "-m", "crosslayer_transcoder.feature_dash", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "feature dashboard" in result.stdout.lower()
    # Both source flags should appear.
    assert "--hf-filename" in result.stdout
    assert "--local-ckpt" in result.stdout
    assert "--out" in result.stdout
