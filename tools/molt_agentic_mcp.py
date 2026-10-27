#!/usr/bin/env python3
"""Training-only MOLT tools for the agentic Delphi pilot.

The server deliberately offers no file access, baseline explanation, or scoring
endpoint. Corpus examples come exclusively from validated training windows.
Quantiles describe these selected windows, not the full activation distribution.
Only verify_hypothesis loads Qwen3-4B and the MOLT encoder on the requested GPU.
"""
from __future__ import annotations

import argparse
import copy
from functools import wraps
import json
import math
from pathlib import Path
import sys
import threading
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crosslayer_transcoder.dashboard.token_display import (
    QWEN_ADDED_TOKENS, display_form, escape_added_tokens,
)


def _safe_strings(value):
    if isinstance(value, str):
        return escape_added_tokens(value)
    if isinstance(value, list):
        return [_safe_strings(v) for v in value]
    if isinstance(value, dict):
        return {_safe_strings(k): _safe_strings(v) for k, v in value.items()}
    return value


def safe_result(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        return _safe_strings(function(*args, **kwargs))
    return wrapped


def restore_probe_tokens(text):
    """Restore only known Qwen display forms for deliberate subject-model probes."""
    for token in QWEN_ADDED_TOKENS:
        text = text.replace(display_form(token), token)
    return text


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return value


class RemoteInspectionEngine:
    """Narrow client: only metadata identity and arbitrary-text inspection."""
    def __init__(self, directory, *, url, device=None):
        self.url = url.rstrip("/")
        local = json.loads((Path(directory) / "metadata.json").read_text())
        with urlopen(self.url + "/metadata.json", timeout=30) as response:
            remote = json.load(response)
        for key in ("checkpoint", "global_step", "model", "layer", "checkpoint_mtime"):
            if key not in local or local[key] != remote.get(key):
                raise ValueError(f"Inspection service {key} does not match evidence")

    def inspect(self, text, prompt_format, feature):
        request = Request(self.url + "/api/inspect", data=json.dumps({
            "prompt": text, "format": prompt_format, "feature": feature,
        }).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=180) as response:
            return json.load(response)


class TrainingEvidence:
    def __init__(self, directory, selection_path, evidence_directory=None, *,
                 device="cuda:0", engine_factory=None, inspection_url=None):
        self.directory = Path(directory)
        self.evidence_directory = Path(evidence_directory or self.directory / "autointerp-layer")
        selection = json.loads(Path(selection_path).read_text())
        self.features = frozenset(int(row["id"]) for row in selection["selected"])
        manifest = json.loads((self.evidence_directory / "manifest.json").read_text())
        settings = manifest.get("settings", manifest)
        self.train_conversations = frozenset(settings["train_conversations"])
        self.test_conversations = frozenset(settings["test_conversations"])
        if not self.train_conversations or self.train_conversations & self.test_conversations:
            raise ValueError("Training and held-out conversation sets must be nonempty/disjoint")
        metadata = json.loads((self.directory / "metadata.json").read_text())
        for name in ("checkpoint", "global_step"):
            if settings.get(name) != metadata.get(name):
                raise ValueError(f"Evidence and model {name} mismatch")
        self.layer = metadata["layer"]
        self.model_name = metadata["model"]
        self.device = device
        self._engine_factory = engine_factory
        if inspection_url is not None:
            self._engine_factory = lambda directory, device: RemoteInspectionEngine(directory, url=inspection_url)
        self._engine = None
        self._lock = threading.Lock()
        self._rows = {}
        # Validate every permitted feature at startup, before serving any examples.
        for feature in self.features:
            payload = json.loads((self.evidence_directory / "evidence" / f"{feature}.json").read_text())
            rows = []
            for row in payload["train"]:
                if row["conversation"] not in self.train_conversations:
                    raise ValueError(f"Feature {feature} contains a non-training conversation")
                tokens, acts = row["str_tokens"], row["activations"]
                if not tokens or len(tokens) != len(acts) or len(tokens) != len(row["tokens"]):
                    raise ValueError("Unaligned training tokens and gates")
                if any(not isinstance(x, str) for x in tokens) or any(
                    isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
                    for x in acts
                ):
                    raise ValueError("Invalid training token or gate")
                # Explicit allowlist: never forward other fields from source JSON.
                rows.append({"conversation": row["conversation"], "start": row["start"],
                             "stop": row["stop"],
                             "coordinate_system": "start/stop are global cache offsets, not within-conversation token positions",
                             "sampling_note": "Windows end at a selected activation peak by construction. End-of-window position is a sampling artifact, not evidence of a trigger.",
                             "evidence_scope": "selected explanation-training windows only; preceding conversation context may be omitted",
                             "tokens": list(tokens),
                             "token_ids": list(row["tokens"]), "activations": list(acts),
                             "max_activation": max(acts),
                             "marked_text": "".join(f"<<{t}>>" if a > 0 else t for t, a in zip(tokens, acts))})
            if not rows:
                raise ValueError(f"Feature {feature} has no training windows")
            self._rows[feature] = sorted(rows, key=lambda row: (-row["max_activation"], row["start"]))

    def _feature(self, feature):
        if type(feature) is not int or feature not in self.features:
            raise ValueError("Feature is not in the permitted pilot cohort")
        return self._rows[feature]

    @safe_result
    def get_latent_info(self, feature):
        rows = self._feature(feature)
        acts = [a for row in rows for a in row["activations"]]
        return {"feature": feature, "layer": self.layer, "model": self.model_name,
                "activation_kind": "nonnegative scalar MOLT transform gate (not transform output magnitude)",
                "evidence_scope": "selected explanation-training windows only; preceding conversation context may be omitted",
                "coordinate_system": "start/stop are global cache offsets, not within-conversation token positions",
                "sampling_note": "Windows end at a selected activation peak by construction. End-of-window position is a sampling artifact, not evidence of a trigger.",
                "training_windows": len(rows), "observed_window_tokens": len(acts),
                "active_window_tokens": sum(a > 0 for a in acts), "training_max_activation": max(acts)}

    @safe_result
    def top_activated_examples(self, feature, count=5, offset=0):
        rows = self._feature(feature)
        _integer(count, "count", 1, 12)
        _integer(offset, "offset", 0, len(rows))
        return {"feature": feature, "scope": "training", "total_available": len(rows),
                "examples": copy.deepcopy(rows[offset:offset + count])}

    @safe_result
    def quantile_samples(self, feature, quantile=0.5, count=3):
        rows = list(reversed(self._feature(feature)))
        if isinstance(quantile, bool) or not isinstance(quantile, (int, float)) or not math.isfinite(quantile) or not 0 <= quantile <= 1:
            raise ValueError("quantile must be between 0 and 1")
        _integer(count, "count", 1, 12)
        center = round(quantile * (len(rows) - 1))
        indices = sorted(range(len(rows)), key=lambda i: (abs(i - center), i))[:count]
        return {"feature": feature, "quantile": quantile,
                "quantile_population": "selected training-window maxima; not corpus quantiles",
                "examples": copy.deepcopy([rows[i] for i in indices])}

    @safe_result
    def verify_hypothesis(self, feature, texts, prompt_format="raw"):
        self._feature(feature)
        if not isinstance(texts, list) or not 1 <= len(texts) <= 8:
            raise ValueError("texts must contain between 1 and 8 samples")
        if any(not isinstance(t, str) or not t.strip() or len(t) > 20000 for t in texts):
            raise ValueError("Each text must contain 1–20,000 characters")
        if sum(map(len, texts)) > 40000:
            raise ValueError("Combined texts must be at most 40,000 characters")
        if prompt_format not in ("raw", "chat"):
            raise ValueError("prompt_format must be raw or chat")
        with self._lock:
            if self._engine is None:
                factory = self._engine_factory
                if factory is None:
                    from crosslayer_transcoder.dashboard.steering import SteeringEngine
                    factory = SteeringEngine
                self._engine = factory(self.directory, device=self.device)
            results = []
            for text in texts:
                inspected = self._engine.inspect(restore_probe_tokens(text), prompt_format, feature)
                tokens = inspected["tokens"]
                acts = [t["gate"] for t in tokens]
                results.append({"tokens": [t["text"] for t in tokens],
                                "token_ids": [t["id"] for t in tokens], "activations": acts,
                                "max_activation": max(acts, default=0),
                                "is_activated": any(a > 0 for a in acts),
                                "marked_text": "".join(f'<<{t["text"]}>>' if t["gate"] > 0 else t["text"] for t in tokens)})
        return {"feature": feature, "prompt_format": prompt_format,
                "activation_kind": "actual Qwen3 forward pass through MOLT scalar gate",
                "results": results}


def build_server(evidence):
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:
        from mcp.server.fastmcp import FastMCP as MCPServer
    server = MCPServer("molt-agentic-training")

    @server.tool()
    def get_latent_info(feature: int) -> dict:
        """Describe a feature using only its selected training-window gate statistics."""
        return evidence.get_latent_info(feature)

    @server.tool()
    def top_activated_examples(feature: int, count: int = 5, offset: int = 0) -> dict:
        """Read strongest selected training windows, with raw per-token gates. Page using offset."""
        return evidence.top_activated_examples(feature, count, offset)

    @server.tool()
    def quantile_samples(feature: int, quantile: float = 0.5, count: int = 3) -> dict:
        """Read windows near a quantile of selected training-window maxima (not corpus percentiles)."""
        return evidence.quantile_samples(feature, quantile, count)

    @server.tool()
    def verify_hypothesis(feature: int, texts: list[str], prompt_format: str = "raw") -> dict:
        """Test 1–8 invented texts with actual Qwen3+MOLT; max 2048 tokens each. Known display forms such as ⟪think⟫ and ⟪im_start⟫ are restored to subject-model control tokens for intentional probes. Results escape every Qwen added token; numeric token IDs are preserved."""
        return evidence.verify_hypothesis(feature, texts, prompt_format)

    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--evidence-directory", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument("--inspection-url", help="Reuse a matching dashboard Qwen3/MOLT engine")
    args = parser.parse_args()
    evidence = TrainingEvidence(args.directory, args.selection, args.evidence_directory, device=args.device, inspection_url=args.inspection_url)
    server = build_server(evidence)
    if args.transport == "streamable-http":
        server.run(transport=args.transport, host=args.host, port=args.port)
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
