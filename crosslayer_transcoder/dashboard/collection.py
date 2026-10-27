"""Resumable, bounded-memory storage for sparse gate collection."""

import json
from pathlib import Path

import numpy as np


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


class GateCollection:
    """Commit whole conversation batches atomically; resume only matching inputs."""

    def __init__(self, root, settings):
        self.root = Path(root)
        self.parts = self.root / "collection"
        self.parts.mkdir(parents=True, exist_ok=True)
        manifest = self.parts / "settings.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != settings:
                raise ValueError(
                    "Collection settings changed; use a new output directory"
                )
        else:
            if (self.root / "activations.npz").exists():
                raise ValueError("Refusing to replace an existing, unsharded sample")
            write_json(manifest, settings)
        self.paths = sorted(self.parts.glob("part-*.npz"))
        self.tokens = self.conversations = self.entries = 0
        for number, path in enumerate(self.paths):
            if path.name != f"part-{number:06d}.npz":
                raise ValueError("Collection has a missing shard")
            with np.load(path) as part:
                offsets = part["offsets"]
                if offsets[0] != self.tokens or np.any(np.diff(offsets) <= 0):
                    raise ValueError("Invalid conversation boundaries in collection")
                self.tokens = int(offsets[-1])
                self.conversations += len(offsets) - 1
                self.entries += int(part["entries"])

    def append(self, records):
        """Records contain tokens and sparse positions relative to each conversation."""
        if not records:
            return
        offsets = self.tokens + np.concatenate(
            [[0], np.cumsum([len(r["tokens"]) for r in records])]
        )
        if offsets[-1] >= np.iinfo(np.int32).max:
            raise ValueError("This cache format supports fewer than 2^31 tokens")
        arrays = dict(
            tokens=np.concatenate([r["tokens"] for r in records]).astype(np.int32),
            offsets=offsets,
            positions=np.concatenate(
                [r["positions"] + lo for r, lo in zip(records, offsets)]
            ).astype(np.int32),
            features=np.concatenate([r["features"] for r in records]).astype(np.int32),
            values=np.concatenate([r["values"] for r in records]).astype(np.float32),
        )
        path = self.parts / f"part-{len(self.paths):06d}.npz"
        tmp = path.with_suffix(".npz.tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays, entries=len(arrays["values"]))
        tmp.replace(path)
        self.paths.append(path)
        self.tokens = int(offsets[-1])
        self.conversations += len(records)
        self.entries += len(arrays["values"])

    def finish(self):
        if not self.paths:
            raise ValueError("No conversations collected")
        arrays = {}
        for key in ("tokens", "offsets", "positions", "features", "values"):
            pieces = []
            for i, path in enumerate(self.paths):
                with np.load(path) as part:
                    value = part[key]
                    pieces.append(value[1:] if key == "offsets" and i else value)
            arrays[key] = np.concatenate(pieces)
        tmp = self.root / "activations.npz.tmp"
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        tmp.replace(self.root / "activations.npz")
        return np.unique(arrays["tokens"])
