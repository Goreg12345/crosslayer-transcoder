#!/usr/bin/env python3
"""Queue 10M-token collection and four-score interpretation behind layer 22."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from crosslayer_transcoder.dashboard.collection import write_json
from tools.molt_checkpoint_step import checkpoint_step

ORDER = [22, 0, 1, 10, 30] + [n for n in range(36) if n not in (22, 0, 1, 10, 30)]
CHECKPOINT_NAME = "N80-rank512to32-lr4e-5-sp7.5e-5-native-ramp80"
REFERENCE = Path("results/molt-qwen-dashboard-campaign-layer22-10m")
PORTAL = Path("results/molt-qwen-campaign-interpretation")
RETRY_DELAY_SECONDS = 6 * 60 * 60


class LayerIncompleteError(RuntimeError):
    """A layer exhausted its worker resumptions; other layers can still proceed."""


def now():
    return datetime.now(timezone.utc).isoformat()


def checkpoint_path(layer):
    return (
        Path(f"checkpoints/molt-qwen3-4b/layer-{layer}")
        / CHECKPOINT_NAME
        / "training.ckpt"
    )


def sample_path(layer):
    return Path(f"results/molt-qwen-dashboard-campaign-layer{layer}-10m")


def unit_active(unit):
    return (
        subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


class Inventory:
    def __init__(self):
        self.checkpoints = {}
        self.jsons = {}

    def read(self, path):
        if not path.exists():
            return {}
        stamp = path.stat().st_mtime_ns
        old = self.jsons.get(path)
        if old and old[0] == stamp:
            return old[1]
        value = json.loads(path.read_text())
        self.jsons[path] = stamp, value
        return value

    def step(self, layer):
        path = checkpoint_path(layer)
        if not path.exists():
            return None
        stat = path.stat()
        stamp = stat.st_size, stat.st_mtime_ns
        old = self.checkpoints.get(layer)
        if old and old[0] == stamp:
            return old[1]
        try:
            step = checkpoint_step(path)
        except (OSError, RuntimeError, EOFError, ValueError):
            return old[1] if old else None
        self.checkpoints[layer] = stamp, step
        return step

    def rows(self):
        rows = []
        retries = self.read(PORTAL / ".api-retries.json")
        for layer in ORDER:
            root = sample_path(layer)
            progress = self.read(root / "collection-progress.json")
            meta = self.read(root / "metadata.json")
            cached = bool(
                progress.get("complete")
                and meta.get("tokens") == 10_000_000
                and meta.get("global_step") == 100_000
                and meta.get("layer") == layer
                and meta.get("checkpoint") == str(checkpoint_path(layer).resolve())
                and (root / "activations.npz").exists()
                and (root / "vocab.json").exists()
            )
            summary = self.read(root / "autointerp-layer" / "summary.json")
            features = summary.get("features", {})
            retry = retries.get(str(layer), {})
            if retry.get("fingerprint") != summary.get("fingerprint"):
                retry = {}
            rows.append(
                dict(
                    layer=layer,
                    checkpoint_step=self.step(layer),
                    cached=cached,
                    tokens=progress.get("tokens", 0),
                    tokens_per_second=progress.get("tokens_per_second"),
                    collection_complete=progress.get("complete", False),
                    activation_page=(root / "index.html").exists(),
                    review_page=(root / "autointerp-layer" / "index.html").exists(),
                    api_state=summary.get("run", {}).get("state", "pending"),
                    api_phase=summary.get("run", {}).get("phase"),
                    interpretation_fingerprint=summary.get("fingerprint"),
                    api_retry_after=retry.get("retry_after"),
                    api_retry_reason=retry.get("reason"),
                    explained=summary.get("run", {}).get("explained", 0),
                    observed=summary.get("run", {}).get("observed"),
                    scored=sum(
                        all(
                            r.get("methods", {}).get(m, {}).get("status") == "complete"
                            for m in ("detection", "fuzz", "active_fuzz", "intruder")
                        )
                        for r in features.values()
                    ),
                )
            )
        return rows


def publish(mode, inventory, state, active=(), error=None):
    rows = inventory.rows()
    write_json(
        PORTAL / f"{mode}-queue.json",
        dict(
            state=state,
            active_layers=list(active),
            updated_at=now(),
            layers=rows,
            error=error,
        ),
    )
    return rows


def _setup_portal():
    PORTAL.mkdir(parents=True, exist_ok=True)
    for layer in ORDER:
        link = PORTAL / f"layer-{layer}"
        destination = sample_path(layer).resolve()
        if not link.is_symlink():
            if link.exists():
                raise ValueError(f"Portal path already exists: {link}")
            link.symlink_to(destination, target_is_directory=True)
        elif link.resolve() != destination:
            raise ValueError("Portal symlink points to a different sample")
    route = Path("results/molt-qwen-dashboard-campaign-layer22/layers")
    if not route.is_symlink():
        if route.exists():
            raise ValueError("Layers route already exists")
        route.symlink_to(PORTAL.resolve(), target_is_directory=True)
    elif route.resolve() != PORTAL.resolve():
        raise ValueError("Layers route points to a different directory")
    source = Path(__file__).with_name("molt_campaign_index.html")
    (PORTAL / "index.html").write_text(source.read_text())
    plan = dict(
        created_at=now(),
        layers=ORDER,
        reference=str(REFERENCE.resolve()),
        target_tokens=10_000_000,
        methods=["detection", "fuzz", "active_fuzz", "intruder"],
        concurrency=32,
        checkpoint_policy="Only completed 100,000-step campaign checkpoints; discover later layers as training finishes",
        collection_gpu="GPU-2ada3848-a497-abbf-c0da-c053a9680677",
        allocator_cap_gib=10,
        api_policy="Wait for layer 22, then one layer at a time; simulation disabled",
    )
    if not (PORTAL / "plan.json").exists():
        write_json(PORTAL / "plan.json", plan)


def setup_portal():
    PORTAL.mkdir(parents=True, exist_ok=True)
    with (PORTAL / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _setup_portal()


def collect_queue(inventory, once=False):
    while True:
        # The first cohort was started immediately; adopt its progress without
        # creating another CUDA process or touching its collection files.
        if unit_active("molt-collect-other-layers.service"):
            rows = inventory.rows()
            active = [
                r["layer"]
                for r in rows
                if r["layer"] != 22
                and inventory.read(
                    sample_path(r["layer"]) / "collection-progress.json"
                ).get("shared_layers")
                and not r["cached"]
            ]
            publish("collection", inventory, "collecting initial cohort", active)
            time.sleep(15)
            continue
        rows = publish("collection", inventory, "checking checkpoints")
        # Publish cached activations before their turn on the remote endpoint.
        # The renderer's per-directory lock also protects against an API publisher.
        for row in rows:
            if row["layer"] == 22 or not row["cached"] or row["activation_page"]:
                continue
            layer = row["layer"]
            with Path("logs/molt-collection-queue-render.log").open("a") as log:
                renderer = subprocess.Popen(
                    [
                        str(Path.cwd() / ".venv-dashboard/bin/python"),
                        "tools/render_molt_dashboard.py",
                        str(sample_path(layer)),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                while renderer.poll() is None:
                    publish(
                        "collection", inventory, "publishing activation pages", [layer]
                    )
                    time.sleep(15)
                if renderer.returncode:
                    raise RuntimeError(f"Activation rendering failed for layer {layer}")
        pending = [r for r in rows if r["layer"] != 22 and not r["cached"]]
        if not pending:
            publish("collection", inventory, "complete")
            return
        ready = [r["layer"] for r in pending if r["checkpoint_step"] == 100_000]
        if not ready:
            publish("collection", inventory, "waiting for completed checkpoints")
            if once:
                return
            time.sleep(30)
            continue
        command = [
            str(Path.cwd() / ".venv/bin/python"),
            "tools/collect_molt_layers.py",
            "--gpu-memory-gib",
            "10",
            "--checkpoints",
            *[str(checkpoint_path(layer)) for layer in ready],
        ]
        print(
            json.dumps(dict(event="start_collection", layers=ready, time=now())),
            flush=True,
        )
        with Path("logs/molt-collection-queue-workers.log").open("a") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                publish("collection", inventory, "collecting", ready)
                time.sleep(15)
        if process.returncode:
            raise RuntimeError(
                f"Collection exited {process.returncode}; shards are resumable; see logs/molt-collection-queue-workers.log"
            )


def explain_layer(layer, inventory):
    root = sample_path(layer)
    worker_command = [
        str(Path.cwd() / "local/autointerp-venv/bin/python"),
        "tools/run_molt_layer_explanations.py",
        str(root),
    ]
    render_command = [
        str(Path.cwd() / ".venv/bin/python"),
        "tools/render_molt_layer.py",
        str(root),
        "--watch",
        "--publish-explorer",
    ]
    for attempt in range(1, 4):
        renderer = None
        with (
            Path(f"logs/molt-autointerp-layer{layer}-queued.log").open(
                "a"
            ) as worker_log,
            Path(f"logs/molt-autointerp-layer{layer}-queued-review.log").open(
                "a"
            ) as review_log,
        ):
            print(
                json.dumps(
                    dict(event="start_layer", layer=layer, attempt=attempt, time=now())
                ),
                flush=True,
            )
            worker = subprocess.Popen(
                worker_command, stdout=worker_log, stderr=subprocess.STDOUT
            )
            while worker.poll() is None:
                rows = publish("api", inventory, "interpreting", [layer])
                current = next(r for r in rows if r["layer"] == layer)
                if renderer is None and current["api_state"] == "running":
                    renderer = subprocess.Popen(
                        render_command,
                        stdout=review_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                time.sleep(15)
            if renderer is not None:
                if worker.returncode and renderer.poll() is None:
                    os.killpg(renderer.pid, signal.SIGTERM)
                while renderer.poll() is None:
                    publish("api", inventory, "publishing results", [layer])
                    time.sleep(15)
            summary = inventory.read(root / "autointerp-layer" / "summary.json")
            if (
                worker.returncode == 0
                and summary.get("run", {}).get("state") == "complete"
            ):
                if renderer is None or renderer.returncode:
                    subprocess.run(
                        render_command[:-2] + ["--publish-explorer"],
                        stdout=review_log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                return
        print(
            json.dumps(
                dict(
                    event="retry_layer",
                    layer=layer,
                    attempt=attempt,
                    exit_code=worker.returncode,
                )
            ),
            flush=True,
        )
        time.sleep(30)
    raise LayerIncompleteError(
        f"Layer {layer} still has incomplete work after resumptions; inspect its queued log"
    )


def api_queue(inventory, once=False):
    retry_path = PORTAL / ".api-retries.json"
    retries = dict(inventory.read(retry_path))
    while True:
        rows = publish("api", inventory, "checking prerequisites")
        if all(r["api_state"] == "complete" for r in rows):
            publish("api", inventory, "complete")
            return
        if unit_active("molt-autointerp-layer22.service"):
            publish("api", inventory, "waiting for active layer 22 to finish", [22])
            if once:
                return
            time.sleep(15)
            continue
        ready = [
            r
            for r in rows
            if r["cached"]
            and r["api_state"] != "complete"
            and (
                retries.get(str(r["layer"]), {}).get("fingerprint")
                != r.get("interpretation_fingerprint")
                or retries.get(str(r["layer"]), {}).get("retry_after", 0) <= time.time()
            )
        ]
        if not ready:
            publish(
                "api", inventory, "waiting for 10M-token caches or deferred retries"
            )
            if once:
                return
            time.sleep(30)
            continue
        layer = ready[0]["layer"]
        try:
            explain_layer(layer, inventory)
        except LayerIncompleteError as exc:
            summary = inventory.read(
                sample_path(layer) / "autointerp-layer/summary.json"
            )
            retries[str(layer)] = dict(
                fingerprint=summary.get("fingerprint"),
                retry_after=time.time() + RETRY_DELAY_SECONDS,
                reason=str(exc),
            )
            write_json(retry_path, retries)
            print(
                json.dumps(
                    dict(event="defer_layer", layer=layer, **retries[str(layer)])
                ),
                flush=True,
            )
            # Preserve the layer's incomplete results, then let the next ready
            # layer use the endpoint. The cooldown persists across service restarts.
        else:
            if str(layer) in retries:
                del retries[str(layer)]
                write_json(retry_path, retries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["collection", "api"])
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit instead of waiting for an unavailable checkpoint/cache",
    )
    args = parser.parse_args()
    setup_portal()
    lock = (PORTAL / f".{args.mode}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inventory = Inventory()
    try:
        (collect_queue if args.mode == "collection" else api_queue)(
            inventory, args.once
        )
    except Exception as exc:
        publish(args.mode, inventory, "error", error=f"{type(exc).__name__}: {exc}")
        raise
