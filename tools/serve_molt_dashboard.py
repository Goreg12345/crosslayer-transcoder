#!/usr/bin/env python3
"""Serve the feature dashboard and a single-GPU token-steering API."""

import argparse
import json
import logging
import math
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import torch

from crosslayer_transcoder.dashboard.steering import SteeringEngine


class Application:
    def __init__(self, engine, additional_engines=()):
        self.engine = engine
        self.engines = {item.layer: item for item in (engine, *additional_engines)}
        self.jobs = {}
        self.guard = threading.RLock()
        self.worker = ThreadPoolExecutor(max_workers=1)

    def engine_for(self, layer=None):
        if layer is None:
            return self.engine
        try:
            if isinstance(layer, bool) or str(int(layer)) != str(layer):
                raise ValueError
            return self.engines[int(layer)]
        except (ValueError, TypeError, KeyError):
            raise ValueError("Unsupported steering layer") from None

    def submit(self, config, engine=None):
        engine = engine or self.engine
        if not engine.lock.acquire(blocking=False):
            raise BlockingIOError("GPU is busy. Wait for the current run or cancel it.")
        identifier = uuid.uuid4().hex
        event = threading.Event()
        with self.guard:
            completed = [
                key
                for key, job in self.jobs.items()
                if job["state"] in ("done", "error", "cancelled")
            ]
            for key in completed[:-15]:
                del self.jobs[key]
            self.jobs[identifier] = {
                "id": identifier,
                "state": "running",
                "phase": "baseline",
                "baseline": "",
                "steered": "",
                "tokens": 0,
                "cancel": event,
            }

        def run():
            try:

                def progress(branch, text, count):
                    with self.guard:
                        self.jobs[identifier].update(
                            phase=branch, tokens=count, **{branch: text}
                        )

                result = engine.compare(config, event, progress)
                with self.guard:
                    self.jobs[identifier].update(
                        result=result, state="cancelled" if event.is_set() else "done"
                    )
            except Exception as exc:
                logging.exception("Steering job failed")
                with self.guard:
                    self.jobs[identifier].update(state="error", error=str(exc))
            finally:
                engine.lock.release()

        self.worker.submit(run)
        return {"id": identifier}

    def job(self, identifier):
        with self.guard:
            if identifier not in self.jobs:
                raise KeyError("Unknown or expired job")
            return {k: v for k, v in self.jobs[identifier].items() if k != "cancel"}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, app, directory, **kwargs):
        self.app = app
        super().__init__(*args, directory=directory, **kwargs)

    def send_json(self, data, status=200):
        payload = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def tool_proxy(self, suffix, body=None):
        # Fixed loopback target, never a client-supplied URL. The CPU sidecar owns
        # MCP dependencies and isolated manual state, not this GPU process.
        if suffix not in ('tools', 'call') and not suffix.startswith('history?feature='):
            return self.send_json({"error": "Unknown tool endpoint"}, 404)
        request = Request("http://127.0.0.1:8006/" + suffix,
                          data=json.dumps(body).encode() if body is not None else None,
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=3600 if body is not None else 15) as response:
                return self.send_json(json.load(response), response.status)
        except HTTPError as exc:
            try: error = json.load(exc)
            except ValueError: error = {"error": "Tool service rejected the request"}
            return self.send_json(error, exc.code)
        except (URLError, TimeoutError, OSError) as exc:
            return self.send_json({"error": "Manual tool service unavailable: " + str(exc)}, 503)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/tools", "/tools/"):
            page = Path(__file__).with_name("molt_tool_playground.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return self.wfile.write(page)
        if path == "/api/tools/tools":
            return self.tool_proxy("tools")
        if path == "/api/tools/history":
            return self.tool_proxy("history?" + urlsplit(self.path).query)
        if path in ("/api/health", "/api/feature-stats"):
            try:
                engine = self.app.engine_for(parse_qs(urlsplit(self.path).query).get("layer", [None])[0])
            except ValueError as exc:
                return self.send_json({"error": str(exc)}, 400)
            if path == "/api/feature-stats":
                return self.send_json(engine.feature_stats)
            return self.send_json(
                {
                    "ready": True,
                    "signed_steering": True,
                    "busy": engine.lock.locked(),
                    "model": engine.meta["model"],
                    "layer": engine.layer,
                    "layers": sorted(self.app.engines),
                    "n_features": engine.meta["n_features"],
                    "global_step": engine.meta["global_step"],
                    "feature_stats_url": f"/api/feature-stats?layer={engine.layer}",
                }
            )
        if path.startswith("/api/jobs/"):
            try:
                return self.send_json(self.app.job(path.split("/")[-1]))
            except KeyError as exc:
                return self.send_json({"error": str(exc)}, 404)
        if path in ("/steer", "/steer/"):
            page = Path(__file__).with_name("molt_steering.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return self.wfile.write(page)
        if path.startswith("/api/"):
            return self.send_json({"error": "Unknown endpoint"}, 404)
        return super().do_GET()

    def do_POST(self):
        try:
            # Tailnet binding controls access. Reject cross-origin browser writes.
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                return self.send_json(
                    {"error": "Cross-origin requests are not accepted"}, 403
                )
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.send_json({"error": "Use application/json"}, 415)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 131072:
                return self.send_json({"error": "Invalid request size"}, 413)
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            path = urlsplit(self.path).path
            if path == "/api/tools/call":
                return self.tool_proxy("call", body)
            engine = self.app.engine_for(parse_qs(urlsplit(self.path).query).get("layer", [body.get("layer")])[0])
            if path == "/api/tokenize":
                return self.send_json(
                    engine.tokenize(body.get("prompt"), body.get("format", "chat"))
                )
            if path == "/api/inspect":
                if not engine.lock.acquire(blocking=False):
                    raise BlockingIOError(
                        "GPU is busy. Wait or cancel the current run."
                    )
                try:
                    result = engine.inspect(
                        body.get("prompt"),
                        body.get("format", "chat"),
                        body.get("feature"),
                    )
                finally:
                    engine.lock.release()
                return self.send_json(result)
            if path == "/api/generate":
                return self.send_json(self.app.submit(engine.validate(body), engine), 202)
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                identifier = path.split("/")[-2]
                with self.app.guard:
                    self.app.jobs[identifier]["cancel"].set()
                return self.send_json({"cancel_requested": True})
            return self.send_json({"error": "Unknown endpoint"}, 404)
        except BlockingIOError as exc:
            return self.send_json({"error": str(exc)}, 409)
        except (ValueError, TypeError) as exc:
            return self.send_json({"error": str(exc)}, 400)
        except KeyError as exc:
            return self.send_json({"error": str(exc)}, 404)
        except Exception as exc:
            logging.exception("API request failed")
            return self.send_json({"error": str(exc)}, 500)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", default="results/molt-qwen-dashboard")
    p.add_argument("--steering-directory", action="append", default=[],
                   help="Additional collected layer to steer using the shared Qwen model")
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--gpu-memory-gib", type=float, help="Cap this process's CUDA allocator"
    )
    args = p.parse_args()
    if args.gpu_memory_gib is not None and (
        not math.isfinite(args.gpu_memory_gib) or args.gpu_memory_gib <= 0
    ):
        p.error("gpu-memory-gib must be finite and positive")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    if device.type == "cuda" and args.gpu_memory_gib is not None:
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, args.gpu_memory_gib * 1024**3 / total), device
        )
    print("Loading Qwen and MOLT steering weights…", flush=True)
    engine = SteeringEngine(args.directory, args.device)
    additional = [SteeringEngine(directory, args.device, shared_engine=engine)
                  for directory in args.steering_directory]
    app = Application(engine, additional)
    server = ThreadingHTTPServer(
        (args.bind, args.port), partial(Handler, app=app, directory=args.directory)
    )
    print(f"Ready: http://{args.bind}:{args.port}/steer", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with app.guard:
            for job in app.jobs.values():
                job["cancel"].set()
        server.server_close()
        app.worker.shutdown(wait=True)


if __name__ == "__main__":
    main()
