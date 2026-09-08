#!/usr/bin/env python3
"""Serve the feature dashboard and a single-GPU token-steering API."""

import argparse
import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import torch

from crosslayer_transcoder.dashboard.steering import SteeringEngine


class Application:
    def __init__(self, engine):
        self.engine = engine
        self.jobs = {}
        self.guard = threading.RLock()
        self.worker = ThreadPoolExecutor(max_workers=1)

    def submit(self, config):
        if not self.engine.lock.acquire(blocking=False):
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

                result = self.engine.compare(config, event, progress)
                with self.guard:
                    self.jobs[identifier].update(
                        result=result, state="cancelled" if event.is_set() else "done"
                    )
            except Exception as exc:
                logging.exception("Steering job failed")
                with self.guard:
                    self.jobs[identifier].update(state="error", error=str(exc))
            finally:
                self.engine.lock.release()

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

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/health":
            return self.send_json(
                {
                    "ready": True,
                    "busy": self.app.engine.lock.locked(),
                    "model": self.app.engine.meta["model"],
                    "layer": self.app.engine.layer,
                    "n_features": self.app.engine.meta["n_features"],
                    "global_step": self.app.engine.meta["global_step"],
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
            engine = self.app.engine
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
                return self.send_json(self.app.submit(engine.validate(body)), 202)
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
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    print("Loading Qwen and MOLT steering weights…", flush=True)
    engine = SteeringEngine(args.directory, args.device)
    app = Application(engine)
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
