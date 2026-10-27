#!/usr/bin/env python3
"""Wait for a specific training process, validate completion, then launch once."""

import argparse
import fcntl
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time


def process_identity(pid):
    """Linux process start time protects against PID reuse; zombies are finished."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return None if fields[0] == "Z" else fields[19]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    args = parser.parse_args()
    state_path = args.state.resolve()
    with state_path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text())
        if state["phase"] != "waiting":
            raise RuntimeError("Queue has already advanced; refusing a duplicate launch")

        def update(phase, **extra):
            state.update(phase=phase, updated_at=time.time(), **extra)
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(state_path)
            print(phase, extra, flush=True)

        try:
            os.chdir(state["repo"])
            update("waiting", watcher_pid=os.getpid())
            while any(
                process_identity(pid) == identity
                for pid, identity in state["predecessor_processes"].items()
            ):
                time.sleep(30)

            update("checking_predecessor")
            import torch

            checkpoint = torch.load(
                state["predecessor_checkpoint"],
                mmap=True,
                map_location="cpu",
                weights_only=False,
            )
            step = checkpoint.get("global_step", -1)
            del checkpoint
            if step < state["required_steps"]:
                raise RuntimeError(f"Predecessor stopped at step {step}; new run not launched")
            if Path(state["new_checkpoint"]).exists():
                raise RuntimeError("New checkpoint already exists; refusing to overwrite it")

            update("checking_tokenization", predecessor_final_step=step)
            subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "tests/test_generation_tokenization.py"],
                check=True,
            )
            # Re-parse immediately before launch, without allocating the model.
            subprocess.run(
                [".venv/bin/clt", "fit", "--config", state["config"], "--print_config"],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            update("starting")
            # The config path is passed as an argument, never interpolated into shell code.
            command = '''set -euo pipefail
set -a
source .env
set +a
exec .venv/bin/clt fit --config "$1"
'''
            with Path(state["training_log"]).open("x") as log:
                process = subprocess.Popen(
                    ["bash", "-c", command, "queued-molt", state["config"]],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={**os.environ, **state.get("environment", {})},
                )
                update("running", training_pid=process.pid, started_at=time.time())
                # A failed trainer may hang joining its still-running data
                # generator. Detect the fatal OOM instead of reporting success
                # merely because the parent process is still alive.
                while process.poll() is None:
                    with Path(state["training_log"]).open("rb") as progress:
                        progress.seek(max(0, progress.seek(0, 2) - 65536))
                        tail = progress.read()
                    if b"torch.OutOfMemoryError:" in tail:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise RuntimeError("Training ran out of GPU memory; stopped its process group")
                    time.sleep(10)
                returncode = process.wait()
            if returncode:
                raise RuntimeError(f"Queued training exited with status {returncode}")
            checkpoint = torch.load(
                state["new_checkpoint"], mmap=True, map_location="cpu", weights_only=False
            )
            step = checkpoint.get("global_step", -1)
            if step < state["required_steps"]:
                raise RuntimeError(f"Queued training stopped at step {step}")
            update("complete", final_step=step)
        except Exception as error:
            update("failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
