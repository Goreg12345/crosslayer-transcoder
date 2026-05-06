"""
Standalone activation producer.

Runs an activation generator in its own process, completely outside Lightning's
DDP launch tree. The buffer it creates is then attached to by N consumer ranks
that run Lightning DDP with `producer_mode: "client"` in the data section of the
training config — see `config/molt-multilayer-gemma3-4b-it-1_plus_3ddp.yaml`.

Lifecycle:
  1. Parse the data section of a Lightning CLI YAML config (same schema as the
     `data:` block consumed by `clt fit`).
  2. Allocate the shared buffer in /dev/shm under the configured
     `shared_memory_name` (no rank suffix; consumers attach by exactly this
     name).
  3. Spawn the existing `DataGeneratorProcess` to fill it on the configured
     `device_map` (e.g. `cuda:0`).
  4. Block on SIGINT/SIGTERM so the buffer + producer stay alive while
     consumers are training.
  5. On shutdown, terminate the generator and unlink the shared memory.

Usage:
    python -m crosslayer_transcoder.data.standalone_producer \
        --config config/molt-multilayer-gemma3-4b-it-1_plus_3ddp.yaml

Trainers in another process (or via `scripts/train_molt_gemma3_1plus3.sh`)
should be started AFTER the standalone producer reports "buffer ready".
"""

import argparse
import logging
import signal
import sys
import time
from typing import Any, Dict

import torch
import torch.multiprocessing as mp
import yaml

from crosslayer_transcoder.data.data_generator import DataGeneratorProcess
from crosslayer_transcoder.data.deployment_policy import DeploymentPolicy
from crosslayer_transcoder.data.shared_memory import SharedActivationBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [producer] %(levelname)s %(message)s")
logger = logging.getLogger("standalone_producer")


def _load_data_section(config_path: str) -> Dict[str, Any]:
    """Pull init_args out of the `data:` block of a Lightning CLI yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "data" not in cfg:
        raise ValueError(f"{config_path}: no 'data' section")
    data = cfg["data"]
    init_args = data.get("init_args", data)
    return init_args


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone activation buffer producer")
    parser.add_argument("--config", required=True, help="Path to Lightning CLI YAML config")
    args = parser.parse_args()

    # Spawn-style child processes are required because the data generator owns
    # CUDA state; fork()ing a CUDA-using process is undefined behavior.
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    init = _load_data_section(args.config)

    # `producer_mode` in the config governs the trainer ranks ("self" = each rank
    # spawns its own producer; "client" = ranks attach to an external producer).
    # The standalone producer is the external producer, so it ignores that field
    # and always plays the producer role.
    logger.info("trainer-side producer_mode in this config: %s", init.get("producer_mode", "self"))

    dtype = getattr(torch, init.get("dtype", "float32"))
    model_dtype = getattr(torch, init.get("model_dtype", "float32"))
    deployment_policy = DeploymentPolicy.from_string(init.get("deployment_policy", "gpu_only"))

    shm_name = init.get("shared_memory_name", "activation_buffer")
    logger.info("Creating shared activation buffer (name=%s)", shm_name)
    shared_buffer = SharedActivationBuffer(
        buffer_size=init["buffer_size"],
        n_in_out=init["n_in_out"],
        n_layers=init["n_layers"],
        activation_dim=init["activation_dim"],
        dtype=dtype,
        shared_memory_name=shm_name,
        timeout_seconds=init.get("timeout_seconds", 30),
        generation_batch_size=init["generation_batch_size"],
        max_sequence_length=init["max_sequence_length"],
        minimum_fill_threshold=init.get("minimum_fill_threshold", 0.0),
        batch_size=init.get("batch_size"),
        create=True,
    )

    wandb_cfg = dict(init.get("wandb_logging") or {})
    if wandb_cfg.get("enabled"):
        base = wandb_cfg.get("run_name") or "data-generator"
        wandb_cfg["run_name"] = f"{base}-producer"

    generator = DataGeneratorProcess(
        shared_buffer=shared_buffer,
        buffer_size=init["buffer_size"],
        n_in_out=init["n_in_out"],
        n_layers=init["n_layers"],
        activation_dim=init["activation_dim"],
        dtype=dtype,
        max_batch_size=init.get("max_batch_size", 50_000),
        model_name=init["model_name"],
        model_dtype=model_dtype,
        dataset_name=init["dataset_name"],
        dataset_split=init.get("dataset_split", "train"),
        max_sequence_length=init["max_sequence_length"],
        generation_batch_size=init["generation_batch_size"],
        refresh_interval=init.get("refresh_interval", 0.1),
        deployment_policy=deployment_policy,
        init_file=init.get("init_file"),
        device_map=init.get("device_map", "cuda:0"),
        wandb_logging=wandb_cfg,
        model_arch=init.get("model_arch", "gpt2"),
    )

    logger.info("Starting generator process on device_map=%s ...", init.get("device_map", "cuda:0"))
    generator.start()
    logger.info(
        "buffer ready — name=%s validity=%s_validity. Consumers can now attach.",
        shm_name,
        shm_name,
    )

    stop = {"flag": False}

    def _handle(signum, frame):
        if not stop["flag"]:
            logger.info("received signal %s, shutting down", signum)
            stop["flag"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    try:
        while not stop["flag"]:
            if not generator.is_alive():
                logger.error("generator process died; exiting")
                return 1
            time.sleep(1.0)
    finally:
        logger.info("terminating generator")
        if generator.is_alive():
            generator.terminate()
            generator.join(timeout=10.0)
            if generator.is_alive():
                logger.warning("force-killing generator")
                generator.kill()
                generator.join()
        try:
            shared_buffer.cleanup()
        except Exception as e:
            logger.warning("buffer cleanup raised: %s", e)
        logger.info("producer shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
