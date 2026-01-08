#!/usr/bin/env python3
"""Download checkpoint from HF and run validation."""

import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "georglange/gpt2-clt-topk-16-f-10k"
CONFIG = "config/topk-clt.yaml"


def download_if_missing(repo_id: str, filename: str, local_dir: str = "checkpoints"):
    local_path = Path(local_dir) / filename
    if local_path.exists():
        return str(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=local_dir, local_dir_use_symlinks=False
    )


def main():
    repo_id = sys.argv[1] if len(sys.argv) > 1 else REPO_ID
    config = sys.argv[2] if len(sys.argv) > 2 else CONFIG

    print(f"Downloading checkpoint from {repo_id}...")
    ckpt = download_if_missing(repo_id, "clt.ckpt")
    print(f"✓ Checkpoint: {ckpt}\n")

    print("Running validation...")
    subprocess.run(["clt", "validate", "--config", config, "--ckpt_path", ckpt, "--trainer.logger", "false"])


if __name__ == "__main__":
    main()

