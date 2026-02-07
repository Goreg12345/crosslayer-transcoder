#!/usr/bin/env python3
"""
Upload a local weights directory (config.yaml + checkpoint.safetensors) to the Hugging Face Hub.

Usage:
    python scripts/upload_to_huggingface.py local/weights/topk-16k-control --repo-id YOUR_USERNAME/crosslayer-transcoder-topk-16k

Requires: pip install huggingface_hub (or use project env; transformers brings it in).
Login: huggingface-cli login  (or set HF_TOKEN).
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser(description="Upload local weights to Hugging Face Hub")
    parser.add_argument(
        "local_dir",
        type=Path,
        help="Local directory containing config.yaml and checkpoint.safetensors (e.g. local/weights/topk-16k-control)",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repo id, e.g. username/crosslayer-transcoder-topk-16k",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload model weights",
        help="Commit message for the upload",
    )
    args = parser.parse_args()

    local_dir = args.local_dir.resolve()
    if not local_dir.is_dir():
        raise SystemExit(f"Not a directory: {local_dir}")

    for name in ("config.yaml", "checkpoint.safetensors"):
        if not (local_dir / name).exists():
            raise SystemExit(f"Missing {name} in {local_dir}")

    api = HfApi()
    create_repo(args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )
    print(f"Uploaded {local_dir} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
