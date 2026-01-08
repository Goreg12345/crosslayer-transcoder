#!/usr/bin/env python3
"""
Script to upload checkpoint to HuggingFace Hub
"""
import os

import dotenv
from huggingface_hub import HfApi

dotenv.load_dotenv()

# Initialize HuggingFace API with token from environment
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload the checkpoints folder
api.upload_folder(
    folder_path="/var/metrics/g/crosslayer-transcoder/crosslayer_transcoder/checkpoints",
    repo_id="georglange/gpt2-clt-topk-16-f-10k",
    repo_type="model",
)

print("✓ Successfully uploaded checkpoint to HuggingFace!")
