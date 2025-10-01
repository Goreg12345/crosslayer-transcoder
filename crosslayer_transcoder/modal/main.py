import os
import sys
from crosslayer_transcoder.modal.app import app, volumes, wandb_secret, retries
from crosslayer_transcoder.main import main as lightning_cli_main


@app.function(
    gpu="A100-40GB",
    volumes=volumes,
    secrets=[wandb_secret],
    timeout=60 * 60 * 24,
    retries=retries,
)
def train(*args, **kwargs):
    # TODO: try resuming from checkpoint

    # NOTE: clear sysargv to allow command-line arg parsing from lightning
    if len(sys.argv) > 1:
        sys.argv[1:] = []

    lightning_cli_main(*args)


@app.local_entrypoint()
def main(*args):
    train.remote(*args)
