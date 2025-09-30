import modal
from pathlib import Path
import logging

volume = modal.Volume.from_name("clt-checkpoints", create_if_missing=True)
cache_volume = modal.Volume.from_name("cache", create_if_missing=True)

cache_volume_path = Path("/hf")
volume_path = Path("/experiments")
HF_HOME_PATH = cache_volume_path
ACTIVATIONS_PATH = volume_path / "activations"
CHECKPOINTS_PATH = volume_path / "checkpoints"
WANDB_PATH = volume_path / "wandb"

volumes = {volume_path: volume, cache_volume_path: cache_volume}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.8.0",
        "nnsight==0.5.0.dev7",
        "datasets==3.6.0",
        "h5py>=3.13.0",
        "einops>=0.8.1",
        "jaxtyping>=0.3.2",
        "lightning>=2.5.1",
        "wandb>=0.19.11",
        "transformers>=4.46.0",
        "numpy>=1.24.0",
        "jsonargparse[signatures]>=4.27.7",
        "huggingface_hub[hf_transfer]",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": HF_HOME_PATH.as_posix()})
    .add_local_python_source("crosslayer_transcoder", "tests")
    .add_local_dir("config", "/config")
)

wandb_secret = modal.Secret.from_name("wandb-secret")

retries = modal.Retries(initial_delay=0.0, max_retries=3)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = modal.App("clt-trainer", image=image)


def volume_commit(volume):
    def inner(func):
        def wrapper(*args, **kwargs):
            logger.info("Committing volume")
            result = func(*args, **kwargs)
            volume.commit()
            logger.info("Volume committed")
            return result

        return wrapper

    return inner
