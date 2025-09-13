from pathlib import Path
import logging
from typing import Optional

from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.utils.buffer import DiscBuffer
import modal
import torch as t

volume = modal.Volume.from_name("clt-checkpoints", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").pip_install(
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
)

image = image.add_local_python_source("crosslayer_transcoder")


app = modal.App("clt-train", image=image)

wandb_secret = modal.Secret.from_name("wandb-secret")


volume_path = Path("/experiments")
ACTIVATIONS_PATH = volume_path / "activations"
CHECKPOINTS_PATH = volume_path / "checkpoints"
WANDB_PATH = volume_path / "wandb"

volumes = {volume_path: volume}

retries = modal.Retries(initial_delay=0.0, max_retries=10)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_directories():
    ACTIVATIONS_PATH.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)
    WANDB_PATH.mkdir(parents=True, exist_ok=True)


@app.function(
    volumes=volumes,
    gpu="a10g",
    retries=retries,
    max_inputs=1,
    timeout=60 * 60 * 24,
    secrets=[wandb_secret],
)
def train_interruptible(*args, **kwargs):
    train(*args, **kwargs)


def train(experiment):
    ensure_directories()
    experiment_dir = CHECKPOINTS_PATH / experiment
    last_checkpoint = experiment_dir / "last.ckpt"

    if last_checkpoint.exists():
        print(f"⚡️ resuming training from the latest checkpoint: {last_checkpoint}")
        train_model(
            ACTIVATIONS_PATH,
            experiment_dir,
            resume_from_checkpoint=last_checkpoint,
        )
        print("⚡️ training finished successfully")
    else:
        print("⚡️ starting training from scratch")
        train_model(ACTIVATIONS_PATH, experiment_dir)


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


@volume_commit(volume)
def get_checkpoint(checkpoint_dir):
    from lightning.pytorch.callbacks import ModelCheckpoint

    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
        every_n_epochs=10,
        filename="{epoch:02d}",
    )


def train_model(data_dir, checkpoint_dir, resume_from_checkpoint=None):
    import lightning as L
    from lightning.pytorch.loggers import WandbLogger

    model = get_model()
    train_loader = get_train_loader(
        data_dir=data_dir, file_name="openai-community_gpt2.h5", accessor="activations"
    )
    checkpoint_callback = get_checkpoint(checkpoint_dir)
    wandb_logger = WandbLogger(project="clt-train-modal")

    logger.info("Compiling model")
    # model = t.compile(model)
    logger.info("Model compiled")

    trainer = L.Trainer(
        max_steps=100_000,
        val_check_interval=1_000,
        limit_val_batches=1,
        check_val_every_n_epoch=None,
        callbacks=[checkpoint_callback],
        precision="16-true",
        accelerator="gpu",
        devices=[0],
        logger=wandb_logger,
        # strategy="ddp",
        # callbacks=[TBProfilerCallback()],
        accumulate_grad_batches=1,
    )

    if resume_from_checkpoint is not None:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            ckpt_path=resume_from_checkpoint,
        )
    else:
        logger.info("Training model from scratch")
        trainer.fit(model, train_loader)


def get_model(checkpoint_path=None):
    from crosslayer_transcoder.model.clt_lightning import (
        JumpReLUCrossLayerTranscoderModule,
    )
    from crosslayer_transcoder.model.clt import CrossLayerTranscoder
    from crosslayer_transcoder.model.clt import Encoder
    from crosslayer_transcoder.model.clt import CrosslayerDecoder
    from crosslayer_transcoder.model.standardize import DimensionwiseInputStandardizer
    from crosslayer_transcoder.model.standardize import DimensionwiseOutputStandardizer
    from crosslayer_transcoder.metrics.replacement_model_accuracy import (
        ReplacementModelAccuracy,
    )
    from crosslayer_transcoder.metrics.dead_features import DeadFeatures

    encoder = Encoder(
        d_acts=768,
        d_features=10_000,
        n_layers=12,
    )
    decoder = CrosslayerDecoder(
        d_acts=768,
        d_features=10_000,
        n_layers=12,
    )
    replacement_model = ReplacementModelAccuracy(
        model_name="openai-community/gpt2",
        device_map="cuda:0",
        loader_batch_size=2,
    )

    dead_features = DeadFeatures(
        n_features=10_000,
        n_layers=12,
        return_per_layer=True,
        return_log_freqs=True,
        return_neuron_indices=True,
    )

    clt = CrossLayerTranscoder(
        encoder=encoder,
        decoder=decoder,
        input_standardizer=DimensionwiseInputStandardizer(
            n_layers=12,
            activation_dim=768,
        ),
        output_standardizer=DimensionwiseOutputStandardizer(
            n_layers=12,
            activation_dim=768,
        ),
        nonlinearity=JumpReLU(
            theta=0.03, bandwidth=0.01, n_layers=12, d_features=10_000
        ),
    )

    return JumpReLUCrossLayerTranscoderModule(
        model=clt,
        replacement_model=replacement_model,
        dead_features=dead_features,
        learning_rate=1e-4,
        compile=True,
        lr_decay_step=80_000,
        lr_decay_factor=0.1,
        lambda_sparsity=0.0007,
        c_sparsity=1,
        use_tanh=True,
        pre_actv_loss=1e-6,
        compute_dead_features=True,
        compute_dead_features_every=500,
    )


def get_train_loader(data_dir, file_name="clt-activations-10M.h5", accessor="tensor"):
    print("⚡ setting up data")
    buffer = DiscBuffer(data_dir / file_name, accessor)
    train_loader = t.utils.data.DataLoader(
        buffer,
        num_workers=20,
        prefetch_factor=2,
        batch_size=4000,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
    )
    return train_loader


@app.local_entrypoint()
def main(experiment: Optional[str] = None):
    if experiment is None:
        from uuid import uuid4

        experiment = uuid4().hex[:8]
    print(f"⚡️ starting interruptible training experiment {experiment}")
    train_interruptible.spawn(experiment).get()
