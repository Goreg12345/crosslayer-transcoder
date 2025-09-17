import logging
from typing import Optional

from crosslayer_transcoder.utils.buffer import DiscBuffer
import torch as t

from crosslayer_transcoder.modal.app import (
    app,
    volume_commit,
    volumes,
    retries,
    wandb_secret,
    volume,
    ACTIVATIONS_PATH,
    CHECKPOINTS_PATH,
    WANDB_PATH,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_directories():
    ACTIVATIONS_PATH.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)
    WANDB_PATH.mkdir(parents=True, exist_ok=True)


@app.function(
    volumes=volumes,
    gpu="A100-40GB",
    # retries=retries,
    max_inputs=1,
    timeout=60 * 60 * 24,
    secrets=[wandb_secret],
)
def train_interruptible(*args, **kwargs):
    train(*args, **kwargs, volume=volume)


def train(experiment, volume=None):
    ensure_directories()

    experiment_dir = CHECKPOINTS_PATH / experiment
    last_checkpoint = experiment_dir / "last.ckpt"

    if last_checkpoint.exists():
        print(f"⚡️ resuming training from the latest checkpoint: {last_checkpoint}")
        train_model(
            ACTIVATIONS_PATH,
            experiment_dir,
            resume_from_checkpoint=last_checkpoint,
            volume=volume,
        )
        print("⚡️ training finished successfully")
    else:
        print("⚡️ starting training from scratch")
        train_model(ACTIVATIONS_PATH, experiment_dir)


@volume_commit(volume)
def get_checkpoint(checkpoint_dir):
    from lightning.pytorch.callbacks import ModelCheckpoint

    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_last=True,
        every_n_epochs=10,
        filename="{epoch:02d}",
    )


def train_model(data_dir, checkpoint_dir, resume_from_checkpoint=None, volume=None):
    import lightning as L
    from lightning.pytorch.loggers import WandbLogger

    L.seed_everything(42)

    model = get_model()
    checkpoint_callback = get_checkpoint(checkpoint_dir)
    wandb_logger = WandbLogger(project="clt-train-modal")

    # logger.info("Compiling model")
    # model = t.compile(model)
    # logger.info("Model compiled")

    trainer = L.Trainer(
        max_steps=100_000,
        val_check_interval=1_000,
        limit_val_batches=1,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
        precision="16-mixed",
        accelerator="gpu",
        devices=[0],
        logger=wandb_logger,
        # strategy="ddp",
        # callbacks=[TBProfilerCallback()],
        accumulate_grad_batches=1,
        # gradient_clip_val=0.5,
        # gradient_clip_algorithm="norm",
    )

    datamodule = get_datamodule()

    if resume_from_checkpoint is not None:
        trainer.fit(
            model=model,
            # train_dataloaders=train_loader,
            ckpt_path=resume_from_checkpoint,
            datamodule=datamodule,
        )
    else:
        logger.info("Training model from scratch")
        trainer.fit(model, datamodule=datamodule)


def get_datamodule():
    from crosslayer_transcoder.data.datamodule import ActivationDataModule

    return ActivationDataModule(
        # Buffer settings
        buffer_size=1_000_000,
        n_in_out=2,
        n_layers=12,
        activation_dim=768,
        dtype="float16",
        max_batch_size=50000,
        # Model settings for activation generation
        model_name="openai-community/gpt2",
        model_dtype="float16",
        # Dataset settings
        dataset_name="Skylion007/openwebtext",
        dataset_split="train",
        max_sequence_length=1024,
        # Generation settings
        generation_batch_size=10,
        refresh_interval=0.1,
        # Memory settings
        shared_memory_name="activation_buffer",
        timeout_seconds=30,
        # File paths
        init_file=str(ACTIVATIONS_PATH / "openai-community_gpt2.h5"),
        accessor="activations",
        # DataLoader settings
        batch_size=1000,
        num_workers=10,
        prefetch_factor=2,
        shuffle=True,
        persistent_workers=True,
        pin_memory=True,
        minimum_fill_threshold=0.2,
        use_shared_memory=False,
        # Device configuration
        device_map="cuda:0",
        deployment_policy="gpu_only",
        # WandB logging configuration for data generation
        wandb_logging={
            "enabled": True,
            "project": "clt",
            "group": None,
            "run_name": "data-generator-jumprelu",
            "tags": ["data-generation"],
            "save_dir": "./wandb",
            "log_interval": 1.0,
        },
    )


@volume_commit(volume)
def get_model(
    checkpoint_path=None,
):
    from crosslayer_transcoder.model.clt_lightning import (
        JumpReLUCrossLayerTranscoderModule,
    )
    from crosslayer_transcoder.model.clt import CrossLayerTranscoder
    from crosslayer_transcoder.model.jumprelu import JumpReLU
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

    input_standardizer = DimensionwiseInputStandardizer(
        n_layers=12,
        activation_dim=768,
    )
    output_standardizer = DimensionwiseOutputStandardizer(
        n_layers=12,
        activation_dim=768,
    )
    nonlinearity = JumpReLU(theta=0.03, bandwidth=0.01, n_layers=12, d_features=10_000)

    clt = CrossLayerTranscoder(
        encoder=encoder,
        decoder=decoder,
        input_standardizer=input_standardizer,
        output_standardizer=output_standardizer,
        nonlinearity=nonlinearity,
    )

    return JumpReLUCrossLayerTranscoderModule(
        model=clt,
        replacement_model=replacement_model,
        dead_features=dead_features,
        learning_rate=1e-4,
        compile=False,
        lr_decay_step=80_000,
        lr_decay_factor=0.1,
        lambda_sparsity=0.0007,
        c_sparsity=1,
        use_tanh=True,
        pre_actv_loss=1e-6,
        compute_dead_features=True,
        compute_dead_features_every=500,
    )


@volume_commit(volume)
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
