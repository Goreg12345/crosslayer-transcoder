import os

# select cuda 0
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

if cuda_available:
    # Number of GPUs
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs:", num_gpus)

    # List each device’s name
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {name}")
else:
    print("No CUDA devices found")


import lightning.pytorch as L
from lightning.pytorch.loggers import WandbLogger

from buffer import DiscBuffer
from clt import CrossLayerTranscoder
from jumprelu import JumpReLU

buffer = DiscBuffer("/var/local/glang/activations/clt-activations-10M.h5", "tensor")

loader = torch.utils.data.DataLoader(
    buffer,
    num_workers=20,
    prefetch_factor=10,
    batch_size=1000,
    shuffle=True,
)


logger = WandbLogger(project="wandb_clt")
trainer = L.Trainer(
    logger=logger,
    max_steps=2000,
    limit_train_batches=2000,
    val_check_interval=100,
    limit_val_batches=1,
    check_val_every_n_epoch=None,
)
trainer.fit(
    model=CrossLayerTranscoder(
        config={
            "d_acts": 768,
            "d_features": 768 * 8,
            "n_layers": 12,
            "lambda": 0.0002,
            "c": 0.1,
            "lr": 1e-3,
        },
        nonlinearity=JumpReLU(
            theta=0.03, bandwidth=1.0, n_layers=12, d_features=768 * 8
        ),
    ),
    train_dataloaders=loader,
    val_dataloaders=loader,
)

# Save checkpoint after training
checkpoint_path = "checkpoints/clt.ckpt"
trainer.save_checkpoint(checkpoint_path)
