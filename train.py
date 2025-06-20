import os

# select cuda 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WANDB_DIR"] = f"{os.getcwd()}/wandb"
os.environ["WANDB_CACHE_DIR"] = f"{os.getcwd()}/wandb_cache"
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
from lightning.pytorch.strategies import ModelParallelStrategy

from buffer import DiscBuffer
from clt_opt import CrossLayerTranscoder
from jumprelu import JumpReLU

buffer = DiscBuffer("/var/local/glang/activations/clt-activations-10M.h5", "tensor")

loader = torch.utils.data.DataLoader(
    buffer,
    num_workers=20,
    prefetch_factor=2,
    batch_size=1000,
    shuffle=False,
    persistent_workers=True,
    pin_memory=True,
)


# logger = WandbLogger(project="wandb_clt")
logger = None

clt = CrossLayerTranscoder(
    config={
        "d_acts": 768,
        "d_features": 768 * 8,
        "n_layers": 12,
        "lambda": 0.0002,
        "c": 0.1,
        "lr": 1e-3,
    },
    nonlinearity=JumpReLU(theta=0.03, bandwidth=1.0, n_layers=12, d_features=768 * 8),
)
# clt = torch.compile(clt)

print("good afternoon")

torch.autograd.set_detect_anomaly(True)


class TBProfilerCallback(L.Callback):
    def on_train_start(self, trainer, *_):
        self.prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=4, warmup=4, active=16),
            on_trace_ready=tensorboard_trace_handler("log/ddp"),
            record_shapes=True,
        )
        self.prof.__enter__()

    def on_train_batch_end(self, trainer, *_):
        self.prof.step()  # one step per batch

    def on_train_end(self, trainer, *_):
        self.prof.__exit__(None, None, None)


strategy = ModelParallelStrategy()
precision = "32-true"

trainer = L.Trainer(
    logger=logger,
    max_steps=2000,
    limit_train_batches=2000,
    val_check_interval=1000,
    limit_val_batches=1,
    check_val_every_n_epoch=None,
    enable_checkpointing=False,
    precision=precision,
    accelerator="gpu",
    devices=2,
    strategy=strategy,
    # callbacks=[TBProfilerCallback()],
    accumulate_grad_batches=4,
)

trainer.fit(
    model=clt,
    train_dataloaders=loader,
    # val_dataloaders=loader,
)

# Save checkpoint after training
checkpoint_path = "checkpoints/clt.ckpt"
trainer.save_checkpoint(checkpoint_path)
