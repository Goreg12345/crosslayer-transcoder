import os

# select cuda 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["WANDB_DIR"] = f"{os.getcwd()}/wandb"
os.environ["WANDB_CACHE_DIR"] = f"{os.getcwd()}/wandb_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
import logging

import lightning.pytorch as L
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import ModelParallelStrategy
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)

from buffer import DiscBuffer
from clt_opt import CrossLayerTranscoder
from jumprelu import JumpReLU

# Check if CUDA is available
from utils import print_gpu_info

# print_gpu_info()


# Enable DTensor dispatch debug logging
logging.basicConfig(level=logging.DEBUG)
dtensor_logger = logging.getLogger("torch.distributed.tensor._dispatch")
dtensor_logger.setLevel(logging.DEBUG)

# Or more specifically, just the dispatch module
logging.getLogger("torch.distributed.tensor._dispatch").setLevel(logging.DEBUG)


buffer = DiscBuffer("/var/local/glang/activations/clt-activations-10M.h5", "tensor")

loader = torch.utils.data.DataLoader(
    buffer,
    num_workers=4,
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
            on_trace_ready=tensorboard_trace_handler("log/ddp_new"),
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
    callbacks=[TBProfilerCallback()],
    # accumulate_grad_batches=4,
)

import contextlib

from torch.utils._python_dispatch import TorchDispatchMode


@contextlib.contextmanager
def debug_autograd_shapes():
    old = torch.is_anomaly_enabled()
    torch.autograd.set_detect_anomaly(True)
    try:
        yield
    finally:
        torch.autograd.set_detect_anomaly(old)


class ShowBackward(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        print("BACKWARD:", func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))


# with ShowBackward(), debug_autograd_shapes():

trainer.fit(
    model=clt,
    train_dataloaders=loader,
    # val_dataloaders=loader,
)

# Save checkpoint after training
checkpoint_path = "checkpoints/clt.ckpt"
trainer.save_checkpoint(checkpoint_path)
