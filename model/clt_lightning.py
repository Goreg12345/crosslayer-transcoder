import gc
import os
import subprocess
import time
from typing import Optional, Tuple

import lightning as L
import psutil
import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float

import wandb
from metrics.dead_features import DeadFeatures
from metrics.replacement_model_accuracy import ReplacementModelAccuracy
from model.clt import CrosslayerDecoder, CrossLayerTranscoder, Decoder
from model.jumprelu import JumpReLU
from model.topk import BatchTopK


def find_gpu_tensors_with_module_context():
    """Find GPU tensors with module context and names"""
    tensors = []
    total_memory = 0

    # Build comprehensive tensor mapping
    tensor_name_map = {}
    module_map = {}

    # Get all loaded modules
    for obj in gc.get_objects():
        if isinstance(obj, torch.nn.Module):
            module_name = obj.__class__.__name__

            # Get parameter names with module context
            for name, param in obj.named_parameters():
                if param.is_cuda:
                    tensor_name_map[id(param)] = f"{module_name}.{name}"
                    module_map[id(param)] = obj

            # Get buffer names with module context
            for name, buffer in obj.named_buffers():
                if buffer.is_cuda:
                    tensor_name_map[id(buffer)] = f"{module_name}.{name}"
                    module_map[id(buffer)] = obj

    # Find all GPU tensors
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            size = obj.element_size() * obj.numel()
            tensor_id = id(obj)

            # Determine tensor name and category
            if tensor_id in tensor_name_map:
                tensor_name = tensor_name_map[tensor_id]
                category = "model"
            elif hasattr(obj, "grad_fn") and obj.grad_fn is not None:
                tensor_name = f"intermediate:{obj.grad_fn.__class__.__name__}"
                category = "computation"
            elif obj.requires_grad:
                tensor_name = "gradient"
                category = "gradient"
            else:
                tensor_name = "temporary"
                category = "temporary"

            tensors.append(
                {
                    "name": tensor_name,
                    "category": category,
                    "shape": tuple(obj.shape),
                    "dtype": obj.dtype,
                    "size_mb": size / (1024**2),
                    "device": obj.device,
                    "requires_grad": obj.requires_grad,
                    "tensor": obj,
                }
            )
            total_memory += size

    # Sort by memory usage (largest first)
    tensors.sort(key=lambda x: x["size_mb"], reverse=True)

    print(f"Total GPU tensors: {len(tensors)}")
    print(f"Total memory: {total_memory / (1024**2):.2f} MB")

    # Group by category
    categories = {}
    for tensor in tensors:
        cat = tensor["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(tensor)

    print("\nBy Category:")
    for cat, tensor_list in categories.items():
        total_cat_memory = sum(t["size_mb"] for t in tensor_list)
        print(f"  {cat}: {len(tensor_list)} tensors, {total_cat_memory:.2f} MB")

    print("\nTop 1000 largest tensors:")
    for i, tensor_info in enumerate(tensors[:1000]):
        print(
            f"{i+1}. [{tensor_info['category']}] {tensor_info['name']}, "
            f"Shape: {tensor_info['shape']}, "
            f"Size: {tensor_info['size_mb']:.2f} MB"
        )

    return tensors


def comprehensive_memory_debug():
    """Get complete GPU memory picture for the correct device"""
    print("=== GPU Memory Debug ===")

    # Get current device
    current_device = torch.cuda.current_device()
    print(f"Current CUDA device: {current_device}")

    # PyTorch's view for current device
    allocated = torch.cuda.memory_allocated(current_device) / 1024**3
    reserved = torch.cuda.memory_reserved(current_device) / 1024**3
    cached = reserved - allocated

    print(f"PyTorch Allocated (cuda:{current_device}): {allocated:.2f} GB")
    print(f"PyTorch Reserved (cuda:{current_device}): {reserved:.2f} GB")
    print(f"PyTorch Cached (cuda:{current_device}): {cached:.2f} GB")

    # Check ALL GPUs with nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"\nAll GPU memory usage:")
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.split(", ")
                    if len(parts) >= 3:
                        gpu_id, used_mb, total_mb = (
                            parts[0],
                            int(parts[1]),
                            int(parts[2]),
                        )
                        highlight = f" *** PyTorch is here ***" if int(gpu_id) == current_device else ""
                        print(
                            f"  GPU {gpu_id}: {used_mb / 1024:.2f} GB / {total_mb / 1024:.2f} GB{highlight}"
                        )
    except Exception as e:
        print(f"nvidia-smi error: {e}")

    # Check processes on ALL GPUs
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,nounits,noheader",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            print(f"\nAll GPU processes:")
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.split(", ")
                    if len(parts) >= 4:
                        gpu_uuid, pid, name, memory = (
                            parts[0],
                            parts[1],
                            parts[2],
                            parts[3],
                        )
                        current_pid = os.getpid()
                        highlight = "*** THIS PROCESS ***" if int(pid) == current_pid else ""
                        print(f"  GPU {gpu_uuid[:8]}...: PID {pid} ({name}) using {memory} MB {highlight}")
    except Exception as e:
        print(f"Error checking GPU processes: {e}")


class CrossLayerTranscoderModule(L.LightningModule):
    def __init__(
        self,
        # Pre-constructed modules
        model: CrossLayerTranscoder,
        replacement_model: Optional[ReplacementModelAccuracy] = None,
        dead_features: Optional[DeadFeatures] = None,
        # Training parameters
        learning_rate: float = 1e-3,
        compile: bool = False,
        # Learning rate schedule parameters
        warmup_steps: int = 0,
        lr_decay_step: int = 0,
        lr_decay_factor: float = 1.0,
        # Dead features computation settings
        compute_dead_features: bool = False,
        compute_dead_features_every: int = 100,
        tokens_till_dead: int = 1_000_000,
        # optimizer parameters
        optimizer: str = "adam",
        beta1: Optional[float] = None,
        beta2: Optional[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(ignore=["model", "replacement_model", "dead_features"])
        # torch.cuda.memory._record_memory_history(max_entries=100_000)

        # Store pre-constructed modules
        self.model = model
        self.replacement_model = replacement_model
        self.dead_features = dead_features

        # Store training parameters
        self.learning_rate = learning_rate
        self.compile = compile
        self.tokens_till_dead = tokens_till_dead

        # Learning rate schedule parameters
        self.warmup_steps = warmup_steps
        self.lr_decay_step = lr_decay_step
        self.lr_decay_factor = lr_decay_factor

        # Dead features computation settings
        self.compute_dead_features = compute_dead_features
        self.compute_dead_features_every = compute_dead_features_every

        # optimizer parameters
        self.optimizer = optimizer
        self.beta1 = beta1
        self.beta2 = beta2

        assert (
            self.model.encoder.n_layers == self.model.decoder.n_layers
        ), "Encoder and decoder must have the same number of layers"

        self.register_buffer(
            "n_lifetime_active",
            torch.zeros((self.model.encoder.n_layers, self.model.encoder.d_features)),
        )
        self.register_buffer(
            "last_active",
            torch.zeros((self.model.encoder.n_layers, self.model.encoder.d_features), dtype=torch.long),
        )
        self.register_buffer(
            "prev_dead_features",
            torch.zeros((self.model.encoder.n_layers, self.model.encoder.d_features), dtype=torch.bool),
        )

    def configure_model(self):
        # Apply compilation if requested
        if self.compile:
            print("Compiling model")
            self = torch.compile(self)

    def forward(self, acts_norm: Float[torch.Tensor, "batch_size n_layers d_acts"]) -> Tuple[
        Float[torch.Tensor, "batch_size n_layers d_features"],  # pre_actvs
        Float[torch.Tensor, "batch_size n_layers d_features"],  # features
        Float[torch.Tensor, "batch_size n_layers d_acts"],  # recons_norm
        Float[torch.Tensor, "batch_size n_layers d_acts"],  # recons
    ]:
        return self.model(acts_norm)

    def log_training_metrics(self, features, recons_norm, recons, mlp_out, batch_idx):
        # Compute MSE and related metrics
        mlp_out_norm = self.model.output_standardizer.standardize(mlp_out)
        mse = (recons_norm - mlp_out_norm) ** 2

        ss_err = (mlp_out_norm - recons_norm) ** 2
        ss_err = ss_err.sum(dim=0)
        ss_total = ((mlp_out_norm - mlp_out_norm.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
        fvu = (ss_err / ss_total).mean()  # (n_layers, d_model)
        self.log("metrics/fraction_of_variance_unexplained", fvu)
        fvu_per_layer = (ss_err / ss_total).mean(dim=-1)
        assert fvu_per_layer.shape == (self.model.encoder.n_layers,)
        for layer in range(self.model.encoder.n_layers):
            self.log(f"layers/fraction_of_variance_unexplained/layer_{layer}", fvu_per_layer[layer])

        # Log n_lifetime_active
        self.log("metrics/n_lifetime_active", (self.n_lifetime_active > 0).float().mean())

        # Log encoder/decoder direction magnitudes
        lens = self.model.encoder.W.norm(p=2, dim=1)  # (n_layers, d_features)
        lens_active = lens[self.n_lifetime_active > 100]
        self.log("model/encoder_direction_magnitude", lens_active.mean())
        lens_inactive = lens[self.n_lifetime_active < 100]
        self.log("model/encoder_direction_magnitude_inactive", lens_inactive.mean())
        if isinstance(self.model.decoder, CrosslayerDecoder):
            lens_dec = [
                self.model.decoder.get_parameter(f"W_{l}").norm(p=2, dim=-1).flatten()
                for l in range(self.model.decoder.n_layers)
            ]
            lens_dec = torch.concat(lens_dec)
        elif isinstance(self.model.decoder, Decoder):
            lens_dec = self.model.decoder.W.norm(p=2, dim=-1).flatten()
        self.log("model/decoder_direction_magnitude", lens_dec.mean())

        # Log MSE per layer
        mse_per_layer = mse.mean(dim=(0, 2))
        for layer in range(self.model.decoder.n_layers):
            self.log(f"layers/train_mse/layer_{layer}", mse_per_layer[layer])

        # Log MSE metrics
        self.log("metrics/train_mse", mse.mean())
        self.log("metrics/train_mse_rescaled", ((recons - mlp_out) ** 2).mean())

        # Log L0 metrics
        self.log("metrics/L0_percent", 100 * (features > 0).float().mean())
        self.log("metrics/recons_standardized_std", recons_norm.std())
        self.log(
            "metrics/L0_avg_per_layer",
            (features > 0).float().sum() / (features.shape[0] * features.shape[1]),
        )

        # Log current learning rate
        if self.lr_schedulers():
            self.log(
                "training/learning_rate",
                self.trainer.optimizers[0].param_groups[0]["lr"],
            )

        # Log L0 table per layer
        if batch_idx % 500 == 1:
            l0_per_layer = (features > 0).float().sum(dim=(0, 2)) / features.shape[0]

            if self.logger and isinstance(self.logger.experiment, wandb.wandb_run.Run):
                table = wandb.Table(
                    data=[[i, v.item()] for i, v in enumerate(l0_per_layer.cpu())],
                    columns=["layer", "L0"],
                )
                self.logger.experiment.log(
                    {"layers/L0_per_layer": wandb.plot.bar(table, "layer", "L0", title="L0 per Layer")},
                    step=self.global_step,
                )

        # Compute and log dead features
        if self.compute_dead_features and self.dead_features is not None:
            self.dead_features.update(features)

        if (
            self.compute_dead_features
            and self.dead_features is not None
            and self.global_step % self.compute_dead_features_every == 0
        ):
            features_stats = self.dead_features.compute()
            dead_total = features_stats["mean"]
            self.log("metrics/dead_features_mean", dead_total)
            if "per_layer" in features_stats:
                dead_per_layer = features_stats["per_layer"]
                for i in range(dead_per_layer.shape[0]):
                    self.log(f"layers/dead_features/layer_{i}", dead_per_layer[i])
            if "log_freqs" in features_stats:
                dead_log_freqs = features_stats["log_freqs"]
                # Log histogram for vertical color visualization
                if self.logger and (
                    isinstance(self.logger.experiment, wandb.wandb_run.Run)
                    or (isinstance(self.logger, L.pytorch.loggers.WandbLogger))
                ):
                    for layer in range(dead_log_freqs.shape[0]):
                        self.logger.experiment.log(
                            {f"layers/log_feature_density/layer_{layer}": dead_log_freqs[layer]},
                            step=self.global_step,
                        )
                    self.logger.experiment.log(
                        {f"training/log_feature_density": dead_log_freqs.flatten()},
                        step=self.global_step,
                    )
                self.log("training/log_feature_density_mean", dead_log_freqs.mean())

            if "layer_indices" in features_stats:
                layer_indices = features_stats["layer_indices"]
                self.prev_dead_features[layer_indices] = True
                ## here

            self.dead_features.reset()

        # Update last_active and compute dead features
        self.last_active[features.detach().sum(dim=0) > 0.0] = 0.0
        self.last_active += features.shape[0]
        idxs_dead = self.last_active > self.tokens_till_dead
        self.log("metrics/dead_features", idxs_dead.float().mean())

        return idxs_dead

    def training_step(self, batch, batch_idx):
        # torch.cuda.empty_cache()
        # gc.collect()
        batch = torch.stack([batch[:, 0], batch[:, 0]], dim=1)  # TODO: remove this

        # TODO: does this prevent complete compilation?
        if batch_idx == 0:
            self.model.initialize_standardizers(batch)

        resid, mlp_out = batch[:, 0], batch[:, 1]
        _, features, recons_norm, recons = self.forward(resid)

        self.n_lifetime_active += (features > 0).sum(0).float()

        mse = (recons_norm - self.model.output_standardizer.standardize(mlp_out)) ** 2
        loss = mse.mean()

        self.log("training/loss", loss)

        # Log training metrics
        self.log_training_metrics(features, recons_norm, recons, mlp_out, batch_idx)

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        torch.cuda.empty_cache()
        gc.collect()
        if self.replacement_model is not None:
            with torch.no_grad():
                self.replacement_model.update(self.model)
                r_acc, r_kl = self.replacement_model.compute()
                self.log("validation/replacement_model_accuracy", r_acc)
                self.log("validation/replacement_model_kl", r_kl)
                self.replacement_model.reset()
        print("exiting val epoch end")
        gc.collect()
        # comprehensive_memory_debug()
        torch.cuda.empty_cache()
        # exit()

    def configure_optimizers(self):
        if self.beta1 is None:
            self.beta1 = 0.9
        if self.beta2 is None:
            self.beta2 = 0.999

        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2)
            )
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2)
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported")

        # Use warmup scheduler if requested
        if self.warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-7,  # Start from very small LR
                end_factor=1.0,  # End at full LR
                total_iters=self.warmup_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": warmup_scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "warmup",
                },
            }

        # Use step decay scheduler if requested (single step down)
        if self.lr_decay_step > 0:
            step_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[self.lr_decay_step],
                gamma=self.lr_decay_factor,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": step_scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "step_decay",
                },
            }

        # No scheduler
        return optimizer


class JumpReLUCrossLayerTranscoderModule(CrossLayerTranscoderModule):
    def __init__(
        self,
        lambda_sparsity: float = 0.0002,
        c_sparsity: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._lambda = lambda_sparsity
        self.c = c_sparsity

    def current_sparsity_penalty(self):
        n_steps = self.trainer.max_steps
        current_step = (
            self.global_step
        )  # use global step instead of batch idx to work with gradient accumulation
        cur_lambda = self._lambda * (current_step / n_steps)
        self.log("training/sparsity_penalty", cur_lambda)
        return cur_lambda

    def training_step(self, batch, batch_idx):
        # torch.cuda.empty_cache()
        # gc.collect()
        batch = torch.stack([batch[:, 0], batch[:, 0]], dim=1)  # TODO: remove this
        # Initialize standardizers
        if batch_idx == 0:
            self.model.initialize_standardizers(batch)

        # Forward pass
        resid, mlp_out = batch[:, 0], batch[:, 1]
        _, features, recons_norm, recons = self.forward(resid)

        self.n_lifetime_active += (features > 0).sum(0).float()

        # Compute MSE loss
        mse = (recons_norm - self.model.output_standardizer.standardize(mlp_out)) ** 2

        dec_norms = torch.zeros_like(features[:1])
        for l in range(self.model.decoder.n_layers):
            W = self.model.decoder.get_parameter(f"W_{l}")  # (from_layer, d_features, d_acts)
            dec_norms[:, : l + 1] = dec_norms[:, : l + 1] + W.norm(p=2, dim=-1)
        tanh = torch.tanh(features * dec_norms * self.c)
        sparsity = self.current_sparsity_penalty() * tanh.sum(dim=[1, 2]).mean()
        self.log("training/sparsity_loss", sparsity)

        loss = mse.mean() + sparsity
        self.log("training/loss", loss)

        # Log training metrics
        self.log_training_metrics(features, recons_norm, recons, mlp_out, batch_idx)

        return loss

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()

        # Log theta values from JumpReLU nonlinearity
        if hasattr(self.model.nonlinearity, "theta"):
            theta = self.model.nonlinearity.theta.squeeze(0)

            if self.logger and (
                isinstance(self.logger.experiment, wandb.wandb_run.Run)
                or (isinstance(self.logger, L.pytorch.loggers.WandbLogger))
            ):
                for layer in range(theta.shape[0]):
                    self.logger.experiment.log(
                        {f"layers/theta/layer_{layer}": theta[layer].cpu()},
                    )
                # Log combined theta values
                self.logger.experiment.log(
                    {"validation/theta": theta.flatten().cpu()},
                )


class TopKCrossLayerTranscoderModule(CrossLayerTranscoderModule):
    def __init__(
        self,
        topk_aux: torch.nn.Module,
        tokens_till_dead: int = 100_000,
        aux_loss_scale: float = 1 / 32,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokens_till_dead = tokens_till_dead
        self.topk_aux = topk_aux
        self.aux_loss_scale = aux_loss_scale
        self.register_buffer(
            "last_active",
            torch.zeros((self.model.encoder.n_layers, self.model.encoder.d_features), dtype=torch.long),
        )

    def training_step(self, batch, batch_idx):
        # torch.cuda.empty_cache()
        # gc.collect()

        if batch_idx == 0:
            self.model.initialize_standardizers(batch)

        resid, mlp_out = batch[:, 0], batch[:, 1]
        pre_actvs, features, recons_norm, recons = self.forward(resid)

        mse = (recons_norm - self.model.output_standardizer.standardize(mlp_out)) ** 2

        self.n_lifetime_active += (features > 0).sum(0).float()

        # Log training metrics using shared method
        idxs_dead = self.log_training_metrics(features, recons_norm, recons, mlp_out, batch_idx)

        # AUXILLIARY LOSS
        idxs_dead = idxs_dead.repeat(features.shape[0], 1, 1)
        dead_pre_actvs = torch.where(idxs_dead, pre_actvs, -torch.inf)
        dead_features = self.topk_aux(dead_pre_actvs)
        aux_recons = self.model.decoder(dead_features)
        recons_err = self.model.output_standardizer.standardize(mlp_out) - recons_norm
        aux_loss = (aux_recons - recons_err) ** 2

        aux_loss = torch.nan_to_num(aux_loss, 0.0)

        loss = mse.mean() + self.aux_loss_scale * aux_loss.mean()
        self.log("training/aux_loss", aux_loss.mean())
        self.log("training/loss", loss)

        # if batch_idx == 18:
        #     gpu_tensors = find_gpu_tensors_with_module_context()
        #     # print(gpu_tensors)
        #     exit()

        if batch_idx == 100_000_000_000:
            torch.cuda.memory._dump_snapshot("snapshot.pickle")

            torch.cuda.memory._record_memory_history(enabled=None)
            exit()
        return loss
