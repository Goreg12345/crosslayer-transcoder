import time
from typing import Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float

import wandb
from metrics.dead_features import DeadFeatures
from metrics.replacement_model_accuracy import ReplacementModelAccuracy
from model.clt import CrossLayerTranscoder
from model.jumprelu import JumpReLU


class CrossLayerTranscoderModule(L.LightningModule):
    def __init__(
        self,
        # Pre-constructed modules
        model: CrossLayerTranscoder,
        replacement_model: Optional[ReplacementModelAccuracy] = None,
        dead_features: Optional[DeadFeatures] = None,
        # Training parameters
        lambda_sparsity: float = 0.0002,
        c_sparsity: float = 0.1,
        learning_rate: float = 1e-3,
        compile: bool = False,
        # Dead features computation settings
        compute_dead_features: bool = False,
        compute_dead_features_every: int = 100,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(
            ignore=["model", "replacement_model", "dead_features"]
        )

        # Store pre-constructed modules
        self.model = model
        self.replacement_model = replacement_model
        self.dead_features = dead_features

        # Store training parameters
        self._lambda = lambda_sparsity
        self.c = c_sparsity
        self.learning_rate = learning_rate
        self.compile = compile

        # Dead features computation settings
        self.compute_dead_features = compute_dead_features
        self.compute_dead_features_every = compute_dead_features_every

    def configure_model(self):
        # Apply compilation if requested
        if self.compile:
            print("Compiling model")
            self.model = torch.compile(self.model)

    def forward(
        self, acts_norm: Float[torch.Tensor, "batch_size n_layers d_acts"]
    ) -> Tuple[
        Float[torch.Tensor, "batch_size n_layers d_features"],
        Float[torch.Tensor, "batch_size n_layers d_acts"],
    ]:
        return self.model(acts_norm)

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.model.initialize_standardizers(batch)

        resid, mlp_out = batch[:, 0], batch[:, 1]
        features, recons_norm, recons = self.forward(resid)

        mse = (
            (recons_norm - self.model.output_standardizer.standardize(mlp_out)) ** 2
        ).mean()

        masked_w = einsum(
            self.model.W_dec,
            self.model.mask,
            "from_layer to_layer d_features d_acts, from_layer to_layer -> from_layer to_layer d_features d_acts",
        )
        l1 = masked_w.norm(p=2, dim=[1, 3])
        tanh = torch.tanh(features * l1 * self.c)
        sparsity = self._lambda * tanh.sum(dim=[1, 2]).mean()

        loss = mse
        if isinstance(self.model.nonlinearity, JumpReLU):
            loss += sparsity
        # TopK does not need sparsity loss

        self.log("train_loss", loss)
        self.log("train_mse", mse)
        self.log("train_mse_rescaled", ((recons - mlp_out) ** 2).mean())
        self.log("train_sparsity", sparsity)
        self.log("L0 (%)", 100 * (features > 0).float().mean())
        self.log("recons_standardized_std", recons_norm.std())
        self.log(
            "L0 (Avg. per layer)",
            (features > 0).float().sum() / (features.shape[0] * features.shape[1]),
        )

        if batch_idx % 500 == 1:
            l0_per_layer = (features > 0).float().sum(dim=(0, 2)) / features.shape[0]

            if self.logger and isinstance(self.logger.experiment, wandb.wandb_run.Run):
                table = wandb.Table(
                    data=[[i, v.item()] for i, v in enumerate(l0_per_layer.cpu())],
                    columns=["layer", "L0"],
                )
                self.logger.experiment.log(
                    {
                        "L0 per layer": wandb.plot.bar(
                            table, "layer", "L0", title="L0 per Layer"
                        )
                    },
                    step=self.global_step,
                )

        if self.dead_features is not None:
            self.dead_features.update(features)

        if (
            self.compute_dead_features
            and self.dead_features is not None
            and self.global_step % self.compute_dead_features_every == 0
        ):
            dead_features_per_layer = self.dead_features.compute()
            dead_features_per_layer = dead_features_per_layer.detach().cpu()
            for i in range(dead_features_per_layer.shape[0]):
                self.log(f"dead_features_layer_{i}", dead_features_per_layer[i])
            self.dead_features.reset()

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        if self.replacement_model is not None:
            self.replacement_model.update(self.model)
            self.log("replacement_model_accuracy", self.replacement_model.compute())
        print("exiting val epoch end")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
