import time
from typing import Tuple

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
from model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)


class CrossLayerTranscoderModule(L.LightningModule):
    def __init__(
        self,
        # Model architecture parameters
        d_acts: int = 768,
        d_features: int = 6144,
        n_layers: int = 12,
        # Nonlinearity parameters
        nonlinearity_theta: float = 0.03,
        nonlinearity_bandwidth: float = 1.0,
        # Standardizer parameters
        activation_dim: int = 768,
        # Training parameters
        lambda_sparsity: float = 0.0002,
        c_sparsity: float = 0.1,
        learning_rate: float = 1e-3,
        compile: bool = False,
        # Replacement model parameters
        replacement_model_name: str = "openai-community/gpt2",
        replacement_model_device_map: str = "cuda:0",
        replacement_model_loader_batch_size: int = 5,
        # Dead features
        compute_dead_features: bool = False,
        compute_dead_features_every: int = 100,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        # Store model parameters
        self.d_acts = d_acts
        self.d_features = d_features
        self.n_layers = n_layers
        self.nonlinearity_theta = nonlinearity_theta
        self.nonlinearity_bandwidth = nonlinearity_bandwidth
        self.activation_dim = activation_dim

        # Store training parameters
        self._lambda = lambda_sparsity
        self.c = c_sparsity
        self.learning_rate = learning_rate
        self.compile = compile

        # Replacement model
        self.replacement_model_name = replacement_model_name
        self.replacement_model_device_map = replacement_model_device_map
        self.replacement_model_loader_batch_size = replacement_model_loader_batch_size
        self.replacement_model = None

        # Dead features
        self.compute_dead_features = compute_dead_features
        self.compute_dead_features_every = compute_dead_features_every

        # Model will be constructed in configure_model
        self.model = None

    def configure_model(self):
        if self.model:
            return
        # Construct nonlinearity
        nonlinearity = JumpReLU(
            theta=self.nonlinearity_theta,
            bandwidth=self.nonlinearity_bandwidth,
            n_layers=self.n_layers,
            d_features=self.d_features,
        )

        # Construct standardizers
        input_standardizer = DimensionwiseInputStandardizer(
            n_layers=self.n_layers, activation_dim=self.activation_dim
        )

        output_standardizer = DimensionwiseOutputStandardizer(
            n_layers=self.n_layers, activation_dim=self.activation_dim
        )

        # Construct the main model
        self.model = CrossLayerTranscoder(
            nonlinearity=nonlinearity,
            input_standardizer=input_standardizer,
            output_standardizer=output_standardizer,
            d_acts=self.d_acts,
            d_features=self.d_features,
            n_layers=self.n_layers,
        )

        self.replacement_model = ReplacementModelAccuracy(
            model_name=self.replacement_model_name,
            device_map=self.replacement_model_device_map,
            loader_batch_size=self.replacement_model_loader_batch_size,
        )

        if self.compute_dead_features:
            self.dead_features = DeadFeatures(
                n_features=self.d_features,
                n_layers=self.n_layers,
                return_per_layer=True,
            )

        if self.compile:
            print("Compiling model")
            self = torch.compile(self)

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

        loss = mse + sparsity

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

        self.dead_features.update(features)

        if (
            self.compute_dead_features
            and self.global_step % self.compute_dead_features_every == 0
        ):
            dead_features_per_layer = self.dead_features.compute()
            dead_features_per_layer = dead_features_per_layer.detach().cpu()
            for i in range(self.n_layers):
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
