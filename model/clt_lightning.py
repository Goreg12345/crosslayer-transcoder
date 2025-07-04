from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float

import wandb
from metrics.replacement_model_accuracy import ReplacementModelAccuracy


class CrossLayerTranscoderModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lambda_sparsity: float = 0.0002,
        c_sparsity: float = 0.1,
        learning_rate: float = 1e-3,
        replacement_model_accuracy: ReplacementModelAccuracy = None,
        compile: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self._lambda = lambda_sparsity
        self.c = c_sparsity
        self.learning_rate = learning_rate
        self.replacement_model = replacement_model_accuracy
        self.compile = compile

    def configure_model(self):
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
