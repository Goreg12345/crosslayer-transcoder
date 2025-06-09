from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float

import wandb
from metrics.replacement_model_accuracy import ReplacementModelAccuracy


class CrossLayerTranscoder(L.LightningModule):
    def __init__(self, config, nonlinearity, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(config)

        self.config = config
        d_acts = config.get("d_acts", 768)
        d_features = config.get("d_features", 768 * 8)
        n_layers = config.get("n_layers", 12)

        # loss hyperparams:
        self._lambda = config.get("lambda", 0.1)
        self.c = config.get("c", 0.1)

        self.nonlinearity = nonlinearity

        self.W_enc = nn.Parameter(torch.empty(n_layers, d_acts, d_features))
        self.W_dec = nn.Parameter(torch.empty(n_layers, n_layers, d_features, d_acts))
        # the mask ensures that features can only contribute to reconstructions of same or later layers
        self.register_buffer("mask", torch.triu(torch.ones(n_layers, n_layers)))

        self.replacement_model = ReplacementModelAccuracy()

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        enc_uniform_thresh = 1 / (self.config.get("d_features", 768 * 8) ** 0.5)
        self.W_enc.data.uniform_(-enc_uniform_thresh, enc_uniform_thresh)

        dec_uniform_thresh = 1 / (
            (self.config.get("d_acts", 768) * self.config.get("n_layers", 12)) ** 0.5
        )
        self.W_dec.data.uniform_(-dec_uniform_thresh, dec_uniform_thresh)

    def forward(
        self, acts_norm: Float[torch.Tensor, "batch_size n_layers d_acts"]
    ) -> Tuple[
        Float[torch.Tensor, "batch_size n_layers d_features"],
        Float[torch.Tensor, "batch_size n_layers d_acts"],
    ]:
        # NORMALIZE activations of each layer because different layers have different ranges

        features = einsum(
            acts_norm,
            self.W_enc,
            "batch_size n_layers d_acts, n_layers d_acts d_features -> batch_size n_layers d_features",
        )

        features = self.nonlinearity(features)
        recons = einsum(
            features,
            self.W_dec,
            self.mask,
            "batch_size from_layer d_features, from_layer to_layer d_features d_acts, "
            "from_layer to_layer -> batch_size to_layer d_acts",
        )

        return features, recons

    def training_step(self, batch, batch_idx):
        mean = batch.mean(dim=-1, keepdim=True)
        std = batch.std(dim=-1, keepdim=True)
        acts_norm = (batch - mean) / std
        resid, mlp_out = acts_norm[:, 0], batch[:, 1]
        # mlp_out = resid.clone()
        features, recons = self.forward(resid)

        # MSE
        mse = ((recons - mlp_out) ** 2).mean()

        # Sparsity
        # W_dec: nlayers x dfeatures x dacts
        # features: batch_size x nlayers x dfeatures
        masked_w = einsum(
            self.W_dec,
            self.mask,
            "from_layer to_layer d_features d_acts, from_layer to_layer -> from_layer to_layer d_features d_acts",
        )
        l1 = masked_w.norm(p=1, dim=[1, 3])  # l1: n_layers x d_features
        tanh = torch.tanh(features * l1 * self.c)
        sparsity = self._lambda * tanh.sum(dim=[1, 2]).mean()  # mean over batch

        loss = mse + sparsity

        self.log("train_loss", loss)
        self.log("train_mse", mse)
        self.log("train_sparsity", sparsity)
        self.log("L0 (%)", 100 * (features > 0).float().mean())
        self.log(
            "L0 (Avg. per layer)",
            (features > 0).float().sum() / (features.shape[0] * features.shape[1]),
        )

        if batch_idx % 100 == 0:
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
        # we just need this here such that on_validation_epoch_end is called
        pass

    def on_validation_epoch_end(self):
        self.replacement_model.update(self)
        self.log("replacement_model_accuracy", self.replacement_model.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr", 1e-3))
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
        return optimizer
