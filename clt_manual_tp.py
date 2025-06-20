from typing import Tuple

import lightning as L
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import einsum
from jaxtyping import Float


class CrossLayerTranscoderManualTP(L.LightningModule):
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

        # Get distributed info
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # Calculate sharded dimensions
        self.d_features_per_device = d_features // self.world_size

        # Create sharded parameters
        self.W_enc = nn.Parameter(
            torch.empty(n_layers, d_acts, self.d_features_per_device)
        )
        self.W_dec = nn.Parameter(
            torch.empty(n_layers, n_layers, self.d_features_per_device, d_acts)
        )

        # the mask ensures that features can only contribute to reconstructions of same or later layers
        self.register_buffer("mask", torch.triu(torch.ones(n_layers, n_layers)))

        # initialize parameters
        self.reset_parameters()

        print(f"Rank {self.rank}: W_enc shape = {self.W_enc.shape}")
        print(f"Rank {self.rank}: W_dec shape = {self.W_dec.shape}")

    def reset_parameters(self):
        enc_uniform_thresh = 1 / (self.d_features_per_device**0.5)
        self.W_enc.data.uniform_(-enc_uniform_thresh, enc_uniform_thresh)

        dec_uniform_thresh = 1 / (
            (self.config.get("d_acts", 768) * self.config.get("n_layers", 12)) ** 0.5
        )
        self.W_dec.data.uniform_(-dec_uniform_thresh, dec_uniform_thresh)

    def forward(
        self, acts_norm: Float[torch.Tensor, "batch_size n_layers d_acts"]
    ) -> Tuple[
        Float[torch.Tensor, "batch_size n_layers d_features_shard"],
        Float[torch.Tensor, "batch_size n_layers d_acts"],
    ]:
        # Forward pass with sharded W_enc (ColwiseParallel equivalent)
        features_shard = einsum(
            acts_norm,
            self.W_enc,
            "batch_size n_layers d_acts, n_layers d_acts d_features_shard -> batch_size n_layers d_features_shard",
        )

        print(f"Rank {self.rank}: features_shard.shape = {features_shard.shape}")

        features_shard = self.nonlinearity(features_shard)

        # Reconstruction with sharded W_dec (RowwiseParallel equivalent)
        recons_shard = einsum(
            features_shard,
            self.W_dec,
            self.mask,
            "batch_size from_layer d_features_shard, from_layer to_layer d_features_shard d_acts, "
            "from_layer to_layer -> batch_size to_layer d_acts",
        )

        # AllReduce to get final reconstruction (sum across devices)
        if self.world_size > 1:
            dist.all_reduce(recons_shard, op=dist.ReduceOp.SUM)

        return features_shard, recons_shard

    def training_step(self, batch, batch_idx):
        mean = batch.mean(dim=-1, keepdim=True)
        std = batch.std(dim=-1, keepdim=True)
        acts_norm = (batch - mean) / std
        resid, mlp_out = acts_norm[:, 0], batch[:, 1]

        features_shard, recons = self.forward(resid)

        # MSE
        mse = ((recons - mlp_out) ** 2).mean()

        # Sparsity (computed on local shard)
        masked_w = einsum(
            self.W_dec,
            self.mask,
            "from_layer to_layer d_features_shard d_acts, from_layer to_layer -> from_layer to_layer d_features_shard d_acts",
        )
        l1 = masked_w.norm(p=1, dim=[1, 3])  # l1: n_layers x d_features_shard
        tanh = torch.tanh(features_shard * l1 * self.c)
        sparsity = self._lambda * tanh.sum(dim=[1, 2]).mean()  # mean over batch

        loss = mse + sparsity

        self.log("train_loss", loss)
        self.log("train_mse", mse)
        self.log("train_sparsity", sparsity)
        self.log("L0 (%)", 100 * (features_shard > 0).float().mean())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr", 1e-3))
        return optimizer
