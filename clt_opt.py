# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any, Optional, Tuple, Union, cast

import lightning as L
import torch
import torch.nn as nn
from einops import einsum, rearrange
from jaxtyping import Float
from torch.distributed._tensor.placement_types import Partial, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (DeviceMesh, DTensor, Replicate, Shard,
                                      distribute_module, distribute_tensor)
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema  # OpSpec,
from torch.distributed.tensor._op_schema import (OpStrategy, PlacementStrategy,
                                                 RuntimeSchemaInfo,
                                                 StrategyType, _is_inplace_op)
from torch.distributed.tensor._ops.utils import (is_tensor_dim_sharded,
                                                 normalize_dim,
                                                 register_op_strategy)
from torch.distributed.tensor.parallel import (RowwiseParallel,
                                               parallelize_module)
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.distributed.tensor.placement_types import (Placement, Replicate,
                                                      Shard)
from torch.nn.modules.activation import ReLU

import wandb
from jumprelu import JumpReLU
from metrics.replacement_model_accuracy import ReplacementModelAccuracy

aten = torch.ops.aten  # convenience alias


@register_op_strategy([aten.select.int])
def select_int_strategy(op_schema):
    """
    args_schema layout in 2.7.1:
        (input_strategy: OpStrategy, selected_dim: int, index: int)

    Returns one PlacementStrategy per input layout.
    """

    # unpack schema -----------------------------------------------------------
    input_strategy: OpStrategy = op_schema.args_schema[0]
    selected_dim: int = op_schema.args_schema[1]

    # handle negative dims once
    ndim = input_strategy.shape.__len__()
    selected_dim = selected_dim if selected_dim >= 0 else selected_dim + ndim

    out_strategy = OpStrategy([])

    # iterate over every existing placement configuration ---------------------
    for plan in input_strategy.strategies:
        in_spec: DTensorSpec = plan.output_spec
        new_placements = []

        for placement in in_spec.placements:
            if placement.is_shard():
                shard_dim = cast(Shard, placement).dim

                if shard_dim == selected_dim:
                    raise NotImplementedError(
                        "Selecting along a dimension that is sharded "
                        "is not supported on PyTorch-2.7.1."
                    )
                # shift shard index left if we dropped an earlier dimension
                if shard_dim > selected_dim:
                    shard_dim -= 1
                placement = Shard(shard_dim)

            # Replicate or Partial stay as-is
            new_placements.append(placement)

        out_spec = DTensorSpec(mesh=in_spec.mesh, placements=tuple(new_placements))

        out_strategy.strategies.append(
            PlacementStrategy(output_specs=out_spec, input_specs=(in_spec,))
        )

    return out_strategy


@register_op_strategy([aten.masked_fill_.Scalar])
def masked_fill_scalar_strategy(op_schema):
    """
    Works on PyTorch 2.7.1 DTensor.
    args_schema = (data_strategy: OpStrategy,
                   mask_strategy: OpStrategy,
                   value: float)
    """
    data_strat, mask_strat = op_schema.args_schema[:2]
    assert isinstance(data_strat, OpStrategy) and isinstance(mask_strat, OpStrategy)

    out_strat = OpStrategy([])

    for d_plan in data_strat.strategies:
        d_spec: DTensorSpec = d_plan.output_spec

        # find compatible mask specs
        compatibles = []
        for m_plan in mask_strat.strategies:
            m_spec: DTensorSpec = m_plan.output_spec
            if not m_spec.is_sharded():
                compatibles.append(m_spec)  # replicated mask always fine
            elif m_spec.placements == d_spec.placements:
                compatibles.append(m_spec)  # identical sharding dim + partition

        if not compatibles:
            raise NotImplementedError("mask sharding not compatible with data")

        # one placement per compatible mask, no cost field
        for m_spec in compatibles:
            out_strat.strategies.append(
                PlacementStrategy(
                    output_specs=d_spec,
                    input_specs=(d_spec, m_spec),  # first two args matter
                )
            )
    return out_strat


"""
@register_op_strategy([aten.slice_backward.default])
def slice_backward_strategy(op_schema):
    grad_strat, dim = op_schema.args_schema[0], op_schema.args_schema[2]
    dim %= len(grad_strat.shape)  # handle negative dims

    out = OpStrategy([])

    for plan in grad_strat.strategies:
        spec = plan.output_spec

        new_placements = [
            Replicate() if p.is_shard() and p.dim == dim else p for p in spec.placements
        ]
        if Replicate() in new_placements:
            print(
                "Warning: using replicate in slice_backward_strategy, could lead to all_gather"
            )
        new_spec = DTensorSpec(mesh=spec.mesh, placements=tuple(new_placements))

        out.strategies.append(
            PlacementStrategy(
                output_specs=new_spec,
                input_specs=(spec,),  # only grad_output carries shards
            )
        )
    print("leaving slice_backward_strategy")
    return out
"""


@register_op_strategy(
    aten.slice_backward.default,
)
def slice_backward_rules(op_schema):
    # func: slice_backward(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt start, SymInt end, SymInt step) -> Tensor
    args_schema = op_schema.args_schema
    input_strategy, dim = args_schema[0], args_schema[2]
    assert isinstance(input_strategy, OpStrategy), f"{input_strategy}"
    output_strategies = []
    for placement_strategy in input_strategy.strategies:
        output_spec = placement_strategy.output_spec
        new_placements: list[Placement] = []
        for placement in output_spec.placements:
            # Redistribute to replicate only if the dim is sharded and matches the slice dim
            if isinstance(placement, Shard) and placement.dim == dim:
                new_placements.append(Replicate())
                print("using replicate in slice_backward_rules")
            else:
                new_placements.append(placement)
        new_spec = DTensorSpec(output_spec.mesh, tuple(new_placements))
        new_strategy = PlacementStrategy(output_specs=new_spec)
        output_strategies.append(new_strategy)
    return OpStrategy(output_strategies)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_acts = config.get("d_acts", 768)
        d_features = config.get("d_features", 768 * 8)
        n_layers = config.get("n_layers", 12)
        self.W = nn.Parameter(
            torch.empty(
                (n_layers, d_acts, d_features),
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        enc_uniform_thresh = 1 / (self.config.get("d_acts", 768) ** 0.5)
        self.W.data.uniform_(-enc_uniform_thresh, enc_uniform_thresh)

    def forward(
        self, acts_norm: Float[torch.Tensor, "batch_size n_layers d_acts"]
    ) -> Float[torch.Tensor, "batch_size n_layers d_features"]:
        return einsum(
            acts_norm,
            self.W,
            "batch_size n_layers d_acts, n_layers d_acts d_features -> batch_size n_layers d_features",
        )


def debug_backward_hook(name):
    def hook(grad):
        print(f"BACKWARD HOOK [{name}]: grad.shape={grad.shape}")
        if hasattr(grad, "placements"):
            print(f"  grad placements: {grad.placements}")
        return grad

    return hook


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_acts = config.get("d_acts", 768)
        d_features = config.get("d_features", 768 * 8)
        n_layers = config.get("n_layers", 12)
        self.c = config.get("c", 0.1)
        self._lambda = config.get("lambda", 0.1)
        for i in range(n_layers):
            self.register_parameter(
                f"W_{i}", nn.Parameter(torch.empty((i + 1, d_features, d_acts)))
            )
        self.reset_parameters()

    def reset_parameters(self):
        dec_uniform_thresh = 1 / (
            (self.config.get("d_features", 768 * 8) * self.config.get("n_layers", 12))
            ** 0.5
        )
        for i in range(self.config.get("n_layers", 12)):
            self.get_parameter(f"W_{i}").data.uniform_(
                -dec_uniform_thresh, dec_uniform_thresh
            )

    def forward(
        self, features: Float[torch.Tensor, "batch_size n_layers d_features"]
    ) -> Float[torch.Tensor, "batch_size n_layers d_acts"]:
        recons = []
        dec_norms = torch.zeros_like(features[:1])  # 1 n_layers d_features
        features.register_hook(debug_backward_hook("features"))
        for l in range(self.config.get("n_layers", 12)):
            W = self.get_parameter(f"W_{l}")
            W.register_hook(debug_backward_hook(f"W_{l}"))
            selected_features = features[:, : l + 1]
            selected_features.register_hook(
                debug_backward_hook(f"selected_features_{l}")
            )
            # this slicing will call slice_backward() in autograd and this will force an all gather for some reason
            # but features should be Shard(2) so
            l_recons = einsum(
                selected_features,
                W,
                "batch_size n_layers d_features, n_layers d_features d_acts -> batch_size n_layers d_acts",
            )
            l_recons.register_hook(debug_backward_hook(f"l_recons_{l}"))
            l_recons = l_recons.sum(dim=1)
            l_recons.register_hook(debug_backward_hook(f"l_recons_sum_{l}"))
            recons.append(l_recons)
            # dec_norms_slice = dec_norms[:, : l + 1]
            ##dec_norms_slice.register_hook(debug_backward_hook(f"dec_norms_slice_{l}"))
            # norms = W.norm(p=2, dim=-1)
            # norms.register_hook(debug_backward_hook(f"norms_{l}"))
            # norms_slice_sum = dec_norms_slice + norms
            # norms_slice_sum.register_hook(debug_backward_hook(f"norms_slice_sum_{l}"))
            # dec_norms[:, : l + 1] = norms_slice_sum
            # dec_norms.register_hook(debug_backward_hook(f"dec_norms_{l}"))
        # recons = rearrange(
        #    recons, "n_layers batch_size d_acts -> batch_size n_layers d_acts"
        # )  # stack + transpose
        recons = torch.stack(recons, dim=1)
        recons.register_hook(debug_backward_hook("recons"))
        # recons = torch.zeros_like(torch.stack(recons, dim=1))

        # Sparsity -> l1 norm of l2 norms -> like in Anthropic's first implementation
        # Alternative: l2 norm of concatenated decoder vectors (not implemented here)
        # features: [batch_size, n_layers, d_features]
        # decoder.W_i: [i+1, d_features, d_acts]

        tanh = torch.tanh(features * dec_norms * self.c)
        sparsity = self._lambda * tanh.sum(dim=[1]).mean(dim=0)  # mean over batch
        sparsity = sparsity.sum()  # sum over the sharded layer last to reduce comms
        return recons, sparsity


class ColParallelEncoder(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Replicate(),)
        self.output_layouts = (output_layouts or Shard(-1),)
        # colwise linear runtime sharding (desired sharding):
        # 1. requires replicate input
        # 2. shard output on last dim
        self.desired_input_layouts = (Replicate(),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts, run_check=False
            )

        # transform the input layouts to the desired layouts of ColwiseParallel
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    def _partition_fn(self, name, module, device_mesh):
        module.register_parameter(
            "W",
            nn.Parameter(
                distribute_tensor(
                    module.W, device_mesh, [Shard(-1)], src_data_rank=self.src_data_rank
                )
            ),
        )

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # outputs is a shard on last dimension DTensor, i.e. Shard(-1)
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )


class ParallelNonlinearity(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Replicate(),)
        self.output_layouts = (output_layouts or Shard(-1),)
        # colwise linear runtime sharding (desired sharding):
        # 1. requires replicate input
        # 2. shard output on last dim
        self.desired_input_layouts = (Replicate(),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        input_tensor = inputs[0]
        assert isinstance(input_tensor, DTensor), "Input must be a DTensor"
        return input_tensor

    def _partition_jump_relu_fn(self, name, module, device_mesh):
        module.register_parameter(
            "theta",
            nn.Parameter(
                distribute_tensor(
                    module.theta,
                    device_mesh,
                    [Shard(-1)],
                    src_data_rank=self.src_data_rank,
                )
            ),
        )
        module.register_buffer(
            "bandwidth",
            distribute_tensor(
                module.bandwidth,
                device_mesh,
                [Replicate()],
                src_data_rank=self.src_data_rank,
            ),
        )

    def _partition_relu_fn(self, name, module, device_mesh):
        pass

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # outputs is a shard on last dimension DTensor, i.e. Shard(-1)
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        if isinstance(module, nn.ReLU):
            _partition_fn = self._partition_relu_fn
        elif isinstance(module, JumpReLU):
            _partition_fn = self._partition_jump_relu_fn
        else:
            raise NotImplementedError(f"Unsupported nonlinearity: {type(module)}")
        return distribute_module(
            module,
            device_mesh,
            _partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )


class RowParallelDecoder(ParallelStyle):
    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        self.input_layouts = (input_layouts or Shard(-1),)
        self.output_layouts = (output_layouts or Replicate(),)
        self.use_local_output = use_local_output

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        input_tensor = inputs[0]
        assert isinstance(
            input_tensor, DTensor
        ), "Input must be a DTensor, got: " + str(type(input_tensor))
        return input_tensor

    def _partition_fn(self, name, module, device_mesh):
        for i in range(module.config.get("n_layers", 12)):
            name = f"W_{i}"
            param = module.get_parameter(name)
            dist_param = nn.Parameter(
                distribute_tensor(
                    param, device_mesh, [Shard(-2)], src_data_rank=self.src_data_rank
                )
            )
            module.register_parameter(name, dist_param)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # Rowwise sharding produces partial output, depending on output layouts:
        # 1. to replicate -> allreduce
        # 2. to shard -> reduce_scatter
        recons, sparsity = outputs
        if recons.placements != output_layouts:
            recons = recons.redistribute(placements=output_layouts, async_op=False)
        if sparsity.placements != output_layouts:
            sparsity = sparsity.redistribute(placements=output_layouts, async_op=False)
        # back to local tensor if use_local_output is True
        return (
            recons.to_local() if use_local_output else recons,
            sparsity.to_local() if use_local_output else sparsity,
        )

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        partition_fn = self._partition_fn
        # rowwise linear runtime sharding requires input tensor shard on last dim
        self.desired_input_layouts: tuple[Placement, ...] = (Shard(-1),)

        return distribute_module(
            module,
            device_mesh,
            partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )


#
class CrossLayerTranscoder(L.LightningModule):
    def __init__(self, config, nonlinearity, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(config)

        self.config = config
        d_acts = config.get("d_acts", 768)
        d_features = config.get("d_features", 768 * 8)
        n_layers = config.get("n_layers", 12)  #

        # loss hyperparams:
        self._lambda = config.get("lambda", 0.1)
        self.c = config.get("c", 0.1)

        self.nonlinearity = nonlinearity

        # Instead of raw parameters

        # self.W_enc = nn.ModuleList(
        #    [nn.Linear(d_acts, d_features, bias=False) for _ in range(n_layers)]
        # )
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # self.replacement_model = ReplacementModelAccuracy()

        # initialize parameters
        # self.reset_parameters()

    def forward(
        self, acts_norm: Float[torch.Tensor, "batch_size n_layers d_acts"]
    ) -> Tuple[
        Float[torch.Tensor, "batch_size n_layers d_features"],
        Float[torch.Tensor, "batch_size n_layers d_acts"],
    ]:

        features = self.encoder(acts_norm)
        # features = torch.stack(features, dim=1)

        features = self.nonlinearity(features)
        recons, sparsity = self.decoder(features)

        return features, recons, sparsity

    def training_step(self, batch, batch_idx):
        mean = batch.mean(dim=-1, keepdim=True)
        std = batch.std(dim=-1, keepdim=True)
        acts_norm = (batch - mean) / std
        resid, mlp_out = acts_norm[:, 0], batch[:, 1]
        # mlp_out = resid.clone()
        features, recons, sparsity = self.forward(resid)

        # MSE
        if hasattr(recons, "wait"):
            recons = recons.wait()
        if hasattr(sparsity, "wait"):
            sparsity = sparsity.wait()
        mse = ((recons - mlp_out) ** 2).mean()

        loss = mse + sparsity

        self.log("train_loss", loss)
        self.log("train_mse", mse)
        self.log("train_sparsity", sparsity)
        self.log("L0 (%)", 100 * (features > 0).float().mean())
        self.log(
            "L0 (Avg. per layer)",
            (features > 0).float().sum() / (features.shape[0] * features.shape[1]),
        )

        if False:  # batch_idx % 100 == 0:
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
        if hasattr(loss, "wait"):
            loss = loss.wait()
        return loss

    def validation_step(self, batch, batch_idx):
        # we just need this here such that on_validation_epoch_end is called
        pass

    def on_validation_epoch_end(self):
        self.replacement_model.update(self)
        self.log("replacement_model_accuracy", self.replacement_model.compute())
        print("exiting val epoch end")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr", 1e-3))
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
        return optimizer

    def configure_model(self):
        print("configure model")
        tp_mesh = self.device_mesh["tensor_parallel"]
        plan = {
            "encoder": ColParallelEncoder(use_local_output=False),
            "nonlinearity": ParallelNonlinearity(use_local_output=False),
            "decoder": RowParallelDecoder(),
        }
        parallelize_module(self, tp_mesh, plan)
