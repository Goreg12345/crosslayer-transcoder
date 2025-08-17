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
from torch.distributed.tensor import (
    DeviceMesh,
    DTensor,
    Replicate,
    Shard,
    distribute_module,
    distribute_tensor,
)
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import OpSchema  # OpSpec,
from torch.distributed.tensor._op_schema import (
    OpStrategy,
    PlacementStrategy,
    RuntimeSchemaInfo,
    StrategyType,
    _is_inplace_op,
)
from torch.distributed.tensor._ops.utils import is_tensor_dim_sharded, normalize_dim, register_op_strategy
from torch.distributed.tensor.parallel import RowwiseParallel, parallelize_module
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard
from torch.nn.modules.activation import ReLU

import wandb
from crosslayer_transcoder.metrics.replacement_model_accuracy import ReplacementModelAccuracy
from crosslayer_transcoder.model.jumprelu import JumpReLU

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
                        "Selecting along a dimension that is sharded " "is not supported on PyTorch-2.7.1."
                    )
                # shift shard index left if we dropped an earlier dimension
                if shard_dim > selected_dim:
                    shard_dim -= 1
                placement = Shard(shard_dim)

            # Replicate or Partial stay as-is
            new_placements.append(placement)

        out_spec = DTensorSpec(mesh=in_spec.mesh, placements=tuple(new_placements))

        out_strategy.strategies.append(PlacementStrategy(output_specs=out_spec, input_specs=(in_spec,)))

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
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, input_layouts, run_check=False)

        # transform the input layouts to the desired layouts of ColwiseParallel
        if input_layouts != desired_input_layouts:
            input_tensor = input_tensor.redistribute(placements=desired_input_layouts, async_op=True)
        return input_tensor

    def _partition_fn(self, name, module, device_mesh):
        module.register_parameter(
            "W",
            nn.Parameter(
                distribute_tensor(module.W, device_mesh, [Shard(-1)], src_data_rank=self.src_data_rank)
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
            partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts),
            partial(self._prepare_output_fn, self.output_layouts, self.use_local_output),
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
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
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
            partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts),
            partial(self._prepare_output_fn, self.output_layouts, self.use_local_output),
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
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        input_tensor = inputs[0]
        assert isinstance(input_tensor, DTensor), "Input must be a DTensor, got: " + str(type(input_tensor))
        return input_tensor

    def _partition_fn(self, name, module, device_mesh):
        for i in range(module.config.get("n_layers", 12)):
            name = f"W_{i}"
            param = module.get_parameter(name)
            dist_param = nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(-2)], src_data_rank=self.src_data_rank)
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
            partial(self._prepare_input_fn, self.input_layouts, self.desired_input_layouts),
            partial(self._prepare_output_fn, self.output_layouts, self.use_local_output),
        )
