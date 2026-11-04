# SPDX-License-Identifier: Apache-2.0
"""Ascend communication at the upstream modular MoE boundary."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import modular_kernel as mk


@dataclass
class NpuExpertTokensMetadata(mk.ExpertTokensMetadata):
    group_list: torch.Tensor
    group_list_type: int
    output_dtype: torch.dtype


@dataclass
class _PendingForward:
    dispatcher: object
    handle: object
    original_shape: torch.Size
    padded_rows: int
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


class NpuMoEPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """One synchronous dispatch/combine for physical-TP replicated inputs.

    Runtime supplies caller-owned communication groups and synchronized
    forward facts. Exactly one physical TP slice is performed. Explicit SP,
    async prepare/finalize and DBO are unsupported. Exceptions leave the
    instance unusable rather than silently discarding an unfinished forward.
    """

    def __init__(self, runtime, num_experts, is_sequence_parallel=False):
        if is_sequence_parallel:
            raise NotImplementedError("NPU MoE prepare requires TP-replicated rows")
        if not isinstance(num_experts, int) or num_experts <= 0:
            raise ValueError("num_experts must be a positive integer")
        self.runtime = runtime
        self.num_experts = num_experts
        self._tp = runtime.tp_group
        if (
            self._tp.world_size <= 0
            or not 0 <= self._tp.rank_in_group < self._tp.world_size
        ):
            raise ValueError("Invalid physical TP group")
        # Resolve all communicator metadata before graph capture starts.
        self._mc2 = runtime.mc2_dispatcher(num_experts)
        self._alltoall = runtime.alltoall_dispatcher(num_experts)
        self._pending = None
        self._busy = False
        self._failed = False

    @property
    def activation_format(self):
        return mk.FusedMoEActivationFormat.Standard

    def topk_indices_dtype(self):
        return torch.int32

    def max_num_tokens_per_rank(self):
        return None

    def num_dispatchers(self):
        return 1

    def output_is_reduced(self):
        return True

    def prepare(
        self,
        a1,
        topk_weights,
        topk_ids,
        num_experts,
        expert_map,
        apply_router_weight_on_input,
        quant_config,
        defer_input_quant=False,
    ):
        if self._busy or self._failed:
            raise RuntimeError(
                "NPU MoE prepare is reentrant or has an unfinished/failed forward"
            )
        self._busy = True
        try:
            if apply_router_weight_on_input or defer_input_quant:
                raise NotImplementedError(
                    "NPU dispatched experts require output routing without deferred input quantization"
                )
            if num_experts != self.num_experts:
                raise ValueError(
                    "Global expert count changed after NPU MoE initialization"
                )
            if (
                a1.ndim != 2
                or topk_ids.ndim != 2
                or topk_weights.shape != topk_ids.shape
            ):
                raise ValueError("Expected 2D MoE rows and matching topk IDs/weights")
            if topk_ids.shape[0] != a1.shape[0]:
                raise ValueError("Routing rows must match original hidden states")
            facts = get_forward_context().additional_kwargs.get("npu_moe")
            if facts is None:
                raise RuntimeError("NPU MoE forward facts are missing")
            tp_size, tp_rank = self._tp.world_size, self._tp.rank_in_group
            if facts.use_mc2:
                padded_rows = facts.padded_num_tokens
                if (
                    padded_rows <= 0
                    or padded_rows < a1.shape[0]
                    or padded_rows % tp_size
                ):
                    raise ValueError(
                        "MC2 padded length must cover rows and align to physical TP"
                    )
                mask = facts.active_mask
                if (
                    mask.dtype != torch.bool
                    or mask.shape != (padded_rows,)
                    or mask.device != a1.device
                ):
                    raise ValueError(
                        "MC2 full active mask must match padded rows/device"
                    )
                dispatcher = self._mc2
            else:
                # Replicated rows need equal contiguous TP slices and a regular
                # all-gather. Unlike native tensor_split's nondivisible branch,
                # align every local prefill length explicitly for this layout.
                padded_rows = (a1.shape[0] + tp_size - 1) // tp_size * tp_size
                dispatcher = self._alltoall
            pad = padded_rows - a1.shape[0]
            hidden = F.pad(a1, (0, 0, 0, pad)) if pad else a1
            ids = topk_ids.to(torch.int32)
            probabilities = topk_weights.to(torch.float32)
            if pad:
                ids = F.pad(ids, (0, 0, 0, pad))
                probabilities = F.pad(probabilities, (0, 0, 0, pad))
            rows = padded_rows // tp_size
            start = tp_rank * rows
            hidden = hidden.narrow(0, start, rows)
            ids = ids.narrow(0, start, rows)
            probabilities = probabilities.narrow(0, start, rows)
            if facts.use_mc2:
                result = dispatcher.dispatch(
                    hidden, ids, probabilities, mask.narrow(0, start, rows)
                )
            else:
                result = dispatcher.dispatch(hidden, ids, probabilities)
            raw_counts = result.group_list
            if result.group_list_type == 0:
                counts = torch.cat((raw_counts[:1], raw_counts[1:] - raw_counts[:-1]))
            elif result.group_list_type == 1:
                counts = raw_counts
            else:
                raise ValueError("Unsupported NPU expert count format")
            metadata = NpuExpertTokensMetadata(
                expert_num_tokens=counts,
                expert_num_tokens_cpu=None,
                group_list=raw_counts,
                group_list_type=result.group_list_type,
                output_dtype=a1.dtype,
            )
            self._pending = _PendingForward(
                dispatcher,
                result.handle,
                a1.shape,
                padded_rows,
                topk_ids,
                topk_weights,
            )
            return (
                result.hidden_states,
                result.pertoken_scale,
                metadata,
                topk_ids,
                topk_weights,
            )
        except BaseException:
            self._failed = True
            raise

    def finalize(
        self,
        output,
        fused_expert_output,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input,
        weight_and_reduce_impl,
    ):
        if self._failed or not self._busy or self._pending is None:
            raise RuntimeError("NPU MoE finalize has no successful matching prepare")
        try:
            pending = self._pending
            if (
                topk_ids is not pending.topk_ids
                or topk_weights is not pending.topk_weights
                or apply_router_weight_on_input
                or output.shape != pending.original_shape
            ):
                raise ValueError("NPU MoE finalize does not match its pending forward")
            combined = pending.dispatcher.combine(fused_expert_output, pending.handle)
            expected = (pending.padded_rows // self._tp.world_size, output.shape[1])
            if (
                combined.shape != expected
                or combined.dtype != output.dtype
                or combined.device != output.device
            ):
                raise ValueError("NPU combined rows do not match original TP layout")
            if self._tp.world_size > 1 and pending.padded_rows > 0:
                # Never alias input/shared-expert storage while gathering.
                gathered = torch.empty(
                    (pending.padded_rows, output.shape[1]),
                    dtype=output.dtype,
                    device=output.device,
                )
                dist.all_gather_into_tensor(
                    gathered, combined.contiguous(), group=self._tp.device_group
                )
            else:
                gathered = combined
            output.copy_(gathered[: pending.original_shape[0]])
            self._pending = None
            self._busy = False
        except BaseException:
            self._failed = True
            raise
