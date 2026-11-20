# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) 2026 FlagOS Contributors
"""Ordinary MC2 communication for unquantized BF16/FP16 experts.

The caller owns a dedicated HCCL process group and its entire lifecycle.
This component neither creates groups nor selects a model execution path.
The BF16/FP16 path uses uniform padded capacity, prefix counts and activity
masks. Hardware and topology support require component validation.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class _MC2CombineHandle:
    owner: object
    process_group: dist.ProcessGroup
    communicator: str
    rank: int
    world_size: int
    expert_ids: torch.Tensor
    expert_weights: torch.Tensor
    active_mask: torch.Tensor
    assist_info: torch.Tensor
    ep_recv_counts: torch.Tensor
    expand_scales: torch.Tensor
    original_shape: torch.Size
    expanded_shape: torch.Size
    output_dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class MoeDispatchResult:
    hidden_states: torch.Tensor
    pertoken_scale: torch.Tensor | None
    group_list: torch.Tensor
    group_list_type: int
    handle: object


class NpuMC2Dispatcher:
    """Dispatch unquantized expert rows and combine an explicit forward handle.

    Group members must call in the same order, including idle ranks using
    false activity masks. Input rows use uniform padded capacity across the
    group. Routing IDs are global and experts are equally sharded. Callers
    retain routing/mask/handle tensors until combine and graph replay finish;
    data may change in static buffers between replays. No handle is cached.
    Recreating a group requires a new dispatcher and graph recapture; this
    component does not manage or validate group recreation.
    """

    def __init__(self, process_group: dist.ProcessGroup, num_experts: int):
        import torch_npu

        if not all(
            hasattr(torch_npu, name)
            for name in (
                "npu_moe_distribute_dispatch_v2",
                "npu_moe_distribute_combine_v2",
            )
        ):
            raise RuntimeError("MC2 requires torch_npu dispatch/combine v2")
        if not isinstance(num_experts, int) or num_experts <= 0:
            raise ValueError("num_experts must be a positive integer")
        self._ops = torch_npu
        self._group = process_group
        self._num_experts = num_experts
        self._owner = object()
        if str(dist.get_backend(self._group)).lower() != "hccl":
            raise ValueError("MC2 requires an active HCCL process group")
        rank = dist.get_rank(self._group)
        world_size = dist.get_world_size(self._group)
        if rank < 0 or world_size <= 1 or self._num_experts % world_size:
            raise ValueError("MC2 requires group membership and equal expert shards")
        backend = self._group._get_backend(torch.device("npu"))
        communicator = backend.get_hccl_comm_name(rank)
        self._rank, self._world_size = rank, world_size
        self._communicator = communicator

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> MoeDispatchResult:
        """Return dispatched BF16/FP16 rows/prefix counts, without computation.

        IDs must be in the global expert range and weights finite; callers
        establish value invariants without host reads inside graph capture.
        """
        if hidden_states.ndim != 2 or hidden_states.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            raise ValueError("MC2 input must be BF16/FP16 [rows, hidden]")
        if hidden_states.device.type != "npu":
            raise ValueError("MC2 requires NPU tensors")
        if (
            expert_ids.ndim != 2
            or expert_ids.dtype != torch.int32
            or expert_ids.shape[0] != hidden_states.shape[0]
            or expert_ids.shape[1] == 0
        ):
            raise ValueError("MC2 IDs must be INT32 [rows, topk]")
        if (
            expert_weights.dtype != torch.float32
            or expert_weights.shape != expert_ids.shape
        ):
            raise ValueError("MC2 weights must be FP32 with the IDs shape")
        if (
            active_mask.dtype != torch.bool
            or active_mask.shape != hidden_states.shape[:1]
        ):
            raise ValueError("MC2 requires BOOL active_mask [rows]")
        if any(
            t.device != hidden_states.device
            for t in (expert_ids, expert_weights, active_mask)
        ):
            raise ValueError("MC2 tensors must share a device")
        expanded, _scales, assist, counts, ep_counts, _tp_counts, expand_scales = (
            self._ops.npu_moe_distribute_dispatch_v2(
                x=hidden_states,
                expert_ids=expert_ids,
                group_ep=self._communicator,
                ep_world_size=self._world_size,
                ep_rank_id=self._rank,
                moe_expert_num=self._num_experts,
                group_tp=self._communicator,
                tp_world_size=1,
                tp_rank_id=0,
                shared_expert_rank_num=0,
                expert_shard_type=0,
                quant_mode=0,
                global_bs=0,
                expert_token_nums_type=0,
                x_active_mask=active_mask,
            )
        )
        if expanded.dtype != hidden_states.dtype:
            raise RuntimeError("Unquantized MC2 must preserve the source token dtype")
        handle = _MC2CombineHandle(
            self._owner,
            self._group,
            self._communicator,
            self._rank,
            self._world_size,
            expert_ids,
            expert_weights,
            active_mask,
            assist,
            ep_counts,
            expand_scales,
            hidden_states.shape,
            expanded.shape,
            hidden_states.dtype,
            hidden_states.device,
        )
        return MoeDispatchResult(expanded, None, counts, 0, handle)

    def combine(self, hidden_states: torch.Tensor, handle: object) -> torch.Tensor:
        """Combine unweighted expert outputs using their matching dispatch state."""
        if (
            not isinstance(handle, _MC2CombineHandle)
            or handle.owner is not self._owner
            or handle.process_group is not self._group
            or handle.communicator != self._communicator
        ):
            raise ValueError("Foreign or stale MC2 combine handle")
        if (
            hidden_states.shape != handle.expanded_shape
            or hidden_states.device != handle.device
            or hidden_states.dtype != handle.output_dtype
        ):
            raise ValueError("MC2 expert output shape/device/dtype must match dispatch")
        output = self._ops.npu_moe_distribute_combine_v2(
            expand_x=hidden_states,
            expert_ids=handle.expert_ids,
            assist_info_for_combine=handle.assist_info,
            ep_send_counts=handle.ep_recv_counts,
            expert_scales=handle.expert_weights,
            group_ep=handle.communicator,
            ep_world_size=handle.world_size,
            ep_rank_id=handle.rank,
            moe_expert_num=self._num_experts,
            tp_send_counts=torch.empty(1, dtype=torch.int32, device=handle.device),
            group_tp=handle.communicator,
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_rank_num=0,
            global_bs=0,
            x_active_mask=handle.active_mask,
            expand_scales=handle.expand_scales,
            comm_quant_mode=0,
        )
        if output.shape != handle.original_shape:
            raise RuntimeError("MC2 combine did not restore the source token shape")
        return output


@dataclass(frozen=True)
class _AllToAllCombineHandle:
    owner: object
    process_group: dist.ProcessGroup
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]
    inverse_local: torch.Tensor | None
    inverse_global: torch.Tensor | None
    expert_weights: torch.Tensor
    original_shape: torch.Size
    expanded_shape: torch.Size
    output_dtype: torch.dtype
    device: torch.device


class NpuAllToAllDispatcher:
    """Unquantized BF16/FP16 AllToAll for eager prefill, adapted from rc1 semantics.

    The caller owns a live HCCL group and equally partitioned contiguous
    experts. Histogram gathering and CPU split vectors are intentionally
    dynamic; FULL graph capture is unsupported. Empty source/receiver ranks
    still participate in every collective but bypass empty local operators.
    IDs must be global and valid, and weights finite. No handle is cached.
    """

    def __init__(self, process_group: dist.ProcessGroup, num_experts: int):
        import torch_npu

        if not isinstance(num_experts, int) or num_experts <= 0:
            raise ValueError("num_experts must be a positive integer")
        if str(dist.get_backend(process_group)).lower() != "hccl":
            raise ValueError("AllToAll requires an active HCCL process group")
        self._rank = dist.get_rank(process_group)
        self._world_size = dist.get_world_size(process_group)
        if self._rank < 0 or self._world_size <= 1 or num_experts % self._world_size:
            raise ValueError("AllToAll requires membership and equal expert shards")
        self._ops = torch_npu
        self._group = process_group
        self._num_experts = num_experts
        self._local_experts = num_experts // self._world_size
        self._owner = object()

    def _check_eager(self) -> None:
        capturing = getattr(self._ops.npu, "is_current_stream_capturing", None)
        if capturing is not None and capturing():
            raise RuntimeError(
                "Dynamic AllToAll supports eager prefill, not graph capture"
            )

    def _exchange(
        self,
        tensor: torch.Tensor,
        output_splits: tuple[int, ...],
        input_splits: tuple[int, ...],
    ) -> torch.Tensor:
        output = tensor.new_empty((sum(output_splits), *tensor.shape[1:]))
        work = dist.all_to_all_single(
            output,
            tensor.contiguous(),
            output_split_sizes=list(output_splits),
            input_split_sizes=list(input_splits),
            group=self._group,
            async_op=True,
        )
        work.wait()
        return output

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> MoeDispatchResult:
        """Exchange unquantized rows after source permutation."""
        self._check_eager()
        if (
            hidden_states.ndim != 2
            or hidden_states.dtype
            not in (
                torch.bfloat16,
                torch.float16,
            )
            or hidden_states.shape[1] == 0
        ):
            raise ValueError("AllToAll input must be BF16/FP16 [rows, hidden]")
        if hidden_states.device.type != "npu":
            raise ValueError("AllToAll requires NPU tensors")
        if (
            expert_ids.ndim != 2
            or expert_ids.dtype != torch.int32
            or expert_ids.shape[0] != hidden_states.shape[0]
            or expert_ids.shape[1] == 0
        ):
            raise ValueError("AllToAll IDs must be INT32 [rows, topk]")
        if (
            expert_weights.dtype != torch.float32
            or expert_weights.shape != expert_ids.shape
        ):
            raise ValueError("AllToAll weights must be FP32 with the IDs shape")
        if any(t.device != hidden_states.device for t in (expert_ids, expert_weights)):
            raise ValueError("AllToAll tensors must share a device")
        if hidden_states.shape[0]:
            histogram = torch.histc(
                expert_ids,
                bins=self._num_experts,
                min=0,
                max=self._num_experts,
            ).to(torch.int64)
        else:
            histogram = torch.zeros(
                self._num_experts, dtype=torch.int64, device=hidden_states.device
            )
        gathered = histogram.new_empty(self._world_size * self._num_experts)
        dist.all_gather_into_tensor(gathered, histogram.contiguous(), group=self._group)
        gathered = gathered.reshape(self._world_size, self._num_experts)
        start = self._rank * self._local_experts
        received_counts = gathered[:, start : start + self._local_experts]
        # Host reads are deliberate: this is the native dynamic prefill path.
        input_splits = tuple(
            histogram.reshape(self._world_size, self._local_experts)
            .sum(1)
            .cpu()
            .tolist()
        )
        output_splits = tuple(received_counts.sum(1).cpu().tolist())
        counts = received_counts.sum(0).to(torch.int64)
        if hidden_states.shape[0]:
            permuted, inverse_local = self._ops.npu_moe_token_permute(
                tokens=hidden_states,
                indices=expert_ids,
                num_out_tokens=expert_ids.numel(),
            )
        else:
            permuted = hidden_states.new_empty((0, hidden_states.shape[1]))
            inverse_local = None
        received = self._exchange(permuted, output_splits, input_splits)
        inverse_global = None
        if received.shape[0] and self._local_experts > 1:
            indices = torch.repeat_interleave(
                torch.arange(
                    self._num_experts, dtype=torch.int32, device=hidden_states.device
                )
                % self._local_experts,
                received_counts.reshape(-1),
            )
            received, inverse_global = self._ops.npu_moe_token_permute(
                received, indices
            )
        handle = _AllToAllCombineHandle(
            self._owner,
            self._group,
            input_splits,
            output_splits,
            inverse_local,
            inverse_global,
            expert_weights,
            hidden_states.shape,
            received.shape,
            hidden_states.dtype,
            hidden_states.device,
        )
        return MoeDispatchResult(received, None, counts, 1, handle)

    def combine(self, hidden_states: torch.Tensor, handle: object) -> torch.Tensor:
        """Undo receiver permutation, reverse exchange, then weight/unpermute."""
        self._check_eager()
        if (
            not isinstance(handle, _AllToAllCombineHandle)
            or handle.owner is not self._owner
            or handle.process_group is not self._group
        ):
            raise ValueError("Foreign or stale AllToAll combine handle")
        if (
            hidden_states.shape != handle.expanded_shape
            or hidden_states.device != handle.device
            or hidden_states.dtype != handle.output_dtype
        ):
            raise ValueError(
                "AllToAll expert output shape/device/dtype must match dispatch"
            )
        if handle.inverse_global is not None:
            hidden_states = self._ops.npu_moe_token_unpermute(
                hidden_states, handle.inverse_global
            )
        source_rows = self._exchange(
            hidden_states, handle.input_splits, handle.output_splits
        )
        if handle.inverse_local is None:
            return source_rows.new_empty(handle.original_shape)
        output = self._ops.npu_moe_token_unpermute(
            permuted_tokens=source_rows,
            sorted_indices=handle.inverse_local.to(torch.int32),
            probs=handle.expert_weights,
            restore_shape=handle.original_shape,
        )
        if output.shape != handle.original_shape:
            raise RuntimeError("AllToAll combine did not restore source token shape")
        return output
