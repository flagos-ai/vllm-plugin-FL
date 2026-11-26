# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) 2026 FlagOS Contributors
"""Owned A3 MoE communication state and pre-graph activity updates."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm_fl.distributed.device_communicators.npu_moe import (
    NpuAllToAllDispatcher,
    NpuMC2Dispatcher,
)


@dataclass(frozen=True)
class NpuMoEForwardState:
    use_mc2: bool
    padded_num_tokens: int
    active_mask: torch.Tensor
    in_profile_run: bool


def is_a3_device() -> bool:
    import torch_npu

    return 250 <= torch_npu.npu.get_soc_version() <= 255


class NpuMoERuntime:
    """One EP communicator owns this runtime, its group, and static mask.

    Only explicitly supported Ascend EP callers opt into this state; other
    hardware continues to use its existing communicator. Group creation is collective
    among the active EP members, including when EP is a subgroup under PP.
    """

    def __init__(self, communicator, config, device):
        from vllm.distributed import get_tp_group

        if config.parallel_config.use_ubatching:
            raise NotImplementedError(
                "NPU MoE communication does not support DBO or ubatching"
            )
        self.tp_group = get_tp_group()
        self.device = torch.device(device)
        if self.device.type != "npu" or self.device.index != torch.npu.current_device():
            raise ValueError("MoE buffers require the active worker's NPU device")
        self._group = None
        self._mc2 = {}
        self._alltoall = {}
        self._closed = False
        tp = self.tp_group.world_size
        options = config.additional_config or {}
        if options.get("enable_fused_mc2", 0) not in (0, False):
            raise NotImplementedError(
                "Unquantized ordinary MoE does not enable fused MC2"
            )
        if options.get("enable_mc2_hierarchy_comm", False):
            raise NotImplementedError(
                "Ordinary BF16 MoE does not enable hierarchical MC2"
            )
        max_batched = config.scheduler_config.max_num_batched_tokens
        capture = config.compilation_config
        if options.get("enable_prefill_mc2", False):
            capacity = max_batched
        elif capture.cudagraph_capture_sizes:
            capacity = capture.max_cudagraph_capture_size
        else:
            capacity = config.scheduler_config.max_num_seqs
        self.mc2_tokens_capacity = min((capacity + tp - 1) // tp, 512) * tp
        mask_rows = max(max_batched, self.mc2_tokens_capacity)
        mask_rows = (mask_rows + tp - 1) // tp * tp
        self._mask = torch.zeros(mask_rows, dtype=torch.bool, device=self.device)
        # Do not reuse the ordinary EP collective communicator for MC2.
        # Local synchronization permits independent, non-overlapping PP groups.
        self._group = dist.new_group(
            ranks=list(communicator.ranks),
            backend="hccl",
            use_local_synchronization=True,
        )
        if (
            dist.get_world_size(self._group) != communicator.world_size
            or dist.get_rank(self._group) != communicator.rank_in_group
        ):
            self.destroy()
            raise RuntimeError("MoE private group does not match the active EP group")

    def mc2_dispatcher(self, num_experts):
        self._check_open()
        if num_experts not in self._mc2:
            self._mc2[num_experts] = NpuMC2Dispatcher(self._group, num_experts)
        return self._mc2[num_experts]

    def alltoall_dispatcher(self, num_experts):
        self._check_open()
        if num_experts not in self._alltoall:
            self._alltoall[num_experts] = NpuAllToAllDispatcher(
                self._group, num_experts
            )
        return self._alltoall[num_experts]

    def prepare_forward(
        self,
        *,
        num_tokens,
        num_actual_tokens,
        dp_metadata,
        cudagraph_runtime_mode,
        in_profile_run=False,
    ):
        from vllm.config import CUDAGraphMode

        self._check_open()
        if num_tokens is None or num_actual_tokens is None:
            raise RuntimeError("A3 MoE requires explicit runner token activity")
        if not 0 <= num_actual_tokens <= num_tokens:
            raise ValueError("Actual token count must fit the local padded forward")
        # FULL DP metadata contains uniform padded capacities, not activity.
        # It is suitable for shared branch selection, never for the mask prefix.
        max_tokens = num_tokens
        if dp_metadata is not None:
            counts = dp_metadata.num_tokens_across_dp_cpu
            if counts.device.type != "cpu":
                raise ValueError("MoE path selection requires CPU DP capacities")
            max_tokens = max(num_tokens, int(counts.max().item()))
        tp = self.tp_group.world_size
        padded = (max_tokens + tp - 1) // tp * tp
        use_mc2 = max_tokens <= self.mc2_tokens_capacity
        if cudagraph_runtime_mode != CUDAGraphMode.NONE and not use_mc2:
            raise RuntimeError("Dynamic AllToAll cannot execute inside a FULL graph")
        if padded > self._mask.numel():
            raise ValueError("MoE token capacity exceeds the reserved activity mask")
        mask = self._mask[:padded]
        # This runs outside the captured graph on every forward, including replay.
        mask[:num_actual_tokens].fill_(True)
        mask[num_actual_tokens:].fill_(False)
        return NpuMoEForwardState(use_mc2, padded, mask, in_profile_run)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("MoE runtime was destroyed; recreate and recapture")

    def destroy(self):
        if not self._closed:
            self._closed = True
            self._mc2.clear()
            self._alltoall.clear()
            group, self._group = self._group, None
            if group is not None:
                dist.destroy_process_group(group)


def get_moe_runtime():
    """Return existing EP-owned state without creating groups in a forward."""
    from vllm.distributed import get_ep_group

    communicator = get_ep_group().device_communicator
    return getattr(communicator, "moe_runtime", None)
