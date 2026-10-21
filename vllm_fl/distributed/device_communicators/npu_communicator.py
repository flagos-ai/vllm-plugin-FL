# Copyright (c) 2026 BAAI. All rights reserved.
"""HCCL collectives for Ascend, including modular MoE DP dispatch/combine."""

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)


class NPUCommunicator(DeviceCommunicatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_all2all:
            if self.all2all_backend != "allgather_reducescatter":
                raise NotImplementedError(
                    "Ascend MoE communication requires allgather_reducescatter"
                )
            from vllm.distributed.device_communicators.all2all import (
                AgRsAll2AllManager,
            )

            self.all2all_manager = AgRsAll2AllManager(self.cpu_group)

    def _check_sizes(self, sizes):
        if len(sizes) != self.world_size or any(size < 0 for size in sizes):
            raise ValueError("Collective sizes must match the active process group")

    def all_gatherv(self, input_, dim=0, sizes=None):
        if dim != 0:
            raise NotImplementedError("Ascend all_gatherv supports dim=0")
        if sizes is not None:
            self._check_sizes(sizes)

        def gather(tensor):
            if sizes is None:
                return self.all_gather(tensor, dim=0)
            if tensor.shape[0] != sizes[self.rank_in_group]:
                raise ValueError("Local rows do not match collective sizes")
            max_rows = max(sizes)
            if max_rows == 0 or self.world_size == 1:
                return tensor
            if min(sizes) == max_rows:
                return self.all_gather(tensor, dim=0)
            # HCCL tensor collectives require equal physical shapes. Preserve
            # DP order, padding before communication and trimming afterwards.
            padded = tensor.new_zeros((max_rows, *tensor.shape[1:]))
            padded[: tensor.shape[0]].copy_(tensor)
            gathered = self.all_gather(padded, dim=0)
            return torch.cat(
                [gathered[rank * max_rows : rank * max_rows + size]
                 for rank, size in enumerate(sizes)], dim=0
            )

        if isinstance(input_, torch.Tensor):
            return gather(input_)
        return [gather(tensor) for tensor in input_]

    def reduce_scatterv(self, input_, dim=-1, sizes=None):
        if sizes is None:
            return self.reduce_scatter(input_, dim=dim)
        self._check_sizes(sizes)
        dim %= input_.ndim
        tensor = input_.movedim(dim, 0)
        if tensor.shape[0] != sum(sizes):
            raise ValueError("Input rows do not match collective sizes")
        max_rows = max(sizes)
        if max_rows == 0 or self.world_size == 1:
            return input_
        if min(sizes) == max_rows:
            return self.reduce_scatter(input_, dim=dim)
        padded = tensor.new_zeros((self.world_size * max_rows, *tensor.shape[1:]))
        offset = 0
        for rank, size in enumerate(sizes):
            padded[rank * max_rows : rank * max_rows + size].copy_(
                tensor[offset : offset + size]
            )
            offset += size
        output = tensor.new_empty((max_rows, *tensor.shape[1:]))
        dist.reduce_scatter_tensor(output, padded, group=self.device_group)
        return output[: sizes[self.rank_in_group]].movedim(0, dim).contiguous()

    def dispatch_router_logits(self, *args, **kwargs):
        if self.use_all2all:
            return self.all2all_manager.dispatch_router_logits(*args, **kwargs)
        return super().dispatch_router_logits(*args, **kwargs)

    def dispatch(self, *args, **kwargs):
        if self.use_all2all:
            return self.all2all_manager.dispatch(*args, **kwargs)
        return super().dispatch(*args, **kwargs)

    def combine(self, *args, **kwargs):
        if self.use_all2all:
            return self.all2all_manager.combine(*args, **kwargs)
        return super().combine(*args, **kwargs)
