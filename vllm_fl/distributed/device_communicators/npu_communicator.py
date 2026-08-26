# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from vllm-ascend v0.20.2rc1:
# vllm_ascend/distributed/device_communicators/npu_communicator.py

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)


class NPUCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: dist.ProcessGroup,
        device: torch.device | None = None,
        device_group: dist.ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        # Initialize the device according to rank, as vLLM-Ascend does.
        self.device = torch.npu.current_device()

        # Keep the CUDA communicator-compatible surface used by graph code.
        self.ca_comm = None

        if self.use_all2all and self.all2all_backend in (
            "naive",
            "allgather_reducescatter",
        ):
            from vllm.distributed.device_communicators.all2all import (
                AgRsAll2AllManager,
            )

            self.all2all_manager = AgRsAll2AllManager(self.cpu_group)

    def _validate_rank(self) -> None:
        assert isinstance(self.world_size, int) and self.world_size > 0, (
            f"world_size must be a positive integer, got {self.world_size!r}"
        )
        assert isinstance(self.rank_in_group, int), (
            f"rank_in_group must be an integer, got {self.rank_in_group!r}"
        )
        assert 0 <= self.rank_in_group < self.world_size, (
            f"rank_in_group ({self.rank_in_group}) must be in "
            f"[0, {self.world_size})"
        )

    @staticmethod
    def _normalize_dim(input_: torch.Tensor, dim: int) -> int:
        assert isinstance(input_, torch.Tensor), (
            f"input must be a Tensor, got {type(input_).__name__}"
        )
        assert input_.dim() > 0, "input tensor must have at least one dimension"
        assert isinstance(dim, int) and -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        return dim % input_.dim()

    def _validate_sizes(
        self,
        sizes: list[int] | None,
        local_size: int,
        *,
        is_gather: bool,
    ) -> list[int]:
        if sizes is None:
            if is_gather:
                return [local_size] * self.world_size
            assert local_size % self.world_size == 0, (
                f"input size ({local_size}) must be divisible by world_size "
                f"({self.world_size}) when sizes is None"
            )
            return [local_size // self.world_size] * self.world_size

        assert isinstance(sizes, list), (
            f"sizes must be a list of integers, got {type(sizes).__name__}"
        )
        assert len(sizes) == self.world_size, (
            f"sizes length ({len(sizes)}) must equal world_size "
            f"({self.world_size})"
        )
        assert all(
            isinstance(size, int) and not isinstance(size, bool) for size in sizes
        ), f"sizes must contain only integers, got {sizes!r}"
        assert all(size >= 0 for size in sizes), (
            f"sizes must be non-negative, got {sizes!r}"
        )
        if is_gather:
            assert local_size == sizes[self.rank_in_group], (
                f"local input size ({local_size}) must equal sizes at "
                f"rank_in_group {self.rank_in_group} "
                f"({sizes[self.rank_in_group]})"
            )
        else:
            assert local_size == sum(sizes), (
                f"input size ({local_size}) must equal sum of sizes "
                f"({sum(sizes)})"
            )
        return list(sizes)

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Gather tensors whose length along ``dim`` differs by rank.

        HCCL's tensor all-gather requires equal input shapes, so each rank pads
        to the largest public size. The gathered rank-major buffer is then
        stripped of padding and concatenated in rank order.
        """
        self._validate_rank()
        is_single_tensor = isinstance(input_, torch.Tensor)
        inputs = [input_] if is_single_tensor else input_
        assert isinstance(inputs, list), (
            f"input must be a Tensor or list[Tensor], got {type(input_).__name__}"
        )

        validated = []
        for tensor in inputs:
            normalized_dim = self._normalize_dim(tensor, dim)
            public_sizes = self._validate_sizes(
                sizes, tensor.shape[normalized_dim], is_gather=True
            )
            validated.append((tensor, normalized_dim, public_sizes))

        outputs = []
        for tensor, normalized_dim, public_sizes in validated:
            moved_input = tensor.movedim(normalized_dim, 0).contiguous()
            max_size = max(public_sizes)
            if self.world_size == 1 or max_size == 0:
                output = moved_input
            elif all(size == public_sizes[0] for size in public_sizes):
                output = moved_input.new_empty(
                    (self.world_size * moved_input.shape[0],)
                    + moved_input.shape[1:]
                )
                dist.all_gather_into_tensor(
                    output, moved_input, group=self.device_group
                )
            else:
                padded_input = moved_input.new_zeros(
                    (max_size,) + moved_input.shape[1:]
                )
                padded_input[: moved_input.shape[0]].copy_(moved_input)
                gathered = moved_input.new_empty(
                    (self.world_size * max_size,) + moved_input.shape[1:]
                )
                dist.all_gather_into_tensor(
                    gathered, padded_input, group=self.device_group
                )
                gathered = gathered.reshape(
                    (self.world_size, max_size) + moved_input.shape[1:]
                )
                output = torch.cat(
                    [
                        gathered[rank, : public_sizes[rank]]
                        for rank in range(self.world_size)
                    ],
                    dim=0,
                )
            outputs.append(output.movedim(0, normalized_dim).contiguous())

        if is_single_tensor:
            return outputs[0]
        return outputs

    def reduce_scatterv(
        self,
        input_: torch.Tensor,
        dim: int = -1,
        sizes: list[int] | None = None,
    ) -> torch.Tensor:
        """Reduce then scatter rank segments of unequal lengths.

        The input is split by the public ``sizes`` vector, each segment is
        padded to the maximum size, and HCCL reduce-scatter operates on the
        resulting equal-size rank chunks. The local output is then truncated.
        """
        self._validate_rank()
        normalized_dim = self._normalize_dim(input_, dim)
        moved_input = input_.movedim(normalized_dim, 0).contiguous()
        public_sizes = self._validate_sizes(
            sizes, moved_input.shape[0], is_gather=False
        )
        max_size = max(public_sizes)

        if self.world_size == 1 or max_size == 0:
            output = moved_input[: public_sizes[self.rank_in_group]]
        elif all(size == public_sizes[0] for size in public_sizes):
            output = moved_input.new_empty(
                (public_sizes[self.rank_in_group],) + moved_input.shape[1:]
            )
            dist.reduce_scatter_tensor(
                output, moved_input, group=self.device_group
            )
        else:
            padded_input = moved_input.new_zeros(
                (self.world_size, max_size) + moved_input.shape[1:]
            )
            for rank, segment in enumerate(moved_input.split(public_sizes, dim=0)):
                padded_input[rank, : public_sizes[rank]].copy_(segment)
            padded_input = padded_input.reshape(
                (self.world_size * max_size,) + moved_input.shape[1:]
            ).contiguous()
            reduced = moved_input.new_empty((max_size,) + moved_input.shape[1:])
            dist.reduce_scatter_tensor(
                reduced, padded_input, group=self.device_group
            )
            output = reduced[: public_sizes[self.rank_in_group]]

        return output.movedim(0, normalized_dim).contiguous()

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            list[torch.Tensor],
        ]
    ):
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

    def combine(
        self,
        hidden_states: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        assert self.all2all_manager is not None
        return self.all2all_manager.combine(hidden_states, is_sequence_parallel)

    def destroy(self) -> None:
        if self.ca_comm is not None:
            self.ca_comm = None
        if self.all2all_manager is not None:
            self.all2all_manager.destroy()
            self.all2all_manager = None

    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        if scatter_dim < 0:
            scatter_dim += input_.dim()
        if gather_dim < 0:
            gather_dim += input_.dim()

        if scatter_sizes is not None and gather_sizes is not None:
            input_list = [
                tensor.contiguous()
                for tensor in torch.split(input_, scatter_sizes, scatter_dim)
            ]
            output_list = []
            tensor_shape_base = input_list[self.rank_in_group].size()
            for index in range(self.world_size):
                tensor_shape = list(tensor_shape_base)
                tensor_shape[gather_dim] = gather_sizes[index]
                output_list.append(
                    torch.empty(
                        tensor_shape,
                        dtype=input_.dtype,
                        device=input_.device,
                    )
                )
        else:
            input_list = [
                tensor.contiguous()
                for tensor in torch.tensor_split(
                    input_, self.world_size, scatter_dim
                )
            ]
            output_list = [
                torch.empty_like(input_list[index])
                for index in range(self.world_size)
            ]

        dist.all_to_all(output_list, input_list, group=self.device_group)
        return torch.cat(output_list, dim=gather_dim).contiguous()
