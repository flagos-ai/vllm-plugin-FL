# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/distributed/device_communicators/cuda_communicator.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator

logger = init_logger(__name__)


class CommunicatorFL(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        tcp_store_group: StatelessProcessGroup | None = None,
        use_all2all: bool = False,
    ):
        # Call the common base directly. NVIDIA adds a CudaCommunicator-compatible
        # bridge for vLLM graph capture, but must not initialize PyNCCL as well as
        # FlagCX through cooperative multiple inheritance.
        DeviceCommunicatorBase.__init__(
            self,
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks,
            global_world_size,
            use_all2all=use_all2all,
        )
        self.pyflagcx_comm: Optional[PyFlagcxCommunicator] = None
        if self.world_size > 1:
            self.pyflagcx_comm = PyFlagcxCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

        if self.use_all2all:
            self._init_all2all_manager(tcp_store_group)

    def _init_all2all_manager(
        self, tcp_store_group: StatelessProcessGroup | None
    ) -> None:
        """Initialize the vLLM 0.28 manager selected by ParallelConfig."""
        if self.all2all_backend in ("naive", "allgather_reducescatter"):
            from vllm.distributed.device_communicators.all2all import (
                AgRsAll2AllManager,
            )

            self.all2all_manager = AgRsAll2AllManager(self.cpu_group, tcp_store_group)
        elif self.all2all_backend == "deepep_high_throughput":
            from vllm.distributed.device_communicators.all2all import (
                DeepEPHTAll2AllManager,
            )

            self.all2all_manager = DeepEPHTAll2AllManager(
                self.cpu_group, tcp_store_group
            )
        elif self.all2all_backend == "deepep_low_latency":
            from vllm.distributed.device_communicators.all2all import (
                DeepEPLLAll2AllManager,
            )

            self.all2all_manager = DeepEPLLAll2AllManager(
                self.cpu_group, tcp_store_group
            )
        elif self.all2all_backend in (
            "mori_high_throughput",
            "mori_low_latency",
        ):
            from vllm.distributed.device_communicators.all2all import (
                MoriAll2AllManager,
            )

            self.all2all_manager = MoriAll2AllManager(
                self.cpu_group, self.all2all_backend
            )
        elif self.all2all_backend == "deepep_v2":
            from vllm.distributed.device_communicators.all2all import (
                DeepEPV2All2AllManager,
            )

            self.all2all_manager = DeepEPV2All2AllManager(
                self.cpu_group,
                tcp_store_group,
                device_group=self.device_group,
            )
        elif self.all2all_backend == "nixl_ep":
            from vllm.distributed.device_communicators.all2all import (
                NixlEPAll2AllManager,
            )

            self.all2all_manager = NixlEPAll2AllManager(self.cpu_group, tcp_store_group)
        elif self.all2all_backend in (
            "flashinfer_all2allv",
            "flashinfer_nvlink_two_sided",
        ):
            if self.all2all_backend == "flashinfer_all2allv":
                logger.warning_once(
                    "'flashinfer_all2allv' is deprecated and has been renamed "
                    "to 'flashinfer_nvlink_two_sided'. It will be removed in "
                    "a future release."
                )
            from vllm.distributed.device_communicators.all2all import (
                FlashInferNVLinkTwoSidedManager,
            )

            self.all2all_manager = FlashInferNVLinkTwoSidedManager(
                self.cpu_group, tcp_store_group
            )
        elif self.all2all_backend == "flashinfer_nvlink_one_sided":
            from vllm.distributed.device_communicators.all2all import (
                FlashInferNVLinkOneSidedManager,
            )

            self.all2all_manager = FlashInferNVLinkOneSidedManager(self.cpu_group)
        else:
            raise ValueError(f"Unknown all2all backend: {self.all2all_backend}")

        logger.info_once(
            "Using %s all2all manager.",
            self.all2all_manager.__class__.__name__,
            scope="global",
        )

    def all_reduce(self, input_):
        assert self.pyflagcx_comm is not None
        out = self.pyflagcx_comm.all_reduce(input_)
        if out is None:
            # fall back to the default all-reduce using PyTorch.
            # this usually happens during testing.
            # when we run the model, allreduce only happens for the TP
            # group, where we always have either custom allreduce or pynccl.
            out = input_.clone()
            torch.distributed.all_reduce(out, group=self.device_group)
        return out

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if dim < 0:
            dim += input_.dim()

        pyflagcx_comm = self.pyflagcx_comm
        if pyflagcx_comm is None or pyflagcx_comm.disabled:
            return DeviceCommunicatorBase.all_gather(self, input_, dim)

        input_size = input_.size()
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        pyflagcx_comm.all_gather(output_tensor, input_.contiguous())
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        return output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
        world_size = self.world_size
        pyflagcx_comm = self.pyflagcx_comm
        assert pyflagcx_comm is not None
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size, ) + input_tensor.shape[1:]

        output = torch.empty(output_shape,
                             dtype=input_tensor.dtype,
                             device=input_tensor.device)

        pyflagcx_comm.reduce_scatter(output, input_tensor)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def reduce_scatterv(self,
                        input_: torch.Tensor,
                        dim: int = -1,
                        sizes: Optional[list[int]] = None):
        world_size = self.world_size
        pyflagcx_comm = self.pyflagcx_comm
        assert pyflagcx_comm is not None
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size, ) + input_tensor.shape[1:]

        output = torch.empty(output_shape,
                             dtype=input_tensor.dtype,
                             device=input_tensor.device)

        if sizes is not None:
            pyflagcx_comm.reduce_scatterv(output, input_tensor, sizes=sizes)
        else:
            pyflagcx_comm.reduce_scatter(output, input_tensor)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size

        pyflagcx_comm = self.pyflagcx_comm
        if pyflagcx_comm is not None and not pyflagcx_comm.disabled:
            pyflagcx_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(self,
             size: torch.Size,
             dtype: torch.dtype,
             src: Optional[int] = None) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pyflagcx_comm = self.pyflagcx_comm
        if pyflagcx_comm is not None and not pyflagcx_comm.disabled:
            pyflagcx_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.world_size == 1:
            return tensor

        pyflagcx_comm = self.pyflagcx_comm
        if pyflagcx_comm is not None and not pyflagcx_comm.disabled:
            pyflagcx_comm.broadcast(tensor, src)
            return tensor
        return DeviceCommunicatorBase.broadcast(self, tensor, src)

    def destroy(self):
        if self.pyflagcx_comm is not None:
            self.pyflagcx_comm = None
        if self.all2all_manager is not None:
            self.all2all_manager.destroy()
            self.all2all_manager = None

    def checkpoint_prepare(self) -> None:
        if self.all2all_manager is not None:
            self.all2all_manager.checkpoint_prepare()

    def checkpoint_restore(self) -> None:
        if self.all2all_manager is not None:
            self.all2all_manager.checkpoint_restore()

    def all_gatherv(self,
                    input_: Union[torch.Tensor, list[torch.Tensor]],
                    dim: int = 0,
                    sizes: Optional[list[int]] = None):
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size
        pyflagcx_comm = self.pyflagcx_comm
        assert pyflagcx_comm is not None and not pyflagcx_comm.disabled

        # 'sizes' is not needed if all inputs in the same group have the same
        # shape
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        def _all_gather_single(input_: torch.Tensor,
                               sizes: Optional[list[int]] = None):
            input_size = input_.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert input_.shape[dim] == sizes[self.rank_in_group], (
                    f"{input_.shape[dim]} != {sizes[self.rank_in_group]}")
                output_size = (sum(sizes), ) + input_size[1:]
            else:
                output_size = (input_size[0] * world_size, ) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(output_size,
                                        dtype=input_.dtype,
                                        device=input_.device)
            if sizes is not None:
                pyflagcx_comm.all_gatherv(output_tensor, input_, sizes=sizes)
            else:
                pyflagcx_comm.all_gather(output_tensor, input_)
            return output_tensor

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        output_list = []
        pyflagcx_comm.group_start()
        for inp in input_:
            output_list.append(_all_gather_single(inp, sizes=sizes))
        pyflagcx_comm.group_end()

        return output_list

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
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
    ):
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors,
        )

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        assert self.all2all_manager is not None
        return self.all2all_manager.combine(hidden_states, is_sequence_parallel)
