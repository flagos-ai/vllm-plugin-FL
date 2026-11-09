# SPDX-License-Identifier: Apache-2.0
"""Reuse vLLM NCCL collectives without loading NVIDIA-only communicators."""

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


class NcclCommunicatorFL(CudaCommunicator):
    """THead uses its NCCL-compatible library through the existing PyNccl API.

    Upstream CudaCommunicator imports FlashInfer even with its all-reduce
    disabled. Its CUDA runtime initialization is not supported on THead.
    Inherit the collective methods and initialize only the supported transport.
    """

    def __init__(
        self,
        cpu_group,
        device=None,
        device_group=None,
        unique_name="",
        global_ranks=None,
        global_world_size=None,
        tcp_store_group=None,
    ):
        DeviceCommunicatorBase.__init__(
            self,
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks,
            global_world_size,
        )
        self.use_custom_allreduce = False
        self.use_torch_symm_mem = False
        self.use_flashinfer_allreduce = False
        self.ca_comm = self.qr_comm = self.symm_mem_comm = self.fi_ar_comm = None
        self.pynccl_comm = None
        if self.world_size > 1:
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group if tcp_store_group is None else tcp_store_group,
                device=self.device,
            )
        if self.use_all2all:
            if self.all2all_backend not in ("naive", "allgather_reducescatter"):
                raise NotImplementedError(
                    f"THead NCCL communicator does not support {self.all2all_backend}"
                )
            from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager

            self.all2all_manager = AgRsAll2AllManager(self.cpu_group, tcp_store_group)
