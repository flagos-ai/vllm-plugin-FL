# Copyright (c) 2026 BAAI. All rights reserved.

"""NVIDIA-specific FlagCX communicator bridge for vLLM 0.28."""

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.cuda_communicator import (
    CudaCommunicator,
)
from vllm.distributed.utils import StatelessProcessGroup

from vllm_fl.distributed.communicator import CommunicatorFL


class NvidiaCommunicatorFL(CommunicatorFL, CudaCommunicator):
    """Use FlagCX while satisfying vLLM's CUDA graph communicator contract.

    ``GroupCoordinator.graph_capture`` checks for ``CudaCommunicator`` and
    reads ``ca_comm``. The generic FL communicator remains device-agnostic;
    this NVIDIA-only bridge supplies the CUDA identity without initializing a
    second PyNCCL/custom-all-reduce stack alongside FlagCX.
    """

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
        super().__init__(
            cpu_group=cpu_group,
            device=device,
            device_group=device_group,
            unique_name=unique_name,
            global_ranks=global_ranks,
            global_world_size=global_world_size,
            tcp_store_group=tcp_store_group,
            use_all2all=use_all2all,
        )

        # Attributes used by vLLM's CudaCommunicator methods and graph-capture
        # context. They remain disabled because FlagCX owns communication.
        self.use_custom_allreduce = False
        self.use_torch_symm_mem = False
        self.use_flashinfer_allreduce = False
        self.use_aiter_allreduce = False
        self.pynccl_comm = None
        self.ca_comm = None
        self.qr_comm = None
        self.symm_mem_comm = None
        self.fi_ar_comm = None
        self.aiter_ar_comm = None
