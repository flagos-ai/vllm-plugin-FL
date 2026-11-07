"""CUDA-compatible FlagCX communicator for vLLM graph capture."""

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from vllm_fl.distributed.communicator import CommunicatorFL


class CudaCommunicatorFL(CommunicatorFL, CudaCommunicator):
    """Satisfy CUDA graph contracts without initializing a second comm stack."""

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
        self.use_custom_allreduce = False
        self.use_torch_symm_mem = False
        self.use_flashinfer_allreduce = False
        self.use_flashinfer_pcie_ipc_allreduce = False
        self.use_aiter_allreduce = False
        self.use_aiter_ag_rs = False
        self.pynccl_comm = None
        self.ca_comm = None
        self.qr_comm = None
        self.symm_mem_comm = None
        self.fi_ar_comm = None
        self.fi_pcie_ipc_ar_comm = None
        self.aiter_ar_comm = None
