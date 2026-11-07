"""CUDA-compatible FlagCX communicator for vLLM graph capture."""

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

from vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision import in_vision_forward
from vllm_fl.distributed.communicator import CommunicatorFL

logger = init_logger(__name__)


class IluvatarCommunicatorFL(CommunicatorFL, CudaCommunicator):
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
        if self.pyflagcx_comm is not None and self.pyflagcx_comm.disabled:
            raise RuntimeError(
                f"FlagCX requested on {self.device} but failed to initialize"
            ) from self.pyflagcx_comm.init_error
        if self.pyflagcx_comm is not None:
            logger.info(
                "Iluvatar FlagCX device communicator active: %s ranks on %s",
                self.world_size,
                self.device,
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

    def all_reduce(self, input_):
        original_dtype = input_.dtype
        if in_vision_forward() and original_dtype in (torch.bfloat16, torch.float16):
            input_ = input_.float()
        output = super().all_reduce(input_)
        return output.to(original_dtype) if output.dtype != original_dtype else output
