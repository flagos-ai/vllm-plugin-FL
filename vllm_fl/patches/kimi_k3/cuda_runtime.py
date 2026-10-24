"""K3 library resolution around the existing MetaX CUDA-runtime ABI wrapper."""
import vllm.envs as envs
from vllm_fl.dispatch.backends.vendor.metax.patches.cuda_wrapper import (
    CudaRTLibrary as NativeCudaRTLibrary,
    find_loaded_library,
)


class CudaRTLibrary(NativeCudaRTLibrary):
    def __init__(self, so_file: str | None = None):
        if so_file is None:
            so_file = envs.VLLM_CUDART_SO_PATH or find_loaded_library("libmcruntime")
            if so_file is None:
                raise RuntimeError(
                    "MACA libmcruntime is not loaded; set VLLM_CUDART_SO_PATH "
                    "to the verified MACA runtime library"
                )
        super().__init__(so_file)
