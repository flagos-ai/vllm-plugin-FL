"""K3 MCCL additions around the existing MetaX ABI wrapper.

Inherited collectives keep the same ctypes signatures and implementation bodies.
Separate caches preserve K3's validated all-symbols-present initialization.
"""
import ctypes
from typing import Any
from vllm.distributed.device_communicators.pynccl_wrapper import (
    Function, buffer_type, ncclComm_t, ncclResult_t, ncclWindow_t,
)
from vllm_fl.dispatch.backends.vendor.metax.patches.pynccl_wrapper import (
    MCCLLibrary as NativeMCCLLibrary,
)
from .runtime_library import find_mccl_library


class MCCLLibrary(NativeMCCLLibrary):
    _group_index = next(
        i for i, fn in enumerate(NativeMCCLLibrary.exported_functions)
        if fn.name == "mcclGroupStart"
    )
    exported_functions = [
        *NativeMCCLLibrary.exported_functions[:_group_index],
        Function("mcclCommAbort", ncclResult_t, [ncclComm_t]),
        *NativeMCCLLibrary.exported_functions[_group_index:],
    ]
    path_to_library_cache: dict[str, Any] = {}
    path_to_dict_mapping: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: str | None = None):
        so_file = so_file or find_mccl_library()
        if so_file not in MCCLLibrary.path_to_library_cache:
            MCCLLibrary.path_to_library_cache[so_file] = ctypes.CDLL(so_file)
        self.lib = MCCLLibrary.path_to_library_cache[so_file]
        if so_file not in MCCLLibrary.path_to_dict_mapping:
            funcs: dict[str, Any] = {}
            for func in MCCLLibrary.exported_functions:
                entry = getattr(self.lib, func.name)
                entry.restype = func.restype
                entry.argtypes = func.argtypes
                funcs[func.name] = entry
            MCCLLibrary.path_to_dict_mapping[so_file] = funcs
        self._funcs = MCCLLibrary.path_to_dict_mapping[so_file]


    def ncclCommAbort(self, comm: ncclComm_t) -> None:
        self.NCCL_CHECK(self._funcs["mcclCommAbort"](comm))


    def ncclCommWindowRegister(
        self, comm: ncclComm_t, buff: buffer_type, size: int, win_flags: int
    ) -> ncclWindow_t:
        raise NotImplementedError("MCCL symmetric-memory windows are not supported")


    def ncclCommWindowDeregister(self, comm: ncclComm_t, window: ncclWindow_t) -> None:
        raise NotImplementedError("MCCL symmetric-memory windows are not supported")
