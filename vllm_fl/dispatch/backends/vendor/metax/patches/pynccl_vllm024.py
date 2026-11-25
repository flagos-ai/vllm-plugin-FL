# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""vLLM 0.24 abort API on the existing FL MCCL wrapper."""

from vllm.distributed.device_communicators import pynccl, pynccl_wrapper

from vllm_fl.dispatch.backends.vendor.metax.patches.pynccl_wrapper import (
    MCCLLibrary as _MCCLLibrary,
    ncclComm_t,
    ncclResult_t,
)


class MCCLLibrary(_MCCLLibrary):
    def __init__(self, so_file=None):
        super().__init__(so_file)
        self._abort = self.lib.mcclCommAbort
        self._abort.restype = ncclResult_t
        self._abort.argtypes = [ncclComm_t]

    def ncclCommAbort(self, comm: ncclComm_t) -> None:
        self.NCCL_CHECK(self._abort(comm))


pynccl.NCCLLibrary = MCCLLibrary
pynccl_wrapper.NCCLLibrary = MCCLLibrary
