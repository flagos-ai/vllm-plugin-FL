# Copyright (c) 2026 BAAI. All rights reserved.

"""MetaX FlagCX bridge for the native vLLM CUDA graph interface."""

import torch
from torch._subclasses.fake_tensor import FakeTensor

from vllm.platforms import current_platform

from vllm_fl.distributed.nvidia_communicator import NvidiaCommunicatorFL


class MetaxCommunicatorFL(NvidiaCommunicatorFL):
    """Reuse the FlagCX-only CUDA API bridge without initializing PyNCCL.

    MetaX Torch exposes the CUDA API. The native V2 graph coordinator requires
    a CudaCommunicator identity and its capture attributes even though FlagCX
    owns the device collectives. The existing bridge provides that contract.
    """

    # Capture history survives native profile/clear/rebuild. Graph replay does
    # not re-enter all_reduce, so this is not a per-replay completion tracker.
    _flagcx_capture_seen = False

    def _uses_real_flagcx(self, input_: torch.Tensor) -> bool:
        return (
            not torch.compiler.is_compiling()
            and isinstance(input_, torch.Tensor)
            and not isinstance(input_, FakeTensor)
            and input_.device.type == "cuda"
            and input_.device == self.device
            and self.world_size > 1
            and self.pyflagcx_comm is not None
            and not self.pyflagcx_comm.disabled
        )

    def all_reduce(self, input_):
        capturing = (
            self._uses_real_flagcx(input_)
            and current_platform.torch_device_fn.is_current_stream_capturing()
        )
        output = super().all_reduce(input_)
        if capturing:
            self._flagcx_capture_seen = True
        return output

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if (
            self._flagcx_capture_seen
            and self._uses_real_flagcx(input_)
            and not current_platform.torch_device_fn.is_current_stream_capturing()
        ):
            # Complete earlier captured work before the noncaptured collective.
            # Keep this outside low-level AG: all_gatherv groups its submissions.
            self.pyflagcx_comm._current_stream().synchronize()
        return super().all_gather(input_, dim)
