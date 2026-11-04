# Copyright (c) 2026 BAAI. All rights reserved.

"""MetaX FlagCX bridge for the native vLLM CUDA graph interface."""

from vllm_fl.distributed.nvidia_communicator import NvidiaCommunicatorFL


class MetaxCommunicatorFL(NvidiaCommunicatorFL):
    """Reuse the FlagCX-only CUDA API bridge without initializing PyNCCL.

    MetaX Torch exposes the CUDA API. The native V2 graph coordinator requires
    a CudaCommunicator identity and its capture attributes even though FlagCX
    owns the device collectives. The existing bridge provides that contract;
    all collective methods and initialization still come from CommunicatorFL.
    """
