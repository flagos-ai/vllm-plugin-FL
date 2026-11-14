# Copyright (c) 2026 BAAI. All rights reserved.

"""NVIDIA FlagCX communicator entry point."""

from vllm_fl.distributed.cuda_flagcx_communicator import CudaFlagcxCommunicatorFL


class NvidiaCommunicatorFL(CudaFlagcxCommunicatorFL):
    """Preserve the NVIDIA platform's communicator class path."""
