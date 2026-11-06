# Copyright (c) 2026 BAAI. All rights reserved.

"""Backward-compatible NVIDIA entry point for the CUDA FlagCX bridge."""

from vllm_fl.distributed.cuda_communicator import CudaCommunicatorFL


class NvidiaCommunicatorFL(CudaCommunicatorFL):
    """Use the shared CUDA graph-compatible FlagCX implementation."""
