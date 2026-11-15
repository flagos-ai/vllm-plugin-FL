"""Compare reverse-rank eager OCR using a fixed logical-rank reduction."""

import functools

import torch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_fl.distributed.communicator import CommunicatorFL


class CanonicalTPReduceExtension:
    """Choose rank-ordered FP32 addition for BF16 TP tensors in eager mode."""


load_model = GPUModelRunner.load_model


@functools.wraps(load_model)
def load_with_canonical_reduce(self, *args, **kwargs):
    original = CommunicatorFL.all_reduce
    reported = set()

    @functools.wraps(original)
    def all_reduce(communicator, tensor):
        if tensor.dtype != torch.bfloat16 or tensor.ndim == 0:
            return original(communicator, tensor)
        if communicator.world_size != 4 or torch.cuda.is_current_stream_capturing():
            return original(communicator, tensor)
        gathered = communicator.all_gather(tensor, dim=0).reshape((4, *tensor.shape))
        accumulator = gathered[0].float()
        for source_rank in range(1, 4):
            accumulator.add_(gathered[source_rank].float())
        if communicator.rank_in_group not in reported:
            reported.add(communicator.rank_in_group)
            print(f"Canonical FP32 TP reduction entered on rank {communicator.rank_in_group}", flush=True)
        return accumulator.to(dtype=tensor.dtype)

    CommunicatorFL.all_reduce = all_reduce
    return load_model(self, *args, **kwargs)


GPUModelRunner.load_model = load_with_canonical_reduce
