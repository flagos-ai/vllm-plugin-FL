# Copyright (c) 2026 BAAI. All rights reserved.
"""GCU V2 logprob top-k compatibility, outside model graph replay."""

import types

import torch


class _LogprobTorch:
    """Delegate everything except GCU topk to the original torch module."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def topk(input, *args, **kwargs):
        if input.device.type != "gcu":
            return torch.topk(input, *args, **kwargs)
        result = torch.topk(input.detach().cpu(), *args, **kwargs)
        return torch.return_types.topk(
            (result.values.to(input.device), result.indices.to(input.device))
        )


def apply_logprob_gcu_patch() -> bool:
    from vllm.v1.worker.gpu.sample import logprob, sampler

    original = logprob.compute_topk_scores
    if getattr(original, "_vllm_fl_gcu_logprob", False):
        return False
    globals_ = dict(original.__globals__, torch=_LogprobTorch())
    patched = types.FunctionType(
        original.__code__,
        globals_,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    patched.__kwdefaults__ = original.__kwdefaults__
    patched.__annotations__ = original.__annotations__
    patched.__doc__ = original.__doc__
    patched._vllm_fl_gcu_logprob = True
    logprob.compute_topk_scores = patched
    # sampler imports this function by value, so update its reference too.
    sampler.compute_topk_scores = patched
    return True
