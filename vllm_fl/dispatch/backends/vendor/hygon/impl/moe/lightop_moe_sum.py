# Copyright (c) 2026 BAAI. All rights reserved.

"""Reuse FlagGems MoE sum inside the Hygon LightOp Marlin call only."""

import functools
import logging
import os
import types

import torch

logger = logging.getLogger(__name__)


class _LightOpWithFlagGemsSum:
    def __init__(self, original):
        self.original = original
        self.prefill_sum_enabled = os.getenv(
            "VLLM_FL_HYGON_PREFILL_MOE_SUM", "1"
        ).lower() not in ("0", "false", "off", "no")

    def __getattr__(self, name):
        return getattr(self.original, name)

    def moe_sum(
        self, input, output, bias=None, expert_mask=None,
        num_local_tokens=None, factor=1.0, expect_m=-1,
    ):
        # Both fast paths require the plain sum ABI and contiguous DV4 shapes.
        if (
            bias is None and expert_mask is None and num_local_tokens is None
            and factor == 1.0 and expect_m == -1
            and input.ndim == 3 and input.shape[0] >= 1
            and input.shape[1:] == (6, 4096)
            and output.shape == (input.shape[0], 4096)
            and input.dtype == output.dtype == torch.bfloat16
            and input.is_cuda and output.device == input.device
            and input.is_contiguous() and output.is_contiguous()
            and input.untyped_storage().data_ptr()
            != output.untyped_storage().data_ptr()
        ):
            if input.shape[0] <= 256:
                from flag_gems import moe_sum

                return moe_sum(input, output)
            if self.prefill_sum_enabled:
                from .prefill_moe_sum import prefill_moe_sum

                return prefill_moe_sum(input, output)
        return self.original.moe_sum(
            input, output, bias, expert_mask, num_local_tokens, factor, expect_m
        )


@functools.lru_cache(maxsize=4)
def _with_flaggems_sum(original):
    if not isinstance(original, types.FunctionType):
        logger.warning("LightOp MoE is not a Python function; retaining native sum")
        return original
    op = original.__globals__.get("op")
    if op is None or not hasattr(op, "moe_sum"):
        logger.warning("LightOp MoE has no op.moe_sum binding; retaining native sum")
        return original
    # LightOp exposes no reduction callback. Give this function a private op
    # binding; never mutate lightop.op or the original function's globals.
    bindings = dict(original.__globals__)
    bindings["op"] = _LightOpWithFlagGemsSum(op)
    adapted = types.FunctionType(
        original.__code__, bindings, original.__name__,
        original.__defaults__, original.__closure__,
    )
    adapted.__kwdefaults__ = original.__kwdefaults__
    functools.update_wrapper(adapted, original)
    return adapted


def adapt_lightop_moe_sum(original):
    backend = os.getenv("VLLM_FL_HYGON_LIGHTOP_MOE_SUM", "flaggems").lower()
    if backend in ("native", "0", "off"):
        return original
    if backend != "flaggems":
        raise ValueError(
            "VLLM_FL_HYGON_LIGHTOP_MOE_SUM must be 'flaggems' or 'native'"
        )
    return _with_flaggems_sum(original)
