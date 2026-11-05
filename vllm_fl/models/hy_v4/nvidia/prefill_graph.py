# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native mixed prefill graphs with private FULL draft graph owners.

No projection, attention, routing, cache or sampling arithmetic is changed.
Runtime-mode dispatch is explicit so the native private FULL wrappers never
nest a breakable capture. Non-PIECEWISE native drafter fallbacks remain eager.
"""

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphWrapper,
    is_breakable_cudagraph_enabled,
)
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available


def supports_mixed_config(config):
    c, s, p = (
        config.compilation_config,
        config.speculative_config,
        config.parallel_config,
    )
    return not (
        not is_breakable_cudagraph_enabled()
        or c.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE
        or getattr(c.mode, "name", None) != "NONE"
        or (c.max_cudagraph_capture_size, tuple(c.cudagraph_capture_sizes))
        not in (
            (1024, (4, 8, 16, 32, 64, 128, 256, 512, 1024)),
            (2048, (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)),
        )
        or c.cudagraph_num_of_warmups < 1
        or s is None
        or s.num_speculative_tokens != 3
        or p.pipeline_parallel_size != 1
        or p.data_parallel_size != 1
        or p.decode_context_parallel_size != 1
        or p.prefill_context_parallel_size != 1
        or p.use_ubatching
        or config.lora_config is not None
    )


def validate_mixed_config(config):
    if not supports_mixed_config(config):
        raise ValueError("unsupported HYV4 native prefill graph configuration")
    # Private draft capacities are independent of prefill graph capture sizes.
    return (1, 2, 4, 8, 16, 32, 64, 128, 256)


class HYV4MixedGraphWrapper(BreakableCUDAGraphWrapper):
    """Native FULL target plus native breakable PIECEWISE on the same model."""

    def __init__(self, runnable, vllm_config):
        validate_mixed_config(vllm_config)
        self.full_wrapper = CUDAGraphWrapper(
            runnable, vllm_config, runtime_mode=CUDAGraphMode.FULL
        )
        super().__init__(runnable, vllm_config)

    def __call__(self, *args, **kwargs):
        if (
            is_forward_context_available()
            and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
        ):
            return self.full_wrapper(*args, **kwargs)
        return super().__call__(*args, **kwargs)

    def clear_graphs(self):
        super().clear_graphs()
        self.full_wrapper.clear_graphs()


class HYV4DraftPiecewiseGraphWrapper(BreakableCUDAGraphWrapper):
    """FULL belongs to the existing private manager; PIECEWISE to native BCG."""

    def __init__(self, runnable, vllm_config):
        validate_mixed_config(vllm_config)
        super().__init__(runnable, vllm_config)

    def __call__(self, *args, **kwargs):
        if (
            not is_forward_context_available()
            or get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.PIECEWISE
        ):
            return self.runnable(*args, **kwargs)
        return super().__call__(*args, **kwargs)


def wrap_native_breakable(runnable, vllm_config, *, draft=False):
    wrapper = HYV4DraftPiecewiseGraphWrapper if draft else HYV4MixedGraphWrapper
    return wrapper(runnable, vllm_config)
