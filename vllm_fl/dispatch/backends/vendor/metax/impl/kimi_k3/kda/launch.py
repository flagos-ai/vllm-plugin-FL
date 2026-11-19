# SPDX-License-Identifier: Apache-2.0
"""K3-local launch bindings for the pinned vLLM KDA kernels."""

from vllm.triton_utils import triton


def local_kernel(native, *, heuristics, **autotune_options):
    """Keep native JIT bodies; own only K3 tuning and dot precision.

    The pinned kernels expose DEFAULT_DOT_PRECISION for their otherwise implicit dots.
    Explicit precision arguments inside the upstream kernel retain precedence.
    Neither the upstream wrappers nor their configuration lists are modified.
    """
    seen = set()
    kernel = native
    while not isinstance(kernel, triton.runtime.JITFunction):
        if not isinstance(
            kernel, (triton.runtime.Heuristics, triton.runtime.Autotuner)
        ):
            raise TypeError("Unsupported wrapper around the pinned vLLM KDA kernel")
        if id(kernel) in seen:
            raise TypeError("Cyclic KDA kernel wrappers")
        seen.add(id(kernel))
        kernel = kernel.fn
    if "DEFAULT_DOT_PRECISION" not in kernel.arg_names:
        raise TypeError("K3 requires the pinned vLLM KDA precision interface")
    tuned = triton.autotune(**autotune_options)(kernel)
    return triton.heuristics(heuristics)(tuned)
