# SPDX-License-Identifier: Apache-2.0
"""Local launch overrides: never replace the kernel exported by an operator library."""

import triton

from vllm_fl.patches.triton_kernel import KernelLaunchMetaProxy


def _jit_function(kernel):
    """Bind known metadata after unwrapping autotuners, without mutating them.

    Passing duplicate metadata through an existing autotuner can conflict with
    its own Config. This handle is owned by the DS path; generic callers retain
    the original library kernel and tuner.
    """
    seen = set()
    while not isinstance(kernel, triton.runtime.JITFunction):
        if id(kernel) in seen:
            raise TypeError("Cyclic or unsupported Triton wrapper")
        seen.add(id(kernel))
        kernel = getattr(kernel, "fn", None)
        if kernel is None:
            raise TypeError("Expected a JITFunction or a wrapper exposing fn")
    return kernel


def local_launch(kernel, overrides):
    return KernelLaunchMetaProxy(_jit_function(kernel), overrides)


def local_autotune(kernel, **options):
    """Rebuild only the local tuner with the accepted configuration set."""
    return triton.autotune(**options)(_jit_function(kernel))
