# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import triton


def launch_handle(kernel):
    """Read the JIT object under decorators for explicit per-call launch metadata."""
    while not isinstance(kernel, triton.JITFunction):
        kernel = kernel.fn
    return kernel
