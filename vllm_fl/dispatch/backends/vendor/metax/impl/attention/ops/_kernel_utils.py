# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""Distinguish executable MACA operators from empty-wheel schemas."""

import torch


def has_cuda_or_composite_kernel(qualified_name: str) -> bool:
    """Schemas and Meta kernels alone cannot execute on a MACA CUDA tensor."""
    for key in (
        "CUDA",
        "CompositeImplicitAutograd",
        "CompositeExplicitAutograd",
        "CompositeExplicitAutogradNonFunctional",
    ):
        try:
            if torch._C._dispatch_has_kernel_for_dispatch_key(qualified_name, key):
                return True
        except RuntimeError:
            # An optional op can be absent entirely. Old Torch builds may
            # also lack one of the composite keys; keep checking other keys.
            continue
    return False
