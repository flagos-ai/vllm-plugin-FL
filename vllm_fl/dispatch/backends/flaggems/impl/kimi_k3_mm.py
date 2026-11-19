"""Version-pinned bridge to FlagGems ad9f113's selected MetaX MM backend.

FlagGems itself loads this backend under the canonical _metax module name.
Keep that identity; importing a second qualified path would duplicate caches.
"""

import flag_gems  # Initializes the library's own backend loader once.
import triton
from _metax.ops.mm import (
    mm_kernel_nt,
    mm_kernel,
    mm_kernel_splitk_partial,
    mm_kernel_splitk_reduce,
    _select_two_step_split_k,
    _generic_two_step_split_k,
    splitk_mm_scenario,
    nt_mm_scenario,
    _prefer_dense_nt_over_generic_splitk,
)


def raw_kernel(kernel):
    seen = set()
    while not isinstance(kernel, triton.runtime.jit.JITFunction):
        if id(kernel) in seen or not hasattr(kernel, "fn"):
            raise RuntimeError(
                "Unsupported or cyclic FlagGems MM launch-wrapper layout"
            )
        seen.add(id(kernel))
        kernel = kernel.fn
    return kernel


NT_KERNEL = raw_kernel(mm_kernel_nt)
MM_KERNEL = raw_kernel(mm_kernel)
PARTIAL_KERNEL = raw_kernel(mm_kernel_splitk_partial)
REDUCE_KERNEL = raw_kernel(mm_kernel_splitk_reduce)
