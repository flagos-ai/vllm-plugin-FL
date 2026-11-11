# SPDX-License-Identifier: Apache-2.0
"""FlagGems ops that must stay on stock torch regardless of configuration.

FlagGems' ``aten::clone`` kernel asserts a CUDA device, but vLLM clones CPU
tensors during engine init (``torch._dynamo.utils.preserve_rng_state`` ->
``torch.clone(torch.random.get_rng_state())``). Ling-mini-2.0 / Ring-mini-2.0
therefore aborted with::

    ValueError: Expected a cuda device, but got: cpu

FlagGems intercepts at the aten dispatch layer, so wrapping ``torch.clone``
after ``flag_gems.enable()`` does not help — the op has to be kept out of the
enable list. On CUDA, clone is only a memory copy, so nothing is lost.
"""

FLAGGEMS_UNUSABLE_OPS = ("clone",)


def exclude_unusable(ops):
    """Return ``ops`` without the ops FlagGems cannot run safely."""
    seen = []
    for op in ops or ():
        if op not in FLAGGEMS_UNUSABLE_OPS and op not in seen:
            seen.append(op)
    return seen


def with_unusable(*lists):
    """Merge op lists and append the always-excluded ops."""
    merged = []
    for ops in (*lists, FLAGGEMS_UNUSABLE_OPS):
        for op in ops or ():
            if op not in merged:
                merged.append(op)
    return merged
