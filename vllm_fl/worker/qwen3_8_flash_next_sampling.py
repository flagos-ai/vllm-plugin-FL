# SPDX-License-Identifier: Apache-2.0
"""CPU sampling metadata contract for Qwen top-20 routing."""

from functools import wraps


def install_sampling_contract(batch):
    if getattr(batch, "_qwen38_sampling_contract", False):
        return
    original = getattr(batch, "_make_sampling_metadata", None)
    if not callable(original):
        return

    def tag(metadata):
        if metadata is not None and metadata.top_k is not None:
            n = batch.num_reqs
            # Existing CPU mirror; no CUDA value read or synchronization.
            metadata.top_k._fl_all_topk20 = bool(
                (batch.top_k_cpu_tensor[:n] == 20).all().item()
            )
        return metadata

    @wraps(original)
    def make_metadata(*args, **kwargs):
        return tag(original(*args, **kwargs))

    batch._make_sampling_metadata = make_metadata
    batch._qwen38_sampling_contract = True
    tag(getattr(batch, "sampling_metadata", None))
