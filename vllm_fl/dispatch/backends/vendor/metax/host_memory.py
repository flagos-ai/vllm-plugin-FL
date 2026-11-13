# SPDX-License-Identifier: Apache-2.0
"""MetaX host-memory views backed by the optional native vendor extension."""

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _load_extension():
    try:
        from vllm_fl import _metax_host_memory  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "MetaX UVA requires the native extension: build plugin-FL with "
            "VLLM_VENDOR=metax in the matching MetaX Torch/MACA environment"
        ) from error


def get_cuda_view_from_cpu_tensor(tensor):
    """Alias pinned CPU storage on the current GPU; keep its owner alive."""
    _load_extension()
    return torch.ops.vllm_fl_metax.host_view(tensor, torch.cuda.current_device())


def register_host_tensor(tensor):
    """Register one contiguous CPU span, retaining it until the view is freed.

    Register during initialization only. The returned GPU view owns the
    registration. Keep it alive for every graph/stream that uses its address.
    Do not register the same or overlapping span twice or externally unregister it.
    """
    _load_extension()
    return torch.ops.vllm_fl_metax.register_host(tensor, torch.cuda.current_device())
