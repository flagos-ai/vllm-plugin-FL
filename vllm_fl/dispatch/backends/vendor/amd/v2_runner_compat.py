# Copyright (c) 2026 BAAI. All rights reserved.

"""AMD compatibility for the upstream vLLM V2 model runner."""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_weak_ref_library: torch.library.Library | None = None


def _weak_ref_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor

    # The graph pool owns this allocation. As in PyTorch's cudagraph_trees,
    # reconstruct a view over a bare pointer without retaining the source storage.
    storage = tensor.untyped_storage()
    construct_storage = torch._C._construct_storage_from_data_pointer
    borrowed = construct_storage(storage.data_ptr(), tensor.device, storage.nbytes())
    result = torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
        borrowed, tensor.storage_offset(), tensor.size(), tensor.stride()
    )
    if tensor.is_conj():
        torch._C._set_conj(result, True)
    if tensor.is_neg():
        torch._C._set_neg(result, True)
    return result


def _register_weak_ref_cuda_kernel() -> None:
    global _weak_ref_library

    # A native ROCm build provides this CUDA-key kernel in _C_stable_libtorch.
    # The EMPTY wheel only has a throwing Composite fallback.
    if torch._C._dispatch_has_kernel_for_dispatch_key("_C::weak_ref_tensor", "CUDA"):
        return
    library = torch.library.Library("_C", "IMPL", "CUDA")
    library.impl("weak_ref_tensor", _weak_ref_tensor)
    # A torch.library registration lives as long as its Library object.
    _weak_ref_library = library


def _no_uva() -> bool:
    return False


def install_v2_runner_compat() -> None:
    """Prepare the AMD worker before constructing the native V2 runner."""
    from vllm.v1.worker.gpu import buffer_utils

    _register_weak_ref_cuda_kernel()
    # The AMD host mapping retains source storage for weight offload. The V2
    # runner's transient pools instead use the upstream explicit-copy buffers.
    buffer_utils.is_uva_available = _no_uva
    logger.info_once("AMD V2 uses explicit-copy input buffers.")
