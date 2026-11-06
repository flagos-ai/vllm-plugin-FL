# Copyright (c) 2026 BAAI. All rights reserved.

"""Helpers that smooth over vLLM symbols moved or removed between releases.

* vLLM 0.32 dropped the residual-scattered sequence-parallelism path, so
  ``vllm.v1.worker.utils.is_residual_scattered_for_sp`` no longer exists.
  FL workers still decide whether a "residual" tensor is sharded across TP
  ranks: on builds that carry the helper we keep vLLM's decision, on newer
  builds the concept is gone and the answer is always "not scattered".
* Batch-invariant setup moved from ``vllm.model_executor.layers`` to
  ``vllm.model_executor.determinism``.
* The private ``_reshape_attention_kv_cache`` helper was removed together with
  the raw-tensor KV-cache allocation path; FL's model runner still allocates
  raw int8 pages, so a local port of the 0.28 helper is kept here.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

try:
    from vllm.v1.worker.utils import is_residual_scattered_for_sp

    HAS_RESIDUAL_SCATTERED_FOR_SP = True
except ImportError:  # vLLM >= 0.32
    HAS_RESIDUAL_SCATTERED_FOR_SP = False

    def is_residual_scattered_for_sp(vllm_config: Any, num_tokens: int) -> bool:
        return False


__all__ = [
    "HAS_RESIDUAL_SCATTERED_FOR_SP",
    "init_batch_invariance",
    "is_residual_scattered_for_sp",
    "reshape_attention_kv_cache",
]


def init_batch_invariance() -> None:
    """Run vLLM's batch-invariance setup at its new (0.32) or legacy path."""
    try:
        from vllm.model_executor.determinism.batch_invariant import (
            init_batch_invariance as _init,
        )
    except ImportError:  # vLLM < 0.32
        from vllm.model_executor.layers.batch_invariant import (
            init_batch_invariance as _init,
        )

    _init()


try:
    from vllm.v1.worker.gpu.attn_utils import (  # type: ignore[attr-defined]
        _reshape_attention_kv_cache as reshape_attention_kv_cache,
    )

    HAS_ATTENTION_KV_CACHE_RESHAPE = True
except ImportError:  # vLLM >= 0.32
    HAS_ATTENTION_KV_CACHE_RESHAPE = False

    def reshape_attention_kv_cache(
        kv_raw_tensor: Any,
        kv_cache_spec: Any,
        kv_cache_shape: tuple[int, ...],
        kv_cache_stride_order: tuple[int, ...],
        num_blocks: int,
        packing: tuple[int, int] | None,
        page_aligned_blocks: bool = False,
    ) -> Any:
        """Port of vLLM 0.28 ``_reshape_attention_kv_cache``.

        Views the flat int8 KV-cache page buffer as the attention backend's
        strided tensor, honouring packed shared allocations and padded pages.
        """
        import torch
        from math import prod

        from vllm.utils.torch_utils import get_dtype_size

        permuted_kv_cache_shape = tuple(
            kv_cache_shape[i] for i in kv_cache_stride_order
        )
        inv_order = [
            kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
        ]
        dtype = kv_cache_spec.dtype

        if packing is not None:
            offset, block_stride = packing
            assert inv_order[0] == 0
            page_bytes = prod(kv_cache_shape[1:]) * get_dtype_size(dtype)
            kv_cache = (
                kv_raw_tensor.view(-1, block_stride)[:, offset : offset + page_bytes]
                .view(dtype)
                .view(permuted_kv_cache_shape)
            )
        elif kv_cache_spec.page_size_padded is not None:
            # Strided view that skips the padding between physical pages.
            assert kv_cache_shape[0] == num_blocks, (
                "Padded KV pages require a num-blocks-first KV cache layout (got "
                f"shape {kv_cache_shape} with num_blocks={num_blocks}); "
                "kv-first layouts are not supported."
            )
            dtype_size = get_dtype_size(kv_cache_spec.dtype)
            page_stride = kv_cache_spec.page_size_bytes // dtype_size

            num_blocks_dim = inv_order[0]
            strides = list(torch.empty(permuted_kv_cache_shape, device="meta").stride())
            strides[num_blocks_dim] = page_stride

            kv_cache = torch.as_strided(
                kv_raw_tensor.view(dtype),
                size=permuted_kv_cache_shape,
                stride=tuple(strides),
            )
        elif page_aligned_blocks:
            assert kv_cache_shape[1] == num_blocks and kv_cache_stride_order == tuple(
                range(len(kv_cache_shape))
            ), (
                "Page-aligned KV blocks expect a default-strided (kv, num_blocks, "
                f"...) layout, got shape {kv_cache_shape} with stride order "
                f"{kv_cache_stride_order} and num_blocks={num_blocks}."
            )
            kv_cache = (
                kv_raw_tensor.view(dtype)
                .view(num_blocks, kv_cache_shape[0], *kv_cache_shape[2:])
                .transpose(0, 1)
            )
        else:
            # No padding — safe to use a contiguous view.
            kv_cache = kv_raw_tensor.view(dtype).view(permuted_kv_cache_shape)

        return kv_cache.permute(*inv_order)


try:
    from vllm.model_executor.layers.rotary_embedding import XDRotaryEmbedding
    from vllm.model_executor.models.interfaces import (
        SupportsXDRoPE,
        supports_xdrope,
    )

    HAS_XDROPE = True
except ImportError:  # vLLM >= 0.32 removed XD-RoPE
    HAS_XDROPE = False

    class XDRotaryEmbedding:  # type: ignore[no-redef]
        """Placeholder for the XD-RoPE helper removed in vLLM 0.32."""

        @staticmethod
        def get_next_input_positions_tensor(**kwargs: Any) -> None:
            raise NotImplementedError(
                "XD-RoPE is not available in this vLLM build; "
                "set the model to use MRoPE instead."
            )

    SupportsXDRoPE = Any  # type: ignore[assignment,misc]

    def supports_xdrope(model: Any) -> bool:  # type: ignore[misc]
        return False


__all__ += ["HAS_XDROPE", "SupportsXDRoPE", "XDRotaryEmbedding", "supports_xdrope"]


try:
    from vllm.v1.outputs import (
        ModelRunnerOutput as _ModelRunnerOutput,
        RoutedExpertsLists,
        RoutedExpertsTensors,
    )

    OUTPUT_SUPPORTS_ROUTED_EXPERTS = "routed_experts" in {
        _f.name for _f in dataclasses.fields(_ModelRunnerOutput)
    }
except ImportError:  # vLLM >= 0.32 dropped routed-expert capture from the output
    OUTPUT_SUPPORTS_ROUTED_EXPERTS = False

    class RoutedExpertsLists:  # type: ignore[no-redef]
        """Placeholder so legacy annotations still evaluate on py3.10."""

    class RoutedExpertsTensors:  # type: ignore[no-redef]
        """Placeholder so legacy annotations still evaluate on py3.10."""


__all__ += [
    "OUTPUT_SUPPORTS_ROUTED_EXPERTS",
    "RoutedExpertsLists",
    "RoutedExpertsTensors",
]


try:
    from vllm.v1.worker.gpu_input_batch import InputBatch as _InputBatch

    INPUT_BATCH_ACCEPTS_KERNEL_BLOCK_SIZES = "kernel_block_sizes" in (
        inspect.signature(_InputBatch.__init__).parameters
    )
except Exception:  # pragma: no cover - defensive
    INPUT_BATCH_ACCEPTS_KERNEL_BLOCK_SIZES = False


def input_batch_kernel_block_sizes(sizes: list[int]) -> dict[str, list[int]]:
    """Return the ``kernel_block_sizes`` kwarg only for builds that accept it.

    vLLM 0.32 removed the kernel-block-size list from ``InputBatch``.
    """
    return {"kernel_block_sizes": sizes} if INPUT_BATCH_ACCEPTS_KERNEL_BLOCK_SIZES else {}


__all__ += ["input_batch_kernel_block_sizes"]


try:
    from vllm.v1.worker.utils import customize_attention_spec
except ImportError:  # vLLM < 0.32
    def customize_attention_spec(backend: Any, spec: Any) -> Any:
        """Legacy two-step customisation removed in vLLM 0.32."""
        indexes = backend.indexes_kv_by_block_stride()
        spec = dataclasses.replace(spec, indexes_kv_by_block_stride=indexes)
        return backend.customize_spec(spec)


__all__ += ["customize_attention_spec"]


def resolve_cudagraph_mode_and_sizes(
    compilation_config: Any,
    min_cg_support: Any,
    min_cg_attn_backend: Any,
    uniform_decode_query_len: int,
    *,
    tensor_parallel_size: int,
    kv_cache_config: Any,
    max_num_reqs: int | None,
    is_profiling: bool,
) -> Any:
    """Call ``CompilationConfig.resolve_cudagraph_mode_and_sizes`` portably.

    vLLM 0.32 replaced the ``tensor_parallel_size`` argument with
    ``piecewise_capture_available``; pass the legacy argument only when the
    installed build still declares it.
    """
    params = inspect.signature(
        compilation_config.resolve_cudagraph_mode_and_sizes
    ).parameters
    extra = {}
    if "tensor_parallel_size" in params:
        extra["tensor_parallel_size"] = tensor_parallel_size
    return compilation_config.resolve_cudagraph_mode_and_sizes(
        min_cg_support,
        min_cg_attn_backend,
        uniform_decode_query_len,
        use_v2_model_runner=False,
        kv_cache_config=kv_cache_config,
        max_num_reqs=max_num_reqs,
        is_profiling=is_profiling,
        **extra,
    )


__all__ += ["resolve_cudagraph_mode_and_sizes"]


def kv_cache_tensor_layers(kv_cache_tensor: Any) -> list[str]:
    """Layer names that share one KV-cache tensor.

    vLLM 0.32 renamed ``KVCacheTensor.shared_by`` to ``layers``.
    """
    layers = getattr(kv_cache_tensor, "layers", None)
    if layers is None:
        layers = kv_cache_tensor.shared_by
    return list(layers)


__all__ += ["kv_cache_tensor_layers"]


try:
    from vllm.v1.worker.utils import (
        allocate_kv_cache as allocate_kv_cache_032,
        map_kv_caches_to_kernel_blocks as map_kv_caches_to_kernel_blocks_032,
    )

    HAS_KV_CACHE_ALLOCATOR = True
except ImportError:  # vLLM < 0.32: FL reshapes raw allocation itself
    HAS_KV_CACHE_ALLOCATOR = False
    allocate_kv_cache_032 = None  # type: ignore[assignment]
    map_kv_caches_to_kernel_blocks_032 = None  # type: ignore[assignment]


__all__ += [
    "HAS_KV_CACHE_ALLOCATOR",
    "allocate_kv_cache_032",
    "map_kv_caches_to_kernel_blocks_032",
]


def kv_block_zeroer_kwargs(
    kv_block_zeroer_cls: Any, *, cache_dtype: str, num_blocks: int
) -> dict[str, Any]:
    """Build ``KVBlockZeroer`` kwargs for the installed vLLM version.

    vLLM 0.32 dropped ``cache_dtype`` and added ``num_blocks``.
    """
    params = inspect.signature(kv_block_zeroer_cls.__init__).parameters
    kwargs: dict[str, Any] = {}
    if "cache_dtype" in params:
        kwargs["cache_dtype"] = cache_dtype
    if "num_blocks" in params:
        kwargs["num_blocks"] = num_blocks
    return kwargs


__all__ += ["kv_block_zeroer_kwargs"]


try:
    from vllm.v1.attention.backend import (
        CommonAttentionMetadata as _CommonAttentionMetadata,
    )

    _COMMON_ATTN_METADATA_FIELDS = {
        _f.name for _f in dataclasses.fields(_CommonAttentionMetadata)
    }
except Exception:  # pragma: no cover - defensive
    _COMMON_ATTN_METADATA_FIELDS = set()


def common_attention_metadata_cpu_kwargs(
    *, seq_lens_cpu: Any, num_computed_tokens_cpu: Any
) -> dict[str, Any]:
    """Optional private CPU-side fields of ``CommonAttentionMetadata``.

    vLLM 0.32 replaced ``_seq_lens_cpu`` / ``_num_computed_tokens_cpu`` with
    ``seq_lens_cpu_upper_bound`` and per-runner caches, so pass the legacy
    fields only when the installed dataclass still declares them.
    """
    kwargs: dict[str, Any] = {}
    if "_seq_lens_cpu" in _COMMON_ATTN_METADATA_FIELDS:
        kwargs["_seq_lens_cpu"] = seq_lens_cpu
    if "_num_computed_tokens_cpu" in _COMMON_ATTN_METADATA_FIELDS:
        kwargs["_num_computed_tokens_cpu"] = num_computed_tokens_cpu
    return kwargs


__all__ += ["common_attention_metadata_cpu_kwargs"]
