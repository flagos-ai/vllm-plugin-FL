# SPDX-License-Identifier: Apache-2.0
"""HY4 runtime capability checks and transactional compatibility installation.

The empty-build adapter changes process-global vLLM symbols. It is
installed only by HY4 construction, under a lock, and rolled back on failure.
It is not an instance-local provider and must not be used to hot-swap models
in an already initialized worker.
"""

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType

from vllm.logger import init_logger

logger = init_logger(__name__)

import torch

from vllm_fl.utils import use_flaggems_op

_INSTALL_LOCK = RLock()
_MISSING = object()


from vllm_fl.dispatch.capabilities import has_device_kernel

_FLAGGEMS_INDEXER_OPS = (
    "indexer_k_quant_and_cache",
    "cp_gather_indexer_k_quant_cache",
    "top_k_per_row_prefill",
    "top_k_per_row_decode",
    "fp8_fp4_mqa_logits",
    "fp8_fp4_paged_mqa_logits",
    "flash_mla_sparse_fwd",
)

_INT8_INDEXER_OPS = {
    "indexer_k_quant_and_cache": "indexer_k_quant_and_cache_int8",
    "fp8_fp4_mqa_logits": "int8_mqa_logits",
    "fp8_fp4_paged_mqa_logits": "int8_paged_mqa_logits",
}


def _resolve_registered_operator(name, *, native_available=True):
    """Resolve once through the common registry; execution errors propagate."""
    from vllm_fl.dispatch.manager import get_default_manager
    from vllm_fl.dispatch.types import BackendImplKind

    manager = get_default_manager()
    dependency = (
        " (requires flash_attn_varlen_func)" if name == "mla_prefill_backend" else ""
    )
    try:
        candidates = manager.resolve_candidates(name)
    except RuntimeError as exc:
        raise RuntimeError(
            f"HY4 {name} has no permitted implementation: {exc}"
        ) from exc
    for implementation in candidates:
        if native_available or implementation.kind != BackendImplKind.VENDOR:
            return implementation
    raise RuntimeError(
        f"HY4 {name}{dependency} has no permitted implementation for the model configuration"
    )


def _native_operator_allowed(name):
    # Explicit user-selected native backends remain explicit, subject to policy.
    from vllm_fl.dispatch.policy import get_policy
    from vllm_fl.dispatch.types import BackendImplKind, OpImpl, match_token

    policy = get_policy()
    implementation = OpImpl(
        name, "vendor.cuda", BackendImplKind.VENDOR, lambda: None, vendor="cuda"
    )
    order = policy.get_per_op_order(name) or policy.get_default_order()
    return policy.is_vendor_allowed("cuda") and any(
        match_token(implementation, token) for token in order
    )


def _flaggems_operator_allowed(name):
    from vllm_fl.dispatch.manager import get_default_manager
    from vllm_fl.dispatch.types import BackendImplKind

    if not use_flaggems_op(name):
        return False
    try:
        candidates = get_default_manager().resolve_candidates(name)
    except RuntimeError:
        return False
    return any(impl.kind == BackendImplKind.DEFAULT for impl in candidates)


@dataclass(frozen=True)
class HY4RuntimePlan:
    # Sparse indexer/MLA provider; query, cache and prefill are resolved separately.
    provider: str
    query_quantizer: Callable
    # Only selected replacements; native cache/concat implementations stay put.
    operations: Mapping[str, Callable]
    prefill_backend: type
    kv_cache_dtypes: tuple[str, ...]
    indexer_dtype: torch.dtype = torch.float8_e4m3fn


def _prefill_is_available(backend) -> bool:
    if not backend.is_available():
        return False
    if backend.get_name() == "FLASH_ATTN":
        # Validate the captured global actually invoked by this backend.
        from vllm.v1.attention.backends.mla.prefill import flash_attn

        return callable(getattr(flash_attn, "flash_attn_varlen_func", None))
    return True


def select_hy4_prefill_backend(vllm_config, native_selector):
    """Only automatic absence of an implementation permits a fallback."""
    explicit = vllm_config.attention_config.mla_prefill_backend is not None
    try:
        backend = native_selector(vllm_config)
        if _prefill_is_available(backend):
            return backend
        if explicit:
            raise RuntimeError(
                "Explicit HY4 MLA prefill backend has no callable implementation"
            )
    except (ImportError, OSError):
        if explicit:
            raise
    except ValueError as exc:
        # vLLM 0.24 signals exhausted automatic candidates with this message.
        # Other configuration errors (and AssertionError) must remain visible.
        if explicit or not str(exc).startswith("No valid MLA prefill backend found"):
            raise
    return None


def _resolve_query_quantizer(dtype=torch.float8_e4m3fn):
    # Indexer precision is independent of Linear/MoE checkpoint quantization.
    name = (
        "per_token_group_quant_int8"
        if dtype == torch.int8
        else "per_token_group_quant_fp8"
    )
    return _resolve_registered_operator(name).fn


def prepare_hy4_runtime(vllm_config) -> HY4RuntimePlan:
    plan = validate_hy4_runtime(vllm_config)
    install_hy4_flaggems_fallback(plan)
    logger.info_once(
        "HY4 runtime: sparse_provider=%s, indexer_dtype=%s, query_quantizer=%s, prefill=%s, KV=%s",
        plan.provider,
        plan.indexer_dtype,
        f"{plan.query_quantizer.__module__}.{plan.query_quantizer.__name__}",
        plan.prefill_backend.get_name(),
        vllm_config.cache_config.cache_dtype,
    )
    return plan


def validate_hy4_runtime(vllm_config) -> HY4RuntimePlan:
    """Resolve reachable implementations before allocating model weights."""
    from vllm.platforms import current_platform

    from vllm_fl.dispatch.types import BackendImplKind

    if not current_platform.is_cuda():
        raise ValueError("HY4 currently requires an NVIDIA CUDA runtime")
    fp8 = current_platform.supports_fp8()
    if not fp8 and not current_platform.has_device_capability(80):
        raise ValueError(
            "HY4 INT8 indexer fallback requires NVIDIA SM80 or newer for BF16 execution"
        )
    indexer_dtype = torch.float8_e4m3fn if fp8 else torch.int8
    cache_dtype = vllm_config.cache_config.cache_dtype
    if cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
        raise ValueError(f"Unsupported HY4 KV cache dtype: {cache_dtype}")
    import vllm.model_executor.layers.attention.mla_attention as mla_attention

    config = vllm_config.model_config.hf_text_config
    native_sparse = fp8 and native_hy4_available(config.index_topk)
    selected = {
        name: _resolve_registered_operator(
            name if fp8 else _INT8_INDEXER_OPS.get(name, name),
            native_available=native_sparse,
        )
        for name in _FLAGGEMS_INDEXER_OPS
    }
    native_indexer = all(
        impl.kind == BackendImplKind.VENDOR for impl in selected.values()
    )
    paged_logits = selected["fp8_fp4_paged_mqa_logits"]
    graph_paged_logits = paged_logits.kind == BackendImplKind.VENDOR or (
        getattr(paged_logits.fn, "_supports_cuda_graph", lambda: False)()
    )
    if (
        not native_indexer
        and not graph_paged_logits
        and not getattr(vllm_config.model_config, "enforce_eager", True)
    ):
        raise ValueError(
            "HY4 portable sparse indexer currently requires enforce_eager=True: "
            "the selected paged MQA logits API has no graph-safe entry point; "
            "upgrade the operator library or use enforce_eager=True"
        )
    cache = _resolve_registered_operator("concat_and_cache_mla")
    concat = _resolve_registered_operator("concat_mla_q")
    native_cache = cache.kind == BackendImplKind.VENDOR
    native_concat = concat.kind == BackendImplKind.VENDOR
    native_prefill = select_hy4_prefill_backend(
        vllm_config, mla_attention.get_mla_prefill_backend
    )
    explicit = vllm_config.attention_config.mla_prefill_backend is not None
    if explicit:
        flaggems_explicit = native_prefill.get_name() == "FLAGGEMS_MLA_PREFILL"
        permitted = (
            _flaggems_operator_allowed("mla_prefill_backend")
            if flaggems_explicit
            else _native_operator_allowed("mla_prefill_backend")
        )
        if not permitted:
            raise RuntimeError(
                "Explicit HY4 MLA prefill backend is disabled by dispatch policy"
            )
        prefill_backend = native_prefill
        flaggems_prefill = prefill_backend.get_name() == "FLAGGEMS_MLA_PREFILL"
    else:
        prefill = _resolve_registered_operator(
            "mla_prefill_backend", native_available=native_prefill is not None
        )
        flaggems_prefill = prefill.kind != BackendImplKind.VENDOR
        prefill_backend = native_prefill if not flaggems_prefill else None

    portable = not (
        native_indexer and native_cache and native_concat and not flaggems_prefill
    )
    if portable:
        if str(cache_dtype).startswith("fp8"):
            raise ValueError(
                "HY4 FP8 KV requires a complete native path including MLA prefill"
            )
        if cache_dtype == "auto" and vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("HY4 FlagGems auto KV requires bfloat16 model dtype")

    implementations = (
        {}
        if native_indexer
        else {
            name if fp8 else _INT8_INDEXER_OPS.get(name, name): impl.fn
            for name, impl in selected.items()
        }
    )
    if not native_cache:
        implementations["concat_and_cache_mla"] = cache.fn
    if not native_concat:
        implementations["concat_mla_q"] = concat.fn
    if flaggems_prefill:
        qk_width = config.qk_nope_head_dim + config.qk_rope_head_dim
        if not (0 < config.v_head_dim <= qk_width <= 256):
            raise ValueError(
                "HY4 FlagGems MLA prefill requires 0 < v_head_dim <= "
                f"qk_head_dim <= 256; got v={config.v_head_dim}, qk={qk_width}"
            )
        implementation = _resolve_registered_operator("flash_attn_varlen_func").fn
        implementations["flash_attn_varlen_func"] = implementation
        prefill_backend = _make_hy4_flaggems_mla_prefill_backend(implementation)
    return HY4RuntimePlan(
        "native" if native_indexer else "flaggems",
        _resolve_query_quantizer(indexer_dtype),
        MappingProxyType(implementations),
        prefill_backend,
        ("auto", "bfloat16") if portable else ("auto", "bfloat16", "fp8", "fp8_e4m3"),
        indexer_dtype=indexer_dtype,
    )


def native_hy4_available(index_topk: int) -> bool:
    """Check the native sparse path, including only reachable decode top-k ops."""
    import vllm._custom_ops as ops
    import vllm.model_executor.layers.sparse_attn_indexer as indexer
    from vllm.platforms import current_platform
    from vllm.utils.deep_gemm import has_deep_gemm
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_sparse_fwd,
        is_flashmla_sparse_supported,
    )

    if not (has_deep_gemm() and is_flashmla_sparse_supported()[0]):
        return False
    kernels = [
        "_C_cache_ops::indexer_k_quant_and_cache",
        "_C_cache_ops::cp_gather_indexer_k_quant_cache",
        "_C::top_k_per_row_prefill",
    ]
    implementations = [
        indexer.fp8_fp4_mqa_logits,
        indexer.fp8_fp4_paged_mqa_logits,
        flash_mla_sparse_fwd,
        ops.indexer_k_quant_and_cache,
        ops.cp_gather_indexer_k_quant_cache,
        ops.top_k_per_row_prefill,
    ]
    if index_topk in (512, 1024, 2048):
        kernels.append("_C::persistent_topk")
        # Dynamic H100 batches can reach either persistent or cooperative top-k.
        if current_platform.has_device_capability(90):
            kernels.append("_C::cooperative_topk")
    else:
        kernels.append("_C::top_k_per_row_decode")
        implementations.append(ops.top_k_per_row_decode)
    return all(has_device_kernel(name, "cuda") for name in kernels) and all(
        callable(implementation) for implementation in implementations
    )


class PatchTransaction:
    def __init__(self):
        self.saved = []

    def set(self, target, name, value):
        self.saved.append((target, name, getattr(target, name, _MISSING)))
        setattr(target, name, value)

    def rollback(self):
        for target, name, old in reversed(self.saved):
            if old is _MISSING:
                delattr(target, name)
            else:
                setattr(target, name, old)


@contextmanager
def patch_transaction():
    with _INSTALL_LOCK:
        tx = PatchTransaction()
        try:
            yield tx
        except BaseException:
            tx.rollback()
            raise


def install_hy4_flaggems_fallback(plan: HY4RuntimePlan) -> bool:
    with patch_transaction() as tx:
        return _install_fallback(tx, plan)


def _make_hy4_flaggems_mla_prefill_backend(implementation):
    # Keep backend imports at model construction, after vLLM config initialization.
    from vllm_fl.attention.mla_prefill import make_flaggems_mla_prefill_backend

    return make_flaggems_mla_prefill_backend(implementation)


def _install_fallback(tx, plan: HY4RuntimePlan) -> bool:
    """Install the resolved plan once in a dedicated HY4 worker."""
    import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer

    if getattr(sparse_indexer, "_hy4_runtime_installed", False):
        return bool(plan.operations)
    implementations = plan.operations
    if plan.provider == "flaggems":
        flash_mla_sparse_fwd = implementations["flash_mla_sparse_fwd"]

        # HY4's sink-capable backend imports these functions by value.  Rebind its
        # globals (and the base module for inherited helper paths) to avoid the
        # native FlashMLA/NV kernel when the wheel is empty.
        import vllm.v1.attention.backends.mla.flashmla_sparse as native_sparse

        from . import flashmla_sparse as hy4_sparse

        tx.set(native_sparse, "flash_mla_sparse_fwd", flash_mla_sparse_fwd)
        tx.set(hy4_sparse, "flash_mla_sparse_fwd", flash_mla_sparse_fwd)

        logger.warning_once(
            "HY4: using the resolved FlagGems indexer/top-k and sparse MLA path."
        )

    import vllm._custom_ops as vllm_custom_ops

    for name in ("concat_and_cache_mla", "concat_mla_q"):
        if name in implementations:
            tx.set(vllm_custom_ops, name, implementations[name])

    # Layer constructors consume the prefill choice already checked at the
    # model boundary, including native prefill paired with a portable indexer.
    import vllm.model_executor.layers.attention.mla_attention as mla_attention
    import vllm.v1.attention.backends.mla.prefill as prefill
    import vllm.v1.attention.backends.mla.prefill.selector as prefill_selector

    def get_hy4_prefill_backend(vllm_config):
        return plan.prefill_backend

    tx.set(mla_attention, "get_mla_prefill_backend", get_hy4_prefill_backend)
    tx.set(prefill, "get_mla_prefill_backend", get_hy4_prefill_backend)
    tx.set(prefill_selector, "get_mla_prefill_backend", get_hy4_prefill_backend)
    tx.set(sparse_indexer, "_hy4_runtime_installed", True)
    return bool(implementations)
