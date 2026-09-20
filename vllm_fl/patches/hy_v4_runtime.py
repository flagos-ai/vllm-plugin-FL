# SPDX-License-Identifier: Apache-2.0
"""HY4 runtime capability checks and transactional compatibility installation.

The empty-build adapter changes process-global vLLM/FlagGems symbols. It is
installed only by HY4 construction, under a lock, and rolled back on failure.
It is not an instance-local provider and must not be used to hot-swap models
in an already initialized worker.
"""

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from threading import RLock
from types import MappingProxyType, SimpleNamespace

from vllm.logger import init_logger

logger = init_logger(__name__)

import torch

from vllm_fl.utils import use_flaggems_op

_INSTALL_LOCK = RLock()
_MISSING = object()


def has_device_kernel(name: str, device_type: str) -> bool:
    """A schema or a Meta kernel alone does not implement runtime execution."""
    key = {"cuda": "CUDA", "cpu": "CPU", "xpu": "XPU"}.get(device_type, "PrivateUse1")
    try:
        return any(
            torch._C._dispatch_has_kernel_for_dispatch_key(name, k)
            for k in (
                key,
                "CompositeImplicitAutograd",
                "CompositeExplicitAutograd",
            )
        )
    except RuntimeError:
        return False


_FLAGGEMS_INDEXER_OPS = (
    "indexer_k_quant_and_cache",
    "cp_gather_indexer_k_quant_cache",
    "top_k_per_row_prefill",
    "top_k_per_row_decode",
    "fp8_fp4_mqa_logits",
    "fp8_fp4_paged_mqa_logits",
    "flash_mla_sparse_fwd",
)


def _load_flaggems_implementations(names: tuple[str, ...]) -> dict[str, Callable]:
    """Validate only the FlagGems implementations selected by the resolver."""
    denied = [name for name in names if not use_flaggems_op(name)]
    if denied:
        raise RuntimeError(
            "HY4 empty-build provider requires FlagGems operators disabled by "
            "the active backend/whitelist/blacklist policy: " + ", ".join(denied)
        )

    if not names:
        return {}
    import flag_gems
    import flag_gems.fused as fused

    implementations = {
        name: getattr(
            flag_gems if name == "flash_attn_varlen_func" else fused,
            name,
            None,
        )
        for name in names
    }
    missing = [
        name
        for name, implementation in implementations.items()
        if not callable(implementation)
    ]
    if missing:
        raise RuntimeError(
            "HY4 required FlagGems implementations unavailable: " + ", ".join(missing)
        )
    return implementations


@dataclass(frozen=True)
class HY4RuntimePlan:
    # Sparse indexer/MLA provider; query, cache and prefill are resolved separately.
    provider: str
    query_quantizer: Callable
    # Only selected replacements; native cache/concat implementations stay put.
    operations: Mapping[str, Callable]
    prefill_backend: type
    kv_cache_dtypes: tuple[str, ...]


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


def _flaggems_query_quantizer(implementation):
    @wraps(implementation)
    def quantize(x, group_size, *, use_ue8m0):
        return implementation(x, group_size, scale_ue8m0=use_ue8m0)

    return quantize


def _resolve_query_quantizer():
    if use_flaggems_op("per_token_group_quant_fp8"):
        import flag_gems

        implementation = getattr(flag_gems, "per_token_group_quant_fp8", None)
        if callable(implementation):
            return _flaggems_query_quantizer(implementation)
    if has_device_kernel("_C::per_token_group_fp8_quant", "cuda"):
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )

        if callable(per_token_group_quant_fp8):
            return per_token_group_quant_fp8
    raise RuntimeError(
        "HY4 per_token_group_quant_fp8 has no permitted implementation: "
        "FlagGems is disabled/unavailable and vLLM has no device/composite kernel"
    )


def prepare_hy4_runtime(vllm_config) -> HY4RuntimePlan:
    plan = validate_hy4_runtime(vllm_config)
    install_hy4_flaggems_fallback(plan)
    logger.info_once(
        "HY4 runtime: sparse_provider=%s, query_quantizer=%s, prefill=%s, KV=%s",
        plan.provider,
        f"{plan.query_quantizer.__module__}.{plan.query_quantizer.__name__}",
        plan.prefill_backend.get_name(),
        vllm_config.cache_config.cache_dtype,
    )
    return plan


def validate_hy4_runtime(vllm_config) -> HY4RuntimePlan:
    """Resolve reachable implementations before allocating model weights."""
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        raise ValueError("HY4 currently requires an NVIDIA CUDA runtime")
    cache_dtype = vllm_config.cache_config.cache_dtype
    if cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
        raise ValueError(f"Unsupported HY4 KV cache dtype: {cache_dtype}")
    import vllm.model_executor.layers.attention.mla_attention as mla_attention

    config = vllm_config.model_config.hf_text_config
    native_indexer = native_hy4_available(config.index_topk)
    native_cache = has_device_kernel("_C_cache_ops::concat_and_cache_mla", "cuda")
    native_concat = has_device_kernel("_C_cache_ops::concat_mla_q", "cuda")
    prefill_backend = select_hy4_prefill_backend(
        vllm_config, mla_attention.get_mla_prefill_backend
    )
    portable = not (
        native_indexer and native_cache and native_concat and prefill_backend
    )
    if portable:
        if str(cache_dtype).startswith("fp8"):
            raise ValueError(
                "HY4 FP8 KV requires a complete native path including MLA prefill"
            )
        if cache_dtype == "auto" and vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("HY4 FlagGems auto KV requires bfloat16 model dtype")

    required = () if native_indexer else _FLAGGEMS_INDEXER_OPS
    if not native_cache:
        required += ("concat_and_cache_mla",)
    if prefill_backend is None:
        qk_width = config.qk_nope_head_dim + config.qk_rope_head_dim
        if not (0 < config.v_head_dim <= qk_width <= 256):
            raise ValueError(
                "HY4 FlagGems MLA prefill requires 0 < v_head_dim <= "
                f"qk_head_dim <= 256; got v={config.v_head_dim}, qk={qk_width}"
            )
        required += ("flash_attn_varlen_func",)

    implementations = _load_flaggems_implementations(required)
    query_quantizer = _resolve_query_quantizer()
    if prefill_backend is None:
        prefill_backend = _make_hy4_flaggems_mla_prefill_backend(
            implementations["flash_attn_varlen_func"]
        )
    if not native_concat:
        implementations["concat_mla_q"] = _concat_hy4_mla_q
    return HY4RuntimePlan(
        "native" if native_indexer else "flaggems",
        query_quantizer,
        MappingProxyType(implementations),
        prefill_backend,
        ("auto", "bfloat16") if portable else ("auto", "bfloat16", "fp8", "fp8_e4m3"),
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


def _concat_hy4_mla_q(ql_nope, q_pe, q_out) -> None:
    nope_width, rope_width = ql_nope.shape[-1], q_pe.shape[-1]
    if q_out.shape[-1] != nope_width + rope_width:
        raise ValueError(
            "HY4 concat_mla_q output width mismatch: "
            f"expected {nope_width + rope_width}, got {q_out.shape[-1]}"
        )
    q_out[..., :nope_width].copy_(ql_nope)
    q_out[..., nope_width:].copy_(q_pe)


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


def _make_hy4_flaggems_mla_prefill_backend(flash_attn_varlen_func) -> type:
    """Build the MLA prefill adapter without importing FA extensions."""
    from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

    class FlagGemsMLAPrefillBackend(MLAPrefillBackend):
        @staticmethod
        def get_name() -> str:
            return "HYV4_FLAGGEMS_MLA_PREFILL"

        @classmethod
        def is_available(cls) -> bool:
            return True

        def _flash_attn_varlen(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            *,
            cu_seqlens_q: torch.Tensor,
            cu_seqlens_k: torch.Tensor,
            max_seqlen_q: int,
            max_seqlen_k: int,
            causal: bool,
            return_softmax_lse: bool,
            out: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            # Dense prefill uses the expanded attention heads (HY4: q/k=256,
            # v=256), not the 576/512 compressed sparse-MLA representation.
            # Pad a smaller value head to match q/k for the FlagGems API.
            maybe_padded_v = v
            if v.shape[-1] != q.shape[-1]:
                maybe_padded_v = torch.nn.functional.pad(
                    v, [0, q.shape[-1] - v.shape[-1]], value=0
                )

            result = flash_attn_varlen_func(
                q=q,
                k=k,
                v=maybe_padded_v,
                max_seqlen_q=max_seqlen_q,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=max_seqlen_k,
                cu_seqlens_k=cu_seqlens_k,
                softmax_scale=self.scale,
                causal=causal,
                return_softmax_lse=return_softmax_lse,
                # ``out`` has the unpadded value width.  Let the Triton
                # wrapper allocate when padding is needed, then copy below.
                out=out if maybe_padded_v is v else None,
            )
            lse = None
            if isinstance(result, tuple):
                result, lse = result[0], result[1]
            if maybe_padded_v is not v:
                result = result[..., : v.shape[-1]]
                if out is not None:
                    out.copy_(result)
                    result = out
            if return_softmax_lse:
                assert lse is not None
                return result, lse
            return result

        def run_prefill_new_tokens(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            return_softmax_lse: bool,
            out: torch.Tensor | None = None,
            output_scale: torch.Tensor | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            if output_scale is not None:
                raise NotImplementedError(
                    "HY4 FlagGems MLA prefill does not support quantized output"
                )
            metadata = self._prefill_metadata
            return self._flash_attn_varlen(
                q,
                k,
                v,
                cu_seqlens_q=metadata.query_start_loc,
                cu_seqlens_k=metadata.query_start_loc,
                max_seqlen_q=metadata.max_query_len,
                max_seqlen_k=metadata.max_query_len,
                causal=True,
                return_softmax_lse=return_softmax_lse,
                out=out,
            )

        def run_prefill_context_chunk(
            self,
            chunk_idx: int,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            metadata = self._prefill_metadata
            assert metadata.chunked_context is not None
            chunked = metadata.chunked_context
            result = self._flash_attn_varlen(
                q,
                k,
                v,
                cu_seqlens_q=metadata.query_start_loc,
                cu_seqlens_k=chunked.cu_seq_lens[chunk_idx],
                max_seqlen_q=metadata.max_query_len,
                max_seqlen_k=chunked.max_seq_lens[chunk_idx],
                causal=False,
                return_softmax_lse=True,
            )
            assert isinstance(result, tuple)
            return result

    return FlagGemsMLAPrefillBackend


def _install_fallback(tx, plan: HY4RuntimePlan) -> bool:
    """Install the resolved plan once in a dedicated HY4 worker."""
    import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer

    if getattr(sparse_indexer, "_hy4_runtime_installed", False):
        return bool(plan.operations)
    implementations = plan.operations
    if plan.provider == "flaggems":
        # FlagGems 5.3.3 advertises Triton TLE support for this runtime, but
        # the bundled Triton TLE language module is missing ``cumsum``.  Its
        # TLE top-k kernels therefore fail while Triton is hashing/compiling
        # the kernel, before any device code can run.  Select the portable
        # non-TLE kernels (the module keeps both implementations) for HY4's
        # local fallback.  This changes the process-global FlagGems module; rollback covers it
        # if installation fails. Use a dedicated HY4 worker.
        import importlib

        top_k_prefill_module = importlib.import_module(
            "flag_gems.fused.top_k_per_row_prefill"
        )
        top_k_decode_module = importlib.import_module(
            "flag_gems.fused.top_k_per_row_decode"
        )
        tx.set(top_k_prefill_module, "HAS_TLE", False)
        tx.set(top_k_decode_module, "HAS_TLE", False)
        # Triton's dependency walker still visits the constexpr-disabled TLE
        # branch while compiling the non-TLE kernels.  Give that walker a
        # cache-key-compatible cumsum symbol; the branch is never emitted
        # because HAS_TLE is false, while the non-TLE path uses tl.cumsum.
        import triton.language as tl

        for top_k_module in (top_k_prefill_module, top_k_decode_module):
            tle = getattr(top_k_module, "tle", None)
            if tle is not None and not hasattr(tle, "cumsum"):
                tx.set(tle, "cumsum", tl.cumsum)

        # ``flash_mla_sparse_fwd`` has a separate TLE gate from the top-k
        # helpers above.  FlagGems selects its TLE implementation on this
        # Triton build, but the bundled ``triton.experimental.tle.language``
        # module is missing ``pipe``; the first HY4 request then fails while
        # Triton hashes the kernel.  Force the portable non-TLE implementation
        # for this worker-wide fallback as well.
        flashmla_sparse_module = importlib.import_module(
            "flag_gems.fused.flashmla_sparse"
        )
        tx.set(flashmla_sparse_module, "HAS_TLE_FLASHMLA_SPARSE", False)

        cp_gather_indexer_k_quant_cache = implementations[
            "cp_gather_indexer_k_quant_cache"
        ]
        flash_mla_sparse_fwd = implementations["flash_mla_sparse_fwd"]
        fp8_fp4_mqa_logits = implementations["fp8_fp4_mqa_logits"]
        fp8_fp4_paged_mqa_logits = implementations["fp8_fp4_paged_mqa_logits"]
        indexer_k_quant_and_cache = implementations["indexer_k_quant_and_cache"]
        top_k_per_row_decode = implementations["top_k_per_row_decode"]
        top_k_per_row_prefill = implementations["top_k_per_row_prefill"]
        # ``sparse_attn_indexer`` captured these names when its custom op was
        # registered.  Rebind the module globals so the already-registered op
        # invokes FlagGems' Triton implementation at runtime.
        tx.set(sparse_indexer, "fp8_fp4_mqa_logits", fp8_fp4_mqa_logits)
        tx.set(sparse_indexer, "fp8_fp4_paged_mqa_logits", fp8_fp4_paged_mqa_logits)
        tx.set(
            sparse_indexer,
            "ops",
            SimpleNamespace(
                indexer_k_quant_and_cache=indexer_k_quant_and_cache,
                cp_gather_indexer_k_quant_cache=cp_gather_indexer_k_quant_cache,
                top_k_per_row_prefill=top_k_per_row_prefill,
                top_k_per_row_decode=top_k_per_row_decode,
            ),
        )

        # vLLM's CUDA indexer selects persistent/cooperative top-k for the
        # supported 512/1024/2048 sizes; the portable path uses generic top-k.
        # Those are part of the omitted vLLM extension, so make only this module's
        # platform view report a non-CUDA execution path for that branch.  The
        # CustomOp dispatcher has already bound ``SparseAttnIndexer.forward_cuda``
        # using its own platform object; this proxy therefore affects only the
        # registered Python implementation and leaves the rest of vLLM unchanged.
        native_sparse_indexer_platform = sparse_indexer.current_platform

        class _HY4FlagGemsIndexerPlatform:
            def is_cuda(self) -> bool:
                return False

            def __getattr__(self, name):
                return getattr(native_sparse_indexer_platform, name)

        tx.set(sparse_indexer, "current_platform", _HY4FlagGemsIndexerPlatform())

        # HY4's sink-capable backend imports these functions by value.  Rebind its
        # globals (and the base module for inherited helper paths) to avoid the
        # native FlashMLA/NV kernel when the wheel is empty.
        import vllm.v1.attention.backends.mla.flashmla_sparse as native_sparse

        from vllm_fl.models import hy_v4_flashmla_sparse as hy4_sparse

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
