# SPDX-License-Identifier: Apache-2.0
"""Asymmetric `DG-R2-K8V4` (`6,272 B/page`) KV-cache sizing, 3-plane view carving,
and 4-in-1 `fused_rope_and_cache_k8v4` registration for Iluvatar BI-V150 (`ivcore11`).

Physical & logical KV-cache layout (`block_size = 16`, `head_size = D in {64, 128, 256}`):
- Logical 5D tensor exposed to vLLM: `(num_blocks, 3, 16, num_kv_heads, D // 2)` (`int8`):
  * Planes `0` & `1`: full `INT8` Key `[16, H_kv, D]` split into two `D // 2` half-planes
    (preserving RoPE phase precision on sensitive Key dot products).
  * Plane `2`: `QuaRot-R2`-smoothed `INT4` Value `[16, H_kv, D]` packed 2 nibbles per `int8` byte (`D // 2` bytes).
- Page-tail scale storage: `compute_k8v4_page_bytes` reserves SME 128B-aligned extra bytes
  at the tail of every physical page (`6,272 B/page = 49 x 128 B` for `H_kv=2, D=128`)
  storing per-(token, head) `bf16x2` packed `(k_scale, v_scale)` factors so Stage-1
  attention loads both scales in a single 32-bit transaction.
- `vllm::fused_rope_and_cache_k8v4` bridges `RopeKvFusionPass` to FlagGems' 4-in-1
  kernel performing in-wave NeoX Q/K RoPE, DB-LSQ clipping search, INT8-K / packed
  INT4-V quantization, and `bf16x2` scale-tail writes in one launch.
"""

from __future__ import annotations

from functools import wraps
import logging
import math
import sys
from typing import Any

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from flag_gems.runtime.backend._iluvatar.fused.paged_gqa_attention import (
    build_k8v8_tail_scale_views,
    carve_k8v4_kv_and_scale_views,
    compute_bf16x2_page_tail_bytes,
    compute_k8v4_page_bytes,
)
from vllm_fl.compilation.rope_kv_fusion_pass import (
    dispatch_layer_qk_norm_rope_and_kv_cache_update,
    dispatch_layer_rope_and_kv_cache_update,
    fake_layer_qk_norm_rope_and_kv_cache_update,
    fake_layer_rope_and_kv_cache_update,
)

# Register 4-in-1 (RoPE+K8V4) and 5-in-1 (QK-Norm+RoPE+K8V4) custom ops emitted by `RopeKvFusionPass`.
if not hasattr(torch.ops.vllm, "fused_rope_and_cache_k8v4"):
    direct_register_custom_op(
        op_name="fused_rope_and_cache_k8v4",
        op_func=dispatch_layer_rope_and_kv_cache_update,
        mutates_args=[],
        fake_impl=fake_layer_rope_and_kv_cache_update,
    )

if not hasattr(torch.ops.vllm, "fused_qk_norm_rope_and_cache_k8v4"):
    direct_register_custom_op(
        op_name="fused_qk_norm_rope_and_cache_k8v4",
        op_func=dispatch_layer_qk_norm_rope_and_kv_cache_update,
        mutates_args=[],
        fake_impl=fake_layer_qk_norm_rope_and_kv_cache_update,
    )

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)

SUPPORTED_K8V4_HEAD_DIMS: tuple[int, ...] = (64, 128, 256)
SUPPORTED_K8V4_HALF_HEAD_DIMS: tuple[int, ...] = (32, 64, 128)
_LOGGED_PAGE_SPEC: bool = False
_ORIG_RESHAPE_KV_CACHE: Any = None


def enable_auto_iluvatar_kv_cache_config(vllm_config: Any) -> bool:
    """Upgrade `cache_config.cache_dtype='auto'` to `'int8_per_token_head'` (`DG-R2-K8V4`)."""
    if vllm_config is None:
        return False
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None or getattr(cache_config, "cache_dtype", None) != "auto":
        return False
    if getattr(cache_config, "block_size", None) not in (None, 16):
        return False

    model_config = getattr(vllm_config, "model_config", None)
    if model_config is not None:
        if any(bool(getattr(model_config, k, False)) for k in ("is_attention_free", "use_mla", "is_hybrid", "is_encoder_decoder")):
            return False
        mm_attr = getattr(model_config, "is_multimodal_model", False)
        if mm_attr() if callable(mm_attr) else bool(mm_attr):
            return False
        dtype = getattr(model_config, "dtype", None)
        if dtype is not None and dtype != torch.bfloat16:
            return False
        if getattr(cache_config, "sliding_window", None) not in (None, 0, -1, (-1, -1)):
            return False
        hf_cfg = getattr(model_config, "hf_text_config", None) or getattr(model_config, "hf_config", None)
        if hf_cfg is not None:
            if bool(getattr(hf_cfg, "attn_logit_softcapping", None)):
                return False
            if not getattr(model_config, "disable_sliding_window", False) and getattr(hf_cfg, "use_sliding_window", True) is not False:
                sw_mc = model_config.get_sliding_window() if hasattr(model_config, "get_sliding_window") else getattr(hf_cfg, "sliding_window", None)
                if sw_mc not in (None, 0, -1, (-1, -1)):
                    return False
            layer_types = getattr(hf_cfg, "layer_types", None)
            if isinstance(layer_types, (list, tuple)) and any(lt != "full_attention" for lt in layer_types):
                return False
        try:
            if model_config.get_head_size() not in SUPPORTED_K8V4_HEAD_DIMS:
                return False
        except Exception:
            return False
        parallel_config = getattr(vllm_config, "parallel_config", None)
        nh = nkv = None
        if parallel_config is not None and hasattr(model_config, "get_num_attention_heads") and hasattr(model_config, "get_num_kv_heads"):
            try:
                nh = int(model_config.get_num_attention_heads(parallel_config))
                nkv = int(model_config.get_num_kv_heads(parallel_config))
            except Exception:
                nh = nkv = None
        if (nh is None or nkv is None) and hf_cfg is not None:
            tp_size = max(1, int(getattr(parallel_config, "tensor_parallel_size", 1) or 1))
            raw_nh = getattr(hf_cfg, "num_attention_heads", None)
            raw_nkv = getattr(hf_cfg, "num_key_value_heads", raw_nh)
            if isinstance(raw_nh, int) and isinstance(raw_nkv, int):
                nh, nkv = raw_nh // tp_size, max(1, raw_nkv // tp_size)
        if nh is not None and nkv is not None:
            if nh <= 0 or nkv <= 0 or (nh % nkv != 0) or not (1 <= (nh // nkv) <= 32):
                return False

    cache_config.cache_dtype = "int8_per_token_head"
    logger.info(
        "[FL-Iluvatar] Auto-upgraded KV cache dtype from 'auto' to "
        "'int8_per_token_head' (DG-R2-K8V4 6272 B/page + BF16x2 Page-Tail)."
    )
    return True


def get_k8v4_kv_cache_shape(
    num_blocks: int, block_size: int, num_kv_heads: int, head_size: int, cache_dtype_str: str = "auto",
) -> tuple[int, ...]:
    """Return 3-plane `(num_blocks, 3, 16, num_kv_heads, head_size // 2)` shape for `DG-R2-K8V4`."""
    if block_size % 16 != 0:
        raise ValueError("Block size must be a multiple of 16.")
    if cache_dtype_str == "int8_per_token_head" and block_size == 16 and head_size in SUPPORTED_K8V4_HEAD_DIMS:
        return (num_blocks, 3, block_size, num_kv_heads, head_size // 2)
    return (num_blocks, 2, block_size, num_kv_heads, head_size)


def is_k8v4_cache_tensor(kv_cache: torch.Tensor) -> bool:
    """Return True if `kv_cache` uses the 3-plane `int8` `DG-R2-K8V4` tensor layout."""
    return (
        kv_cache.dtype == torch.int8
        and kv_cache.ndim == 5
        and kv_cache.shape[1] == 3
        and kv_cache.shape[4] in SUPPORTED_K8V4_HALF_HEAD_DIMS
    )


def _is_bf16x2_spec_eligible(spec: Any) -> bool:
    """Return True if `AttentionSpec` uses `int8_per_token_head` with 16-aligned geometry."""
    qmode = getattr(spec, "kv_quant_mode", None)
    if qmode is None or not getattr(qmode, "is_per_token_head", False) or getattr(spec, "dtype", None) != torch.int8:
        return False
    hs, bs, nkv = getattr(spec, "head_size", 0), getattr(spec, "block_size", 0), getattr(spec, "num_kv_heads", 0)
    return isinstance(hs, int) and isinstance(bs, int) and isinstance(nkv, int) and hs > 0 and hs % 16 == 0 and bs > 0 and bs % 16 == 0 and nkv > 0


def _is_k8v4_spec_eligible(spec: Any) -> bool:
    """Return True if `AttentionSpec` qualifies for `DG-R2-K8V4` (`block_size=16`, equal K/V head dim)."""
    if not _is_bf16x2_spec_eligible(spec) or getattr(spec, "block_size", 0) != 16:
        return False
    hs = getattr(spec, "head_size", 0)
    hsv = getattr(spec, "head_size_v", hs)
    return hs in SUPPORTED_K8V4_HEAD_DIMS and (hs if hsv is None else hsv) == hs


def bind_kv_cache_bundle(impl: Any, kv_cache: torch.Tensor) -> tuple[Any, ...]:
    """Carve and memoize K/V payload views, `bf16x2` scale views, and stride metadata on `impl`.

    Computes 64-bit indexing requirements (`wide_kv` when max byte/element offset
    exceeds `INT32_MAX`) once at KV-cache bind time so steady-state decode steps
    perform an O(1) signature check (`_fl_bound_kv_sig`) with zero view allocations.
    """
    if getattr(impl, "_is_per_token_head_quant", False) and kv_cache.dtype == torch.int8:
        if is_k8v4_cache_tensor(kv_cache):
            kc, vc, ks, vs, ps = carve_k8v4_kv_and_scale_views(kv_cache)
        else:
            kc, vc = kv_cache.unbind(1)
            ks, vs, ps = build_k8v8_tail_scale_views(kv_cache)
        impl._k_scale_cache, impl._v_scale_cache, impl._fl_packed_scale_cache = ks, vs, ps
        ks_s = tuple(int(x) for x in ks.stride())
        vs_s = tuple(int(x) for x in vs.stride())
        ps_s = tuple(int(x) for x in ps.stride()) if ps is not None else (0, 0, 0)
        int8_kv = True
    else:
        kc, vc = kv_cache.unbind(1)
        ks, vs, ps = kc, vc, None
        ks_s = vs_s = ps_s = (0, 0, 0)
        int8_kv = False

    kc_s, vc_s = tuple(int(x) for x in kc.stride()), tuple(int(x) for x in vc.stride())
    # Enable 64-bit KV pointer arithmetic whenever max tensor offset could overflow int32.
    wide_kv = any(sum((sz - 1) * st for sz, st in zip(t.shape, t.stride())) > 2**31 - 1 for t in (kc, vc))
    v_int4 = int8_kv and (vc.shape[-1] * 2 == impl.head_size)
    # When packed `bf16x2` page-tail scales are present, pass `ps` for both K/V scale args
    # and signal packed mode via `attn_vs_s = (0, 0, 0)`.
    use_packed = int8_kv and (ps is not None)
    attn_ks, attn_vs = (ps, ps) if use_packed else (ks, vs)
    attn_ks_s, attn_vs_s = (ps_s, (0, 0, 0)) if use_packed else (ks_s, vs_s)

    bundle = (kc, vc, ks, vs, ps, attn_ks, attn_vs, kc_s, vc_s, ks_s, vs_s, ps_s, attn_ks_s, attn_vs_s, wide_kv, int8_kv, v_int4, use_packed)
    impl._fl_kv_ptr, impl._fl_key_cache, impl._fl_value_cache = kv_cache.data_ptr(), kc, vc
    impl._fl_bound_kv_sig = (kv_cache.data_ptr(), kv_cache.shape, kv_cache.dtype, kv_cache.storage_offset())
    impl._fl_bound_kv_bundle = bundle
    return bundle


def _k8v4_reshape_attention_kv_cache(
    kv_raw_tensor: torch.Tensor, kv_cache_spec: Any, kv_cache_shape: tuple[int, ...],
    kv_cache_stride_order: tuple[int, ...], num_blocks: int, packing: tuple[int, int] | None,
) -> torch.Tensor:
    """Reshape vLLM's raw byte buffer (`num_blocks * page_size_bytes`) into the 5D logical KV view.

    Because `AttentionSpec.page_size_bytes` includes extra bytes for the `bf16x2`
    per-token-head scale tail (`flat.numel() > needed`), slices `flat[:needed]`
    for the logical 5D tensor while keeping the full raw buffer backing storage
    intact so `carve_k8v4_kv_and_scale_views` can map the trailing scale region.
    """
    if (
        getattr(getattr(kv_cache_spec, "kv_quant_mode", None), "name", None) == "INT8_PER_TOKEN_HEAD"
        and kv_cache_spec.dtype == torch.int8
        and len(kv_cache_shape) == 5
        and packing is None
        and getattr(kv_cache_spec, "page_size_padded", None) is None
    ):
        permuted_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
        inv_order = [kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))]
        flat = kv_raw_tensor.view(kv_cache_spec.dtype)
        needed = math.prod(permuted_shape)
        return (flat[:needed] if flat.numel() > needed else flat).view(permuted_shape).permute(*inv_order)
    return _ORIG_RESHAPE_KV_CACHE(kv_raw_tensor, kv_cache_spec, kv_cache_shape, kv_cache_stride_order, num_blocks, packing)


def install_reshape_kv_hook_on_module(mod: Any) -> None:
    """Idempotently replace `_reshape_attention_kv_cache` on `mod` with `_k8v4_reshape_attention_kv_cache`."""
    global _ORIG_RESHAPE_KV_CACHE
    if mod is not None and hasattr(mod, "_reshape_attention_kv_cache") and mod._reshape_attention_kv_cache is not _k8v4_reshape_attention_kv_cache:
        if _ORIG_RESHAPE_KV_CACHE is None:
            _ORIG_RESHAPE_KV_CACHE = mod._reshape_attention_kv_cache
        mod._reshape_attention_kv_cache = _k8v4_reshape_attention_kv_cache


def install_kv_cache_k8v4_hooks() -> None:
    """Install `AttentionSpec.page_size_bytes` and `_reshape_attention_kv_cache` hooks."""
    import vllm.v1.kv_cache_interface as kv_iface

    cls = getattr(kv_iface, "AttentionSpec", None)
    prop = cls.__dict__.get("page_size_bytes") if cls is not None else None
    if isinstance(prop, property) and prop.fget is not None and not getattr(prop.fget, "_fl_k8v4_patched", False):
        orig_fget = prop.fget

        @wraps(orig_fget)
        def _fl_page_size_bytes(self: Any) -> int:
            global _LOGGED_PAGE_SPEC
            # Asymmetric 3-plane K8V4 (`block_size=16`): 128B SME-aligned payload + bf16x2 scale tail.
            if _is_k8v4_spec_eligible(self):
                total_bytes = compute_k8v4_page_bytes(self.block_size, self.num_kv_heads, self.head_size)
                if (padded := getattr(self, "page_size_padded", None)) is not None:
                    return int(padded)
                if not _LOGGED_PAGE_SPEC:
                    _LOGGED_PAGE_SPEC = True
                    logger.info(
                        "[FL-DG-R2-K8V4] Asymmetric K8V4 KV page_size_bytes=%d B (head_dim=%d, %d x 128B SME-aligned).",
                        total_bytes, int(self.head_size), total_bytes // 128,
                    )
                return total_bytes
            # Symmetric K8V8 fallback: append bf16x2 page-tail bytes after standard payload.
            if _is_bf16x2_spec_eligible(self):
                total_bytes = int(self.real_page_size_bytes) + compute_bf16x2_page_tail_bytes(self.block_size, self.num_kv_heads)
                padded = getattr(self, "page_size_padded", None)
                return int(padded) if padded is not None else total_bytes
            return orig_fget(self)

        setattr(_fl_page_size_bytes, "_fl_k8v4_patched", True)
        cls.page_size_bytes = property(_fl_page_size_bytes)

    try:
        import vllm.v1.worker.gpu.attn_utils as attn_utils

        install_reshape_kv_hook_on_module(attn_utils)
    except Exception:
        logger.debug("Deferred attn_utils KV-reshape hook until worker init.")
    for mod_name in ("vllm.v1.worker.gpu_model_runner", "vllm_fl.worker.model_runner"):
        install_reshape_kv_hook_on_module(sys.modules.get(mod_name))
