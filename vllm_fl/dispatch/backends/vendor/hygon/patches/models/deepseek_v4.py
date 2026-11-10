# Copyright (c) 2026 BAAI. All rights reserved.

"""DeepSeek-V4-specific Hygon model and attention compatibility patches."""

import functools
import importlib
import logging
import os
from types import MethodType
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _patch_mhc_ops() -> None:
    """Use the Hygon vLLM ROCm MHC correctness path.

    vLLM 0.24's generic TileLang prenorm and fused-post-pre reductions assume
    a 32-lane warp. Hygon vLLM either uses wave64 AITER operators or the
    PyTorch ROCm definition. FL does not depend on the separate HCU plugin;
    the implementation module discovers AITER directly and retains the same
    PyTorch fallback. Fused post+pre is composed from those validated leaves.
    """
    mhc_layers = importlib.import_module("vllm.model_executor.layers.mhc")
    mhc_impl = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.hygon.impl.other.mhc"
    )

    pre_cls = mhc_layers.MHCPreOp
    original_pre = pre_cls.forward_hip
    if not getattr(original_pre, "_vllm_fl_hygon", False):

        @functools.wraps(original_pre)
        def _mhc_pre_hygon(
            self,
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits=1,
            norm_weight=None,
            norm_eps=0.0,
        ):
            return mhc_impl.mhc_pre_hygon(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                norm_weight,
                norm_eps,
            )

        _mhc_pre_hygon._vllm_fl_hygon = True
        pre_cls.forward_hip = _mhc_pre_hygon

    post_cls = mhc_layers.MHCPostOp
    original_post = post_cls.forward_hip
    if not getattr(original_post, "_vllm_fl_hygon", False):

        @functools.wraps(original_post)
        def _mhc_post_hygon(
            self,
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        ):
            return mhc_impl.mhc_post_hygon(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
            )

        _mhc_post_hygon._vllm_fl_hygon = True
        post_cls.forward_hip = _mhc_post_hygon

    fused_cls = mhc_layers.MHCFusedPostPreOp
    original_fused = fused_cls.forward_hip
    if not getattr(original_fused, "_vllm_fl_hygon", False):

        @functools.wraps(original_fused)
        def _mhc_fused_post_pre_hygon(
            self,
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits=1,
            tile_n=1,
            norm_weight=None,
            norm_eps=0.0,
        ):
            return mhc_impl.mhc_fused_post_pre_hygon(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                tile_n,
                norm_weight,
                norm_eps,
            )

        _mhc_fused_post_pre_hygon._vllm_fl_hygon = True
        fused_cls.forward_hip = _mhc_fused_post_pre_hygon

    logger.info("Patched DeepSeek-V4 MHC with Hygon AITER/reference ops")


def _hygon_has_cutedsl() -> bool:
    """CuTeDSL kernels in DeepSeek-V4 are NVIDIA-specific."""
    return False


def _patch_deepseek_v4_cutedsl_selection() -> None:
    """Keep Hygon DeepSeek-V4 indexer/cache operations on Triton kernels.

    Upstream imports ``has_cutedsl`` into these modules and uses package
    presence alone to select NVIDIA CuTeDSL kernels.  A Hygon environment may
    contain the ``cutlass`` Python package while lacking
    ``vllm.vllm_flash_attn.cute``.  vllm_hcu uses the Triton FP8 indexer and
    cache kernels instead, so override only the two module-local probes.
    """
    module_names = (
        "vllm.models.deepseek_v4.common.ops.fused_indexer_q",
        "vllm.models.deepseek_v4.common.ops.cache_utils",
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        module.has_cutedsl = _hygon_has_cutedsl

    logger.info("Disabled NVIDIA CuTeDSL DeepSeek-V4 ops on Hygon")


def _patch_deepseek_v4_bf16_indexer_cache() -> None:
    """Add the validated Hygon BF16 Lightning Indexer branch.

    The main MLA/SWA cache remains controlled by ``--kv-cache-dtype``.  This
    patch changes only the separate Lightning Indexer Q/K cache from
    FP8+scale to BF16, then switches the compressor writer and Q RoPE output
    to their BF16 ABIs.  The corresponding prefill/decode MQA flow is selected
    later in :func:`_patch_sparse_attn_indexer` by inspecting the cache dtype.
    """
    import torch

    bf16_impl = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.hygon.impl.attention.bf16_indexer"
    )
    if not bf16_impl.use_bf16_indexer_cache():
        logger.info("Hygon BF16 Lightning Indexer cache is disabled by environment")
        return

    attention_module = importlib.import_module("vllm.models.deepseek_v4.attention")
    compressor_module = importlib.import_module("vllm.models.deepseek_v4.compressor")
    common_q_module = importlib.import_module(
        "vllm.models.deepseek_v4.common.ops.fused_indexer_q"
    )

    indexer_cls = attention_module.DeepseekV4Indexer
    original_init = indexer_cls.__init__
    if not getattr(original_init, "_vllm_fl_hygon_bf16_indexer", False):

        @functools.wraps(original_init)
        def _init_bf16_indexer(self, *args, **kwargs) -> None:
            original_init(self, *args, **kwargs)
            if self.use_fp4_kv:
                raise AssertionError(
                    "Hygon BF16 and MXFP4 Indexer caches are mutually exclusive"
                )

            # Cache allocation happens after model construction from this
            # layer's KVCacheSpec, so changing the spec fields here is early
            # enough and preserves the object references held by compressor
            # and SparseAttnIndexer.
            self.use_bf16_cache = True
            self.k_cache.head_dim = self.head_dim
            self.k_cache.dtype = torch.bfloat16
            self.compressor.use_bf16_cache = True
            self.indexer_op.use_bf16_cache = True

        _init_bf16_indexer._vllm_fl_hygon_bf16_indexer = True
        indexer_cls.__init__ = _init_bf16_indexer

    original_q = attention_module.fused_indexer_q_rope_quant
    if not getattr(original_q, "_vllm_fl_hygon_bf16_indexer", False):

        @functools.wraps(original_q)
        def _fused_indexer_q_rope_bf16(
            positions,
            index_q,
            cos_sin_cache,
            index_weights,
            weights_softmax_scale,
            weights_head_scale,
            use_fp4=False,
        ):
            if use_fp4:
                return original_q(
                    positions,
                    index_q,
                    cos_sin_cache,
                    index_weights,
                    weights_softmax_scale,
                    weights_head_scale,
                    use_fp4=True,
                )
            return bf16_impl.fused_indexer_q_rope_bf16(
                positions,
                index_q,
                cos_sin_cache,
                index_weights,
                weights_softmax_scale,
                weights_head_scale,
            )

        _fused_indexer_q_rope_bf16._vllm_fl_hygon_bf16_indexer = True
        attention_module.fused_indexer_q_rope_quant = _fused_indexer_q_rope_bf16
        common_q_module.fused_indexer_q_rope_quant = _fused_indexer_q_rope_bf16

    original_store = compressor_module.compress_norm_rope_store_triton
    if not getattr(original_store, "_vllm_fl_hygon_bf16_indexer", False):

        @functools.wraps(original_store)
        def _compress_norm_rope_store_hygon(**kwargs) -> None:
            kv_cache = kwargs["kv_cache"]
            if kv_cache.dtype == torch.bfloat16:
                bf16_impl.compress_norm_rope_store_bf16(**kwargs)
                return
            original_store(**kwargs)

        _compress_norm_rope_store_hygon._vllm_fl_hygon_bf16_indexer = True
        compressor_module.compress_norm_rope_store_triton = (
            _compress_norm_rope_store_hygon
        )

    logger.info("Patched DeepSeek-V4 Lightning Indexer to use Hygon BF16 cache")


def _patch_deepseek_v4_qnorm_rope_kv_insert() -> None:
    """Use Hygon LightOp's 8-argument fused Q/KV insert ABI.

    vLLM 0.24.0 added ``q_head_padded`` to its CUDA op and changed the result
    from an in-place mutation to a returned padded Q tensor.  Hygon's ROCm
    backend does not pad Q heads, while the validated LightOp kernel retains
    the older 8-argument in-place ABI.  Call that kernel directly and return Q.
    """
    import torch

    rocm_module = importlib.import_module("vllm.models.deepseek_v4.amd.rocm")
    attention_cls = rocm_module.DeepseekV4ROCMAiterMLAAttention
    original = attention_cls._fused_qnorm_rope_kv_insert
    if getattr(original, "_vllm_fl_hygon", False):
        return

    @functools.wraps(original)
    def _fused_qnorm_rope_kv_insert_hygon(
        self,
        q,
        kv,
        positions,
        attn_metadata,
    ):
        if not isinstance(attn_metadata, dict):
            return original(self, q, kv, positions, attn_metadata)

        swa_kv_cache = self.swa_cache_layer.kv_cache
        if swa_kv_cache.dtype != torch.uint8:
            return original(self, q, kv, positions, attn_metadata)

        swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
        if swa_metadata is None:
            raise AssertionError("DeepSeek-V4 SWA metadata is missing")
        if self.padded_heads != self.n_local_heads:
            raise AssertionError(
                "Hygon LightOp Q/KV insert requires unpadded ROCm Q heads"
            )

        try:
            from lightop import op as lightop_op
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Hygon DeepSeek-V4 requires LightOp's "
                "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
            ) from exc

        kernel = getattr(
            lightop_op,
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
            None,
        )
        if kernel is None:
            raise RuntimeError(
                "The installed LightOp lacks "
                "lightop.op.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
            )

        swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
        kernel(
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            positions.to(torch.int64),
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )
        return q

    _fused_qnorm_rope_kv_insert_hygon._vllm_fl_hygon = True
    attention_cls._fused_qnorm_rope_kv_insert = _fused_qnorm_rope_kv_insert_hygon
    logger.info("Patched DeepSeek-V4 fused Q/KV insert with Hygon LightOp")


def _uses_bf16_indexer_cache(indexer) -> bool:
    import torch

    if getattr(indexer, "use_bf16_cache", False):
        return True
    configured_dtype = getattr(indexer.k_cache, "dtype", None)
    if configured_dtype is not None:
        return configured_dtype == torch.bfloat16
    cache = indexer.k_cache.kv_cache
    return cache.numel() > 0 and cache.dtype == torch.bfloat16


def _patch_sparse_attn_indexer() -> None:
    """Route Hygon to vLLM's native ROCm indexer with FL operator leaves.

    Keep the current metadata/workspace flow in this module.  Reuse the cache
    quantization/gather Triton kernels already shipped in
    ``vllm.v1.attention.ops.rocm_aiter_mla_sparse``, and replace only the
    unavailable DeepGEMM/CUDA-extension leaves with FL's LightOp adapters.
    This mirrors the Hygon selection rule without taking a runtime dependency
    on the separate vllm_hcu plugin.
    """
    import torch

    sparse_indexer = importlib.import_module(
        "vllm.model_executor.layers.sparse_attn_indexer"
    )
    indexer_cls = sparse_indexer.SparseAttnIndexer
    original = indexer_cls.forward_hip
    if getattr(original, "_vllm_fl_hygon", False):
        return

    sparse_impl = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.hygon.impl.attention.sparse_attn_indexer"
    )
    bf16_impl = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.hygon.impl.attention.bf16_indexer"
    )
    sparse_impl.install_sparse_indexer_hygon_ops(sparse_indexer)

    from vllm.utils.torch_utils import _encode_layer_name

    @functools.wraps(original)
    def _forward_hip_hygon(self, hidden_states, q_quant, k, weights):
        if _uses_bf16_indexer_cache(self):
            if isinstance(q_quant, tuple) or q_quant.dtype != torch.bfloat16:
                raise AssertionError(
                    "Hygon BF16 Indexer cache requires a BF16 query tensor"
                )
            return bf16_impl.sparse_attn_indexer_bf16_hygon(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                weights,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )

        use_native_rocm = (
            self.skip_k_cache_insert or not sparse_indexer.rocm_aiter_ops.is_enabled()
        )
        if not use_native_rocm:
            return original(self, hidden_states, q_quant, k, weights)

        if self.use_fp4_cache:
            raise AssertionError("Hygon sparse indexer does not support FP4 cache")
        if isinstance(q_quant, tuple):
            raise AssertionError("Hygon sparse indexer expects FP8 q_quant")

        # Keep the registered fake/mutation boundary during torch.compile.
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_quant,
            None,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            False,
        )

    _forward_hip_hygon._vllm_fl_hygon = True
    indexer_cls.forward_hip = _forward_hip_hygon
    logger.info("Patched DeepSeek-V4 sparse indexer with FL-only Hygon ops")


_flash_mla_fallback_warned: dict[str, bool] = {}

# 诊断: 记录前几次 flash_mla 尝试的结果与捕获状态(一次性, 定位服务期不生效原因)
_flash_mla_attempts = {"n_ok": 0, "n_fail": 0, "last_reason": None, "logged": 0}


def _flash_mla_attempt_log(success: bool, reason: str | None = None) -> None:
    st = _flash_mla_attempts
    if success:
        st["n_ok"] += 1
    else:
        st["n_fail"] += 1
        st["last_reason"] = reason
    total = st["n_ok"] + st["n_fail"]
    if total in (1, 50, 500) and st["logged"] < 3:
        st["logged"] += 1
        import torch as _torch

        logger.warning(
            "[flash_mla] attempt #%d: ok=%d fail=%d last_reason=%s "
            "torch_capturing=%s",
            total,
            st["n_ok"],
            st["n_fail"],
            st["last_reason"],
            _torch.cuda.is_current_stream_capturing(),
        )


def _patch_dv4_compile_model() -> None:
    """Plugin-side compile model wiring (replaces vllm source edits).

    1. @support_torch_compile on upstream DeepseekV4Model (from plugin,
       not vllm source)
    2. fine-grained attention custom ops
    3. splitting_ops only for piecewise capture
    4. aux stream injection (VLLM_ROCM_ENABLE_AUX_STREAMS=1)
    """
    import vllm.models.deepseek_v4.amd.model as amd_model
    from vllm.compilation.decorators import support_torch_compile

    if getattr(amd_model.DeepseekV4Model, "_vllm_fl_compile_patched", False):
        return

    from vllm_fl.models.hygon.deepseek_v4 import (
        _install_graph_boundaries,
        configure_attention_graph_boundaries,
    )

    original_init = amd_model.DeepseekV4Model.__init__

    def _patched_init(self, *, vllm_config, prefix=""):
        configure_attention_graph_boundaries(vllm_config.compilation_config)

        import os as _os
        aux_stream_list = None
        if (
            _os.environ.get("VLLM_ROCM_ENABLE_AUX_STREAMS", "0") == "1"
            and torch.cuda.is_available()
        ):
            aux_stream_list = [torch.cuda.Stream() for _ in range(3)]

        original_init(self, vllm_config=vllm_config, prefix=prefix)

        from vllm.model_executor.models.utils import PPMissingLayer
        for layer in self.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if aux_stream_list is not None:
                attn = getattr(layer, "attn", None)
                if attn is not None and getattr(attn, "aux_stream_list", None) is None:
                    attn.aux_stream_list = aux_stream_list
            _install_graph_boundaries(layer)

        logger.info(
            "Enabled Hygon DV4 compile boundaries for %d layers (aux=%s)",
            self.end_layer - self.start_layer,
            aux_stream_list is not None,
        )

    _patched_init._vllm_fl_compile_patched = True
    amd_model.DeepseekV4Model.__init__ = _patched_init

    # @support_torch_compile 重新挂到已 patch 的类上
    amd_model.DeepseekV4Model = support_torch_compile(amd_model.DeepseekV4Model)
    logger.info("Patched DeepseekV4Model with @support_torch_compile (plugin-side)")


def _warn_flash_mla_fallback_once(reason: str) -> None:
    if not _flash_mla_fallback_warned.get(reason):
        _flash_mla_fallback_warned[reason] = True
        logger.warning(
            "DeepSeek-V4 sparse attention falls back to vLLM Triton ragged "
            "kernels (%s)",
            reason,
        )


def _patch_attention_force_eager_break() -> None:
    """Force DV4 ``attention_impl`` to stay an eager segment in FULL captures.

    flash_mla 的 sched_meta 是一次性的: schedule 按当次 lens 规划, 烤进图后
    replay 复用过期调度 → 数值错误(实测偏差 >100%)。而 vllm 的
    ``@eager_break_during_capture`` 在 FULL 模式下放行捕获, 会让捕获期的
    flash_mla 回退 ragged 后被烤进 decode FULL 图 —— replay 期 attention
    python 不再执行, flash_mla 永远不生效(实测 serving 0 次)。

    此补丁在捕获激活时无条件走 ``add_eager``(绕过 FULL 放行): 捕获期
    attention 以 eager 段执行(flash_mla 正常运行), replay 期 eager 段以
    当步 forward context 的 metadata 现跑 flash_mla —— sched_meta 按步
    身份缓存(见 ``_SchedMetaCache``), 每步重新规划, 正确性保持。
    """
    import torch

    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    from vllm.utils.torch_utils import weak_ref_tensor

    attention_mod = importlib.import_module("vllm.models.deepseek_v4.attention")
    cls = attention_mod.DeepseekV4Attention
    original = cls.attention_impl
    if getattr(original, "_vllm_fl_force_eager", False):
        return

    @functools.wraps(original)
    def attention_impl_force_eager(self, *args, **kwargs):
        cap = BreakableCUDAGraphCapture.current()
        if cap is None or not cap._capturing:
            return original(self, *args, **kwargs)
        # 与 vllm 的 eager_break wrapper 相同的弱引用模式: 避免 replay lambda
        # 钉住 cudagraph pool 的 tensor 槽位。
        weak_args = tuple(
            weak_ref_tensor(a) if isinstance(a, torch.Tensor) else a for a in args
        )
        weak_kwargs = {
            k: weak_ref_tensor(v) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        return cap.add_eager(lambda: original(self, *weak_args, **weak_kwargs))

    attention_impl_force_eager._vllm_fl_force_eager = True
    cls.attention_impl = attention_impl_force_eager
    logger.info(
        "Patched DeepSeek-V4 attention_impl: force eager break under FULL "
        "capture (flash_mla single-use sched_meta)"
    )


def _patch_sparse_attn_flash_mla() -> None:
    """Route the DeepSeek-V4 main sparse attention to Hygon FlashMLA.

    The Hygon FlashMLA wheel reads the same paged fp8_ds_mla layout the
    plugin already writes, so only the compute leaf changes: decode goes to
    ``flash_mla_with_kvcache`` (sparse fp8, SWA + compressed topk segments)
    and prefill to ``flash_mla_sparse_fwd`` (bf16 gathered KV).  Measured on
    BW1000 the kernels are 2~17x faster than vLLM's Triton ragged kernels at
    the fixed business shapes (see bench_flash_mla_vs_ragged.py).

    Select with ``VLLM_FL_SPARSE_ATTN_IMPL`` = ``flash_mla`` (default) or
    ``triton``; any adapter refusal (``NotSupported``) or unexpected error
    falls back to the upstream path for that call.
    """
    import vllm_fl.envs as fl_envs

    flash_impl = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.hygon.impl.attention.sparse_attn_flash_mla"
    )
    if flash_impl._resolve_flash_mla() is None:
        logger.info(
            "FlashMLA not importable; keeping vLLM Triton ragged sparse attention"
        )
        return

    rocm_module = importlib.import_module("vllm.models.deepseek_v4.amd.rocm")

    # flash_mla 激活时, 阻止 FULL 捕获把回退 ragged 烤进 decode 图
    # (否则 replay 期 attention python 不执行, flash_mla 永不生效)。
    # 实测(trim6/TP4, dv4_20260909_fix2_graph): attention kernel 358→67us、
    # AG_i8 519→163us, 但每层 eager 的 host 抖动使 MoE custom AR
    # (cross_device_reduce_2stage) 125→820us(占 GPU 40.5%), 净吞吐
    # 3584→2432 tok/s(-32%)。默认关闭; eager 模式(无图)或后续 host
    # 减负后可开。VLLM_FL_FLASH_MLA_EAGER_BREAK=1 启用。
    if fl_envs.VLLM_FL_SPARSE_ATTN_IMPL == "flash_mla" and os.environ.get(
        "VLLM_FL_FLASH_MLA_EAGER_BREAK", "0"
    ) == "1":
        _patch_attention_force_eager_break()

    # ---- prefill: module-level leaf, dense indices + lens -------------------
    original_prefill = rocm_module.rocm_sparse_attn_prefill
    if not getattr(original_prefill, "_vllm_fl_hygon", False):

        @functools.wraps(original_prefill)
        def rocm_sparse_attn_prefill_hygon(
            q,
            kv,
            indices,
            topk_length,
            scale,
            head_dim,
            nope_head_dim,
            rope_head_dim,
            attn_sink,
            output,
            ragged_indices=None,
            ragged_indptr=None,
        ):
            if (
                fl_envs.VLLM_FL_SPARSE_ATTN_IMPL == "flash_mla"
                and ragged_indices is None
                and ragged_indptr is None
            ):
                try:
                    flash_impl.sparse_attn_prefill_flash_mla(
                        q=q,
                        kv=kv,
                        indices=indices,
                        topk_length=topk_length,
                        scale=scale,
                        head_dim=head_dim,
                        nope_head_dim=nope_head_dim,
                        rope_head_dim=rope_head_dim,
                        attn_sink=attn_sink,
                        output=output,
                    )
                    return
                except flash_impl.NotSupported as exc:
                    _warn_flash_mla_fallback_once(f"prefill: {exc}")
                except Exception as exc:  # noqa: BLE001 - never break serving
                    _warn_flash_mla_fallback_once(
                        f"prefill unexpected {type(exc).__name__}: {exc}"
                    )
            return original_prefill(
                q,
                kv,
                indices,
                topk_length,
                scale,
                head_dim,
                nope_head_dim,
                rope_head_dim,
                attn_sink,
                output,
                ragged_indices=ragged_indices,
                ragged_indptr=ragged_indptr,
            )

        rocm_sparse_attn_prefill_hygon._vllm_fl_hygon = True
        rocm_module.rocm_sparse_attn_prefill = rocm_sparse_attn_prefill_hygon

    # ---- decode: layer method so C4A (ratio 4) gets dense indices ----------
    attention_cls = rocm_module.DeepseekV4ROCMAiterMLAAttention
    original_forward_decode = attention_cls._forward_decode
    if getattr(original_forward_decode, "_vllm_fl_hygon", False):
        return

    def _decode_flash_mla(
        layer,
        q,
        kv_cache,
        swa_metadata,
        attn_metadata,
        swa_only,
        output,
    ):
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            if layer.compress_ratio == 4:
                # Dense slot mapping for the compressed cache: one fused
                # kernel replaces vLLM's ragged pack (FlashMLA wants dense).
                assert layer.topk_indices_buffer is not None
                assert swa_metadata.is_valid_token is not None
                topk_indices, topk_lens = flash_impl.compute_global_topk_indices_and_lens(
                    layer.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[: swa_metadata.num_decodes],
                    attn_metadata.block_size // layer.compress_ratio,
                    swa_metadata.is_valid_token[:num_decode_tokens],
                )
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        flash_impl.sparse_attn_decode_flash_mla(
            q=q,
            kv_cache=kv_cache if not swa_only else None,
            swa_k_cache=layer.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_metadata.decode_swa_lens,
            attn_sink=layer.attn_sink,
            scale=layer.scale,
            head_dim=layer.head_dim,
            nope_head_dim=layer.nope_head_dim,
            rope_head_dim=layer.rope_head_dim,
            output=output,
            step_token=swa_metadata,
        )

    @functools.wraps(original_forward_decode)
    def _forward_decode_hygon(
        self,
        q,
        kv_cache,
        swa_metadata,
        attn_metadata,
        swa_only,
        output,
    ):
        if fl_envs.VLLM_FL_SPARSE_ATTN_IMPL == "flash_mla":
            try:
                _decode_flash_mla(
                    self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output
                )
                _flash_mla_attempt_log(success=True)
                return
            except flash_impl.NotSupported as exc:
                _warn_flash_mla_fallback_once(f"decode: {exc}")
                _flash_mla_attempt_log(success=False, reason=f"NotSupported: {exc}")
            except Exception as exc:  # noqa: BLE001 - never break serving
                _warn_flash_mla_fallback_once(
                    f"decode unexpected {type(exc).__name__}: {exc}"
                )
                _flash_mla_attempt_log(
                    success=False, reason=f"{type(exc).__name__}: {exc}"
                )
        return original_forward_decode(
            self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output
        )

    _forward_decode_hygon._vllm_fl_hygon = True
    attention_cls._forward_decode = _forward_decode_hygon
    logger.info(
        "Patched DeepSeek-V4 sparse attention decode/prefill with Hygon FlashMLA"
    )


def _patch_deepseek_v4_bf16_fp32_gemm() -> None:
    """Route the compressor score projections to hipBLASLt on Hygon.

    ``DeepseekV4Attention.attn_gemm_parallel_execute`` computes both
    compressor projections with ``torch.mm(hidden, W.T, out_dtype=fp32)``.
    On this ROCm stack rocBLAS mis-picks the kernel for skinny decode
    batches (1268us for 64x1024x4096 = 0.08% peak; ~2.4ms/decode
    iteration across the 6 projections).  The replacement keeps the method
    identical except those two GEMMs, which go through
    ``linear_bf16_fp32_hygon`` (hipBLASLt, same bf16-in/fp32-out numerics,
    shape-routed; measured 1.4~38.9x, see performance_opt/dv4_model/gemm).

    ``VLLM_FL_BF16_FP32_GEMM=torch`` restores the upstream behavior.
    """
    import torch

    import vllm.envs as vllm_envs

    attention_mod = importlib.import_module("vllm.models.deepseek_v4.attention")
    attention_cls = attention_mod.DeepseekV4Attention
    original = attention_cls.attn_gemm_parallel_execute
    if getattr(original, "_vllm_fl_hygon", False):
        return

    gemm_impl = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.hygon.impl.gemm.bf16_fp32_gemm"
    )
    execute_in_parallel = attention_mod.execute_in_parallel

    def attn_gemm_parallel_execute_hygon(self, hidden_states):
        # Mirrors vllm-0.24.0 attention.py; only the two score GEMMs differ.
        aux_streams = self.aux_stream_list
        if aux_streams is not None:
            assert len(aux_streams) >= 3
            aux_streams = aux_streams[:3]

        aux_fns: list = [None, None, None]

        if self.compressor is not None:
            compressor = self.compressor

            def compressor_kv_score() -> torch.Tensor:
                return gemm_impl.linear_bf16_fp32_hygon(
                    hidden_states, compressor.fused_wkv_wgate.weight
                )

            aux_fns[0] = compressor_kv_score

        if self.indexer is not None:
            indexer = self.indexer

            def indexer_weights_proj() -> torch.Tensor:
                weights, _ = indexer.weights_proj(hidden_states)
                return weights

            def indexer_compressor_kv_score() -> torch.Tensor:
                return gemm_impl.linear_bf16_fp32_hygon(
                    hidden_states, indexer.compressor.fused_wkv_wgate.weight
                )

            aux_fns[1] = indexer_weights_proj
            aux_fns[2] = indexer_compressor_kv_score

        def fused_wqa_wkv() -> torch.Tensor:
            qr_kv, _ = self.fused_wqa_wkv(hidden_states)
            return qr_kv

        qr_kv, (kv_score, indexer_weights, indexer_kv_score) = execute_in_parallel(
            fused_wqa_wkv,
            aux_fns,
            self.ln_events[0],
            self.ln_events[1:4],
            aux_streams,
            enable=hidden_states.shape[0]
            <= vllm_envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
        )

        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    attn_gemm_parallel_execute_hygon._vllm_fl_hygon = True
    attention_cls.attn_gemm_parallel_execute = attn_gemm_parallel_execute_hygon
    logger.info(
        "Patched DeepSeek-V4 compressor projections with hipBLASLt bf16->fp32 GEMM"
    )


def _scale_inv_alias(param_name: str) -> str | None:
    """Return the FP8-style alias for an INT8 scale parameter name."""
    if param_name.endswith(".weight_scale"):
        return f"{param_name[: -len('.weight_scale')]}.weight_scale_inv"
    if param_name.endswith("_weight_scale"):
        return f"{param_name[: -len('_weight_scale')]}_weight_scale_inv"
    return None


def _install_compressed_tensors_scale_fallback(model_cls: type) -> None:
    """Install a target-aware compressed-tensors scale-name fallback.

    The upstream DeepSeek-V4 AMD mapper emits the FP8 name
    ``weight_scale_inv`` for checkpoint keys ending in ``.scale``. W8A8 INT8
    modules instead register ``weight_scale``. Keep the upstream name as the
    first choice, and expose an ``*_inv`` alias only when the corresponding
    non-inverse parameter exists and the inverse parameter does not.

    Injecting aliases into the local ``params_dict`` built by upstream
    ``load_weights`` avoids copying that version-sensitive method while still
    covering stacked attention parameters, fused MoE expert parameters, and
    ordinary parameters. The aliases exist only for the duration of loading.
    """
    original_load_weights = model_cls.load_weights
    if getattr(original_load_weights, "_vllm_fl_hygon_scale_fallback", False):
        return

    original_init = model_cls.__init__

    @functools.wraps(original_init)
    def _init_hygon_scale_fallback(self, *args, **kwargs) -> None:
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None and args:
            vllm_config = args[0]
        original_init(self, *args, **kwargs)
        self._vllm_fl_hygon_quant_config = getattr(vllm_config, "quant_config", None)

    @functools.wraps(original_load_weights)
    def _load_weights_hygon_scale_fallback(self, weights):
        quant_config = getattr(self, "_vllm_fl_hygon_quant_config", None)
        if quant_config is None or quant_config.get_name() != "compressed-tensors":
            return original_load_weights(self, weights)

        original_named_parameters = self.named_parameters
        alias_to_actual: dict[str, str] = {}

        def _named_parameters_with_scale_aliases(_self, *args, **kwargs):
            parameters = list(original_named_parameters(*args, **kwargs))
            existing_names = {name for name, _ in parameters}
            yield from parameters

            for actual_name, parameter in parameters:
                alias = _scale_inv_alias(actual_name)
                if alias is None or alias in existing_names:
                    continue
                alias_to_actual[alias] = actual_name
                yield alias, parameter

        had_instance_override = "named_parameters" in vars(self)
        previous_override: Any = vars(self).get("named_parameters")
        object.__setattr__(
            self,
            "named_parameters",
            MethodType(_named_parameters_with_scale_aliases, self),
        )
        try:
            loaded_params = original_load_weights(self, weights)
        finally:
            if had_instance_override:
                object.__setattr__(self, "named_parameters", previous_override)
            else:
                object.__delattr__(self, "named_parameters")

        if loaded_params is None:
            return None
        return {alias_to_actual.get(name, name) for name in loaded_params}

    _load_weights_hygon_scale_fallback._vllm_fl_hygon_scale_fallback = True
    model_cls.__init__ = _init_hygon_scale_fallback
    model_cls.load_weights = _load_weights_hygon_scale_fallback


def _patch_compressed_tensors_scale_fallback() -> None:
    """Patch DeepSeek-V4 AMD loading without changing its global mapper."""
    import vllm.models.deepseek_v4.amd.model as amd_model

    _install_compressed_tensors_scale_fallback(amd_model.DeepseekV4Model)
    logger.info("Patched DeepSeek-V4 compressed-tensors scale fallback for Hygon")


# Public names are used by the central Hygon patch orchestrator. Keep the
# private aliases above so existing downstream imports remain compatible.
patch_mhc_ops = _patch_mhc_ops
patch_deepseek_v4_cutedsl_selection = _patch_deepseek_v4_cutedsl_selection
patch_deepseek_v4_bf16_indexer_cache = _patch_deepseek_v4_bf16_indexer_cache
patch_deepseek_v4_qnorm_rope_kv_insert = _patch_deepseek_v4_qnorm_rope_kv_insert
patch_sparse_attn_indexer = _patch_sparse_attn_indexer
patch_sparse_attn_flash_mla = _patch_sparse_attn_flash_mla
patch_deepseek_v4_bf16_fp32_gemm = _patch_deepseek_v4_bf16_fp32_gemm
patch_compressed_tensors_scale_fallback = _patch_compressed_tensors_scale_fallback


__all__ = [
    "patch_compressed_tensors_scale_fallback",
    "patch_deepseek_v4_bf16_fp32_gemm",
    "patch_deepseek_v4_bf16_indexer_cache",
    "patch_deepseek_v4_cutedsl_selection",
    "patch_deepseek_v4_qnorm_rope_kv_insert",
    "patch_mhc_ops",
    "patch_sparse_attn_flash_mla",
    "patch_sparse_attn_indexer",
]
