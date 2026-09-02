# Copyright (c) 2026 BAAI. All rights reserved.
"""Runtime patches to run DeepSeek-V4 (MLA sparse + W8A8-INT8) on the
T-Head PPU backend with community vLLM.

Ported behaviors are referenced from the vendor's downstream vLLM fork
(0.20.1+ppu), which runs this model with:
  - int8 W8A8 MoE routed to a triton kernel path
  - FlashMLA sparse attention + fp8_ds_mla KV cache

All patches must be idempotent: general plugins load in the API server,
model-inspection, and every worker subprocess.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def _patch_int8_moe_quant_scheme():
    """Allow int8 W8A8 MoE on cuda-alike (non-NVIDIA) platforms.

    Upstream TritonExperts._supports_quant_scheme gates the
    (kInt8StaticChannelSym, kInt8DynamicTokenSym) pair behind
    current_platform.is_cuda() + SM7.5, which is NVIDIA-only.
    The triton int8 MoE kernels themselves are portable; the vendor fork
    runs them on PPU. Relax the check to is_cuda_alike().
    """
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
        TritonExperts,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        QuantKey,
        kInt8DynamicTokenSym,
        kInt8StaticChannelSym,
    )
    from vllm.platforms import current_platform

    if getattr(TritonExperts, "_fl_int8_patch", False):
        return

    orig = TritonExperts._supports_quant_scheme

    @staticmethod
    def _supports_quant_scheme(
        weight_key: "QuantKey | None",
        activation_key: "QuantKey | None",
    ) -> bool:
        if (weight_key, activation_key) == (
            kInt8StaticChannelSym,
            kInt8DynamicTokenSym,
        ) and current_platform.is_cuda_alike():
            return True
        return orig(weight_key, activation_key)

    TritonExperts._supports_quant_scheme = _supports_quant_scheme
    TritonExperts._fl_int8_patch = True
    logger.info("[vllm_fl] patched TritonExperts to accept int8 W8A8 MoE "
                "on cuda-alike platform %s", current_platform.device_name)


def _patch_flashmla_ops():
    """Route vllm.v1.attention.ops.flashmla to the PPU flash_mla pip package.

    The upstream module binds flash_mla_sparse_fwd/get_mla_metadata/... to
    vllm.third_party.flashmla (needs the compiled vllm._flashmla_C extension,
    absent in an empty-target build). The vendor fork instead imports the
    same-named APIs from the standalone `flash_mla` package on PPU. Rebind
    module attributes and availability checks accordingly.

    Must run before any downstream module does
    `from vllm.v1.attention.ops.flashmla import ...` (i.e. during platform
    plugin registration, prior to model import).
    """
    try:
        import flash_mla
    except ImportError:
        logger.warning("[vllm_fl] flash_mla package not found; "
                       "DeepSeek-V4 sparse attention will be unavailable")
        return

    import vllm.v1.attention.ops.flashmla as fl_mod

    if getattr(fl_mod, "_fl_thead_patch", False):
        return

    def _available(*_a, **_k):
        return True, None

    fl_mod._is_flashmla_available = _available
    fl_mod.is_flashmla_dense_supported = _available
    fl_mod.is_flashmla_sparse_supported = _available

    fl_mod.FlashMLASchedMeta = flash_mla.FlashMLASchedMeta
    fl_mod.flash_mla_sparse_fwd = flash_mla.flash_mla_sparse_fwd
    fl_mod.flash_mla_with_kvcache = flash_mla.flash_mla_with_kvcache
    fl_mod.get_mla_metadata = flash_mla.get_mla_metadata

    fl_mod._fl_thead_patch = True
    logger.info("[vllm_fl] flashmla ops rebound to PPU flash_mla package")


def _patch_int8_o_proj():
    """Swap DeepseekV4FlashMLAAttention._o_proj for the vendor INT8 path.

    Upstream _o_proj is fp8-only (fp8e4nv triton + deep_gemm fp8_einsum),
    neither available on PPU. The replacement dispatches per-layer: INT8
    wo_a uses the ported fp32-einsum path; anything else falls back to the
    original implementation.
    """
    import torch
    from vllm.models.deepseek_v4.nvidia.flashmla import (
        DeepseekV4FlashMLAAttention,
    )

    if getattr(DeepseekV4FlashMLAAttention, "_fl_o_proj_patch", False):
        return

    from vllm_fl.ops.deepseek_v4_o_proj import int8_o_proj

    orig_o_proj = DeepseekV4FlashMLAAttention._o_proj

    def _o_proj(self, o, positions):
        if self.wo_a.weight.dtype == torch.int8:
            return int8_o_proj(self, o, positions)
        return orig_o_proj(self, o, positions)

    DeepseekV4FlashMLAAttention._o_proj = _o_proj
    DeepseekV4FlashMLAAttention._fl_o_proj_patch = True
    logger.info("[vllm_fl] patched DeepseekV4FlashMLAAttention._o_proj "
                "with INT8 fp32-einsum path")


def _patch_topk_softplus_sqrt():
    """Route topk_softplus_sqrt routing to the pure-torch fallback.

    Upstream vllm_topk_softplus_sqrt calls the compiled
    _moe_C::topk_hash_softplus_sqrt except on XPU, where it uses
    _topk_softplus_sqrt_torch. The empty-target build has no _moe_C, so take
    the torch fallback on thead as well.
    """
    import vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router as r

    if getattr(r, "_fl_softplus_patch", False):
        return

    orig = r.vllm_topk_softplus_sqrt
    torch_impl = r._topk_softplus_sqrt_torch

    def vllm_topk_softplus_sqrt(*args, **kwargs):
        return torch_impl(*args, **kwargs)

    r.vllm_topk_softplus_sqrt = vllm_topk_softplus_sqrt
    r._fl_orig_vllm_topk_softplus_sqrt = orig
    r._fl_softplus_patch = True
    logger.info("[vllm_fl] topk_softplus_sqrt routed to torch fallback")


def _patch_moe_align_block_size():
    """Bridge vllm._custom_ops.moe_align_block_size to FlagGems triton.

    The upstream wrapper calls torch.ops._moe_C.moe_align_block_size
    (compiled ext, absent in empty builds). FlagGems ships an equivalent
    triton kernel with the same in-place contract. Resolved via the ops
    module attribute at call time, so patching the module function suffices
    (same approach as vllm_fl.patches.moe_sum).
    """
    import vllm._custom_ops as ops_module

    if getattr(ops_module.moe_align_block_size, "_fl_thead_patch", False):
        return

    def moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        expert_map=None,
    ):
        if expert_map is not None:
            # Passing expert_map here means ignore_invalid_experts=True
            # (EP ranks filtering foreign experts inside the kernel), which
            # the FlagGems kernel does not implement.
            raise NotImplementedError(
                "moe_align_block_size with in-kernel expert_map filtering "
                "is not supported by the FlagGems fallback (EP deployment)."
            )
        from flag_gems import moe_align_block_size_triton

        moe_align_block_size_triton(
            topk_ids,
            num_experts,
            block_size,
            sorted_token_ids,
            experts_ids,
            num_tokens_post_pad,
        )

    moe_align_block_size._fl_thead_patch = True
    ops_module.moe_align_block_size = moe_align_block_size
    logger.info("[vllm_fl] moe_align_block_size routed to FlagGems triton")


def _patch_sparse_indexer_ops():
    """Bridge the sparse-attn-indexer compiled ops to FlagGems triton.

    The DeepSeek sparse indexer calls four _C/_C_cache_ops kernels via
    vllm._custom_ops (all absent in empty builds); FlagGems v5.3+ ships
    signature-compatible triton implementations for each.
    """
    import vllm._custom_ops as ops_module

    if getattr(ops_module, "_fl_indexer_patch", False):
        return

    import flag_gems.fused as gf

    ops_module.indexer_k_quant_and_cache = gf.indexer_k_quant_and_cache
    ops_module.cp_gather_indexer_k_quant_cache = (
        gf.cp_gather_indexer_k_quant_cache
    )
    ops_module.top_k_per_row_prefill = gf.top_k_per_row_prefill
    ops_module.top_k_per_row_decode = gf.top_k_per_row_decode

    ops_module._fl_indexer_patch = True
    logger.info("[vllm_fl] sparse indexer cache/topk ops routed to FlagGems")


def _patch_indexer_q_quant():
    """Swap fused_indexer_q_rope_quant for the PPU int8 port.

    Upstream's FP8 path needs fp8e4nv triton codegen (CuteDSL or
    tl.float8e4nv, both NVIDIA-only). The replacement emits int8 Q with the
    scale folded into weights — accepted directly by the PPU deep_gemm
    mqa-logits kernels. Both the defining module and attention.py (which
    from-imports the symbol) must be patched.
    """
    import vllm.models.deepseek_v4.attention as attn_mod
    import vllm.models.deepseek_v4.common.ops.fused_indexer_q as fiq_mod

    if getattr(fiq_mod, "_fl_thead_patch", False):
        return

    from vllm_fl.ops.deepseek_v4_indexer_q import fused_indexer_q_rope_quant_ppu

    fiq_mod.fused_indexer_q_rope_quant = fused_indexer_q_rope_quant_ppu
    attn_mod.fused_indexer_q_rope_quant = fused_indexer_q_rope_quant_ppu
    fiq_mod._fl_thead_patch = True
    logger.info("[vllm_fl] indexer Q quant routed to PPU int8 port")


def _patch_sparse_indexer_forward():
    """Route SparseAttnIndexer's forward to the ported PPU indexer op.

    forward_native rejects non-CUDA/ROCm/XPU platforms, and forward_cuda's
    torch.ops.vllm.sparse_attn_indexer drives fp8_fp4_mqa_logits, which
    asserts q.dtype == k.dtype — but our indexer Q is INT8 on PPU. The
    ported op dispatches to deep_gemm int8_(paged_)mqa_logits instead.
    """
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

    if getattr(SparseAttnIndexer, "_fl_thead_patch", False):
        return

    # Import registers torch.ops.vllm.fl_ppu_sparse_attn_indexer.
    import vllm_fl.ops.deepseek_v4_ppu_indexer  # noqa: F401
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
    from vllm.utils.torch_utils import _encode_layer_name

    # Same contract as upstream sparse_attn_indexer's decorator: the body
    # has data-dependent shapes (workspace slicing, chunk loops), which must
    # run eagerly during breakable-cudagraph capture, not be recorded.
    @eager_break_during_capture
    def _run_ppu_indexer(*args):
        return torch.ops.vllm.fl_ppu_sparse_attn_indexer(*args)

    def forward_ppu(self, hidden_states, q_quant, k, weights):
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return _run_ppu_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
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
            self.use_fp4_cache,
        )

    import torch

    SparseAttnIndexer.forward_oot = forward_ppu
    SparseAttnIndexer.forward_native = forward_ppu
    SparseAttnIndexer._fl_thead_patch = True
    logger.info("[vllm_fl] SparseAttnIndexer forward -> PPU int8 indexer op")


def _patch_compressor_cache_insert():
    """Swap the compressor cache-insert launcher for the SM80/PPU port.

    Upstream kernels hard-cast with tl.float8e4nv; the ported variants use
    the software _encode_e4m3fn instead and drop the NVIDIA launch_pdl
    kwarg. compressor.py binds the launcher via module attribute at call
    time in the non-NVIDIA branch, but also from-imports it, so patch both
    modules.
    """
    import vllm.models.deepseek_v4.compressor as comp_mod
    import vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache as fcq_mod

    if getattr(fcq_mod, "_fl_thead_patch", False):
        return

    from vllm_fl.ops.deepseek_v4_compress_cache import (
        compress_norm_rope_store_triton_ppu,
    )

    fcq_mod.compress_norm_rope_store_triton = compress_norm_rope_store_triton_ppu
    comp_mod.compress_norm_rope_store_triton = compress_norm_rope_store_triton_ppu
    fcq_mod._fl_thead_patch = True
    logger.info("[vllm_fl] compressor cache insert routed to SM80/PPU port")


def _patch_indexer_num_sms():
    """Make the MLA indexer size its scheduler buffer via deep_gemm.

    Upstream sizes scheduler_metadata_buffer as (num_compute_units + 1, 2);
    PPU deep_gemm's paged mqa-logits kernel validates the buffer against
    its own get_num_sms() (20 on PPU, vs 64 multiprocessors reported by
    the device), so the vendor fork uses deep_gemm's number. Rebind the
    module-level num_compute_units reference in the indexer.
    """
    import vllm.v1.attention.backends.mla.indexer as idx_mod

    if getattr(idx_mod, "_fl_thead_patch", False):
        return

    import deep_gemm

    def num_compute_units_ppu(device_id: int = 0) -> int:
        return int(deep_gemm.get_num_sms())

    idx_mod.num_compute_units = num_compute_units_ppu
    idx_mod._fl_thead_patch = True
    logger.info("[vllm_fl] indexer num_sms -> deep_gemm.get_num_sms()")


def _patch_int8_weights_mapper():
    """Add the vendor's INT8 branch to the DeepSeek-V4 weights mapper.

    This checkpoint's config still says expert_dtype="fp4" (inherited from
    the base model) but the weights are compressed-tensors INT8, whose
    quant method registers per-channel scales as ``weight_scale`` — the
    fp4 mapper's blanket ``.scale -> .weight_scale_inv`` rule then breaks
    loading (KeyError on fused_wqa_wkv.weight_scale_inv). The vendor fork
    maps every ``.scale`` to ``.weight_scale`` for int8/int4 checkpoints.
    Wrap the model __init__ to install that mapper when the active quant
    config is compressed-tensors.
    """
    import re

    import vllm.models.deepseek_v4.nvidia.model as nv_model

    if getattr(nv_model, "_fl_mapper_patch", False):
        return

    orig_make = nv_model._make_deepseek_v4_weights_mapper

    def _make_int8_mapper():
        mapper = orig_make("fp8")  # closest base: no expert .scale special-case
        # Replace the scale rule: int8 per-channel scales register as
        # ``weight_scale`` everywhere (dense, fused, MoE experts).
        mapper.orig_to_new_regex = {
            re.compile(r"\.scale$"): ".weight_scale",
        }
        return mapper

    cls = nv_model.DeepseekV4ForCausalLM
    orig_init = cls.__init__

    import functools

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        vllm_config = kwargs.get("vllm_config") or (args[0] if args else None)
        quant_config = getattr(vllm_config, "quant_config", None)
        method = getattr(quant_config, "get_name", lambda: None)()
        if method == "compressed-tensors":
            self.hf_to_vllm_mapper = _make_int8_mapper()
            logger.warning("[vllm_fl] int8 weights mapper installed "
                           "(.scale -> .weight_scale)")

    cls.__init__ = __init__
    nv_model._fl_mapper_patch = True


def _patch_disable_cutedsl():
    """Force has_cutedsl() to False on PPU.

    The container ships nvidia_cutlass_dsl (satisfying the `cutlass` module
    probe), so upstream's CuteDSL fast paths activate and then fail on the
    NVIDIA-only `quack` dependency and SM90+ codegen. Rebind the probe in
    every module that from-imported it so the triton fallbacks are taken.
    """
    import vllm.utils.import_utils as iu
    import vllm.models.deepseek_v4.common.ops.cache_utils as cu
    import vllm.models.deepseek_v4.common.ops.fused_indexer_q as fiq

    if getattr(iu, "_fl_cutedsl_patch", False):
        return

    def has_cutedsl() -> bool:
        return False

    iu.has_cutedsl = has_cutedsl
    cu.has_cutedsl = has_cutedsl
    fiq.has_cutedsl = has_cutedsl
    iu._fl_cutedsl_patch = True
    logger.info("[vllm_fl] has_cutedsl forced False (quack/CuteDSL is "
                "NVIDIA-only); triton fallbacks in effect")


def _patch_dequant_gather():
    """Swap the K-cache dequant-gather for the SM80/PPU software-decode port.

    Even with CuteDSL disabled, the upstream triton fallback bitcasts
    uint8 -> tl.float8e4nv, which PPU triton refuses to compile. The vendor
    SM80 kernel decodes E4M3FN in software. cache_utils is the defining
    module; sparse_mla/flashmla re-import the symbol, so patch by module
    attribute everywhere it is from-imported.
    """
    import vllm.models.deepseek_v4.common.ops.cache_utils as cu

    if getattr(cu, "_fl_dequant_patch", False):
        return

    from vllm_fl.ops.deepseek_v4_dequant_gather import (
        dequantize_and_gather_k_cache_ppu,
    )

    cu.dequantize_and_gather_k_cache = dequantize_and_gather_k_cache_ppu
    # Re-imported symbol holders:
    import importlib

    for mod_name in (
        "vllm.models.deepseek_v4.sparse_mla",
        "vllm.models.deepseek_v4.nvidia.flashmla",
        "vllm.models.deepseek_v4.common.ops",
    ):
        try:
            m = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(m, "dequantize_and_gather_k_cache"):
            m.dequantize_and_gather_k_cache = dequantize_and_gather_k_cache_ppu

    cu._fl_dequant_patch = True
    logger.info("[vllm_fl] dequant-gather K cache routed to SM80/PPU port")


def _patch_int8_moe_deepgemm_backend():
    """Register the ported PPUDeepGemmExperts as the preferred int8 MoE.

    The vendor fork's oracle tries PPU_DEEPGEMM > ACEXT > BATCHED > TRITON on
    PPU; upstream only knows TRITON. Wrap select_int8_moe_backend to try the
    ported PPUDeepGemmExperts first (same is_supported_config contract) and
    fall back to the original selection if it rejects the deployment.
    """
    import vllm.model_executor.layers.fused_moe.oracle.int8 as oracle
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk

    if getattr(oracle, "_fl_deepgemm_patch", False):
        return

    from vllm_fl.ops.ppu_deep_gemm_moe import PPUDeepGemmExperts

    orig_select = oracle.select_int8_moe_backend

    def select_int8_moe_backend(config, weight_key=None, activation_key=None, **kw):
        # Preserve original default keys if caller relied on them.
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kInt8DynamicTokenSym,
            kInt8StaticChannelSym,
        )

        wk = weight_key if weight_key is not None else kInt8StaticChannelSym
        ak = activation_key if activation_key is not None else kInt8DynamicTokenSym

        activation_format = (
            mk.FusedMoEActivationFormat.BatchedExperts
            if config.moe_parallel_config.use_batched_activation_format
            else mk.FusedMoEActivationFormat.Standard
        )
        if config.moe_backend == "auto":
            import os

            if os.environ.get("VLLM_FL_DISABLE_DEEPGEMM_MOE") == "1":
                logger.warning(
                    "[vllm_fl] PPUDeepGemmExperts disabled via env; "
                    "using upstream selection"
                )
                return orig_select(config, weight_key, activation_key, **kw)
            supported, reason = PPUDeepGemmExperts.is_supported_config(
                PPUDeepGemmExperts, config, wk, ak, activation_format
            )
            if supported:
                logger.warning(
                    "[vllm_fl] Using ported PPUDeepGemmExperts int8 MoE backend"
                )
                return oracle.Int8MoeBackend.TRITON, PPUDeepGemmExperts
            logger.warning(
                "[vllm_fl] PPUDeepGemmExperts rejected config (%s); "
                "falling back to upstream selection", reason,
            )
        return orig_select(config, weight_key, activation_key, **kw)

    oracle.select_int8_moe_backend = select_int8_moe_backend
    # compressed_tensors_moe imports the symbol at module load; rebind there too.
    import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 as ct_int8  # noqa: E501

    if hasattr(ct_int8, "select_int8_moe_backend"):
        ct_int8.select_int8_moe_backend = select_int8_moe_backend

    # Vendor behavior: on PPU always build a W8A8 quant config even with
    # dynamic per-token activation scales (a1_scale is None). Upstream's
    # maker falls back to W8A16 in that case, which turns quant_dtype into
    # None and routes PPUDeepGemmExperts onto its bf16 path (dtype assert).
    from vllm.model_executor.layers.fused_moe.config import (
        int8_w8a8_moe_quant_config,
    )

    orig_make_quant = oracle.make_int8_moe_quant_config

    def make_int8_moe_quant_config_ppu(
        w1_scale,
        w2_scale,
        a1_scale=None,
        a2_scale=None,
        w1_bias=None,
        w2_bias=None,
        per_act_token_quant=False,
    ):
        import os

        if os.environ.get("VLLM_FL_DISABLE_DEEPGEMM_MOE") == "1":
            # A/B escape hatch: keep upstream W8A16 fallback semantics that
            # TritonExperts expects.
            return orig_make_quant(
                w1_scale, w2_scale, a1_scale, a2_scale,
                w1_bias, w2_bias, per_act_token_quant,
            )
        return int8_w8a8_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            per_act_token_quant=per_act_token_quant,
        )

    oracle.make_int8_moe_quant_config = make_int8_moe_quant_config_ppu
    if hasattr(ct_int8, "make_int8_moe_quant_config"):
        ct_int8.make_int8_moe_quant_config = make_int8_moe_quant_config_ppu
    oracle._fl_deepgemm_patch = True


def _patch_asymmetric_capture_sizes():
    """Sparse PIECEWISE sizes, dense FULL sizes for breakable cudagraphs.

    With BreakableCUDAGraphWrapper each PIECEWISE batch size persists its
    inter-segment activations (~3.5 MiB/token, measured on PPU) — a dense
    size list OOMs. FULL decode graphs are monolithic and reuse the pool
    across sizes (~1 MiB/graph). So: keep every capture size for FULL
    (kills decode padding waste) and thin PIECEWISE to a sparse subset.
    """
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    from vllm.config import CUDAGraphMode
    import os

    if getattr(CudagraphDispatcher, "_fl_asym_patch", False):
        return
    # Applies in BOTH breakable and compile modes: on PPU each PIECEWISE
    # graph's inter-segment/output buffers are not reclaimed across sizes
    # (~1.6-2.2 GiB per size measured), so the mixed-batch size list must
    # stay sparse. FULL decode graphs reuse the pool (~70 MiB/size) and
    # keep the dense list.

    PIECEWISE_KEEP = (1, 8, 64, 512)

    orig_init_keys = CudagraphDispatcher.initialize_cudagraph_keys

    def initialize_cudagraph_keys(self, cudagraph_mode, uniform_decode_query_len=1):
        orig_init_keys(self, cudagraph_mode, uniform_decode_query_len)
        pw_keys = self.cudagraph_keys.get(CUDAGraphMode.PIECEWISE)
        if not pw_keys:
            return
        kept = {d for d in pw_keys if d.num_tokens in PIECEWISE_KEEP}
        dropped = len(pw_keys) - len(kept)
        if dropped > 0:
            self.cudagraph_keys[CUDAGraphMode.PIECEWISE] = kept
            logger.warning(
                "[vllm_fl] thinned PIECEWISE cudagraph sizes: kept %d, "
                "dropped %d (breakable-graph per-size memory cost)",
                len(kept), dropped,
            )

    CudagraphDispatcher.initialize_cudagraph_keys = initialize_cudagraph_keys
    CudagraphDispatcher._fl_asym_patch = True


def _patch_torch_compile_model():
    """Vendor-style full-graph mode: decorate DeepseekV4Model with
    support_torch_compile so PIECEWISE graphs come from the vLLM compile
    pipeline instead of BreakableCUDAGraphWrapper (whose per-size segment
    activations cost ~1.6 GiB/size and forced us down to 8 sizes).

    Activated only when VLLM_FL_DSV4_TORCH_COMPILE=1: the compile pipeline
    on the empty-target PPU build is unproven, so keep breakable as default.
    Also pre-set VLLM_USE_BREAKABLE_CUDAGRAPH=0 (the DeepseekV4 auto-enable
    in VllmConfig would otherwise force CompilationMode.NONE).
    """
    import os

    if os.environ.get("VLLM_FL_DSV4_TORCH_COMPILE") != "1":
        return

    os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")

    # Our opaque attention/indexer ops must be graph-splitting ops, else
    # PIECEWISE capture bakes attention (metadata-dependent!) into the graph
    # and replays garbage. Upstream's list only knows the NVIDIA op names.
    from vllm.config.compilation import CompilationConfig

    for op in ("vllm::fl_dsv4_attention", "vllm::fl_ppu_sparse_attn_indexer"):
        if op not in CompilationConfig._attention_ops:
            CompilationConfig._attention_ops.append(op)

    # Pre-warm every @cache'd module probe reachable from the compiled
    # forward: dynamo refuses to trace importlib.util.find_spec, but a
    # cache-hit lru_cache call is constant-folded.
    try:
        from vllm.utils.import_utils import has_deep_gemm, has_cutedsl

        has_deep_gemm()
        has_cutedsl()
        from vllm_fl.ops.ppu_deep_gemm import (
            is_deep_gemm_supported,
            get_deep_gemm_best_configs,
            get_mk_alignment_for_contiguous_layout,
        )

        is_deep_gemm_supported()
        get_deep_gemm_best_configs()
        get_mk_alignment_for_contiguous_layout(is_blockwise=False)
        get_mk_alignment_for_contiguous_layout(is_blockwise=True)

        # dynamo traces through even functools.cache'd functions; freeze
        # runtime module probes reachable from the compiled forward into
        # constant lambdas so no importlib call remains in the graph.
        import vllm.utils.deep_gemm as up_dg

        _dg_supported = bool(up_dg.is_deep_gemm_supported())
        up_dg.is_deep_gemm_supported = lambda: _dg_supported
        import vllm.utils.import_utils as iu2

        _has_dg = bool(iu2.has_deep_gemm())
        iu2.has_deep_gemm = lambda: _has_dg
    except Exception as e:  # pragma: no cover - best effort
        logger.warning("[vllm_fl] probe pre-warm failed: %s", e)

    import vllm.models.deepseek_v4.nvidia.model as nv_model

    if getattr(nv_model, "_fl_compile_patch", False):
        return

    from vllm.compilation.decorators import support_torch_compile

    # mHC: opaque custom ops around the tilelang kernels. A fully-native
    # torch version (vllm_fl/ops/deepseek_v4_mhc_native.py, verified
    # 1-ulp-equivalent) was tried but inductor's generated kernels blow up
    # the PPU triton backend (ppu-llc CalledProcessError) and compile time
    # (~32 min/graph); revisit if the toolchain matures.
    from vllm_fl.ops.deepseek_v4_mhc_ops import (
        hc_head_fused_opaque,
        mhc_fused_post_pre_opaque,
        mhc_post_opaque,
        mhc_pre_opaque,
    )

    nv_model.mhc_pre_tilelang = mhc_pre_opaque
    nv_model.mhc_post_tilelang = mhc_post_opaque
    nv_model.mhc_fused_post_pre_tilelang = mhc_fused_post_pre_opaque
    nv_model.hc_head_fused_kernel_tilelang = hc_head_fused_opaque

    # Route attention_impl through an opaque custom op (vendor-style
    # torch.ops.vllm.deepseek_v4_attention): dynamo cannot trace workspace
    # management / FlashMLA / indexer internals.
    import vllm_fl.ops.deepseek_v4_attn_op  # noqa: F401 (registers the op)
    import vllm.models.deepseek_v4.attention as attn_mod

    _orig_attn_forward = attn_mod.DeepseekV4Attention.forward

    def _compiled_attn_forward(self, positions, hidden_states, llama_4_scaling=None):
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self.attn_gemm_parallel_execute(hidden_states)
        )
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr, kv = attn_mod.fused_q_kv_rmsnorm(
            qr, kv, self.q_norm.weight.data, self.kv_norm.weight.data, self.eps
        )
        torch.ops.vllm.fl_dsv4_attention(
            hidden_states, qr, kv, kv_score, indexer_kv_score,
            indexer_weights, positions, o_padded, self.prefix,
        )
        o = o_padded[:, : self.n_local_heads, :]
        return self._o_proj(o, positions)

    attn_mod.DeepseekV4Attention.forward = _compiled_attn_forward

    nv_model.DeepseekV4Model = support_torch_compile(nv_model.DeepseekV4Model)
    # DeepseekV4ForCausalLM binds the class attribute at definition time.
    nv_model.DeepseekV4ForCausalLM.model_cls = nv_model.DeepseekV4Model
    nv_model._fl_compile_patch = True
    logger.warning("[vllm_fl] DeepseekV4Model wrapped with "
                   "support_torch_compile (full-graph mode)")




def _patch_empty_int_zero():
    """Zero-fill only integer aten::empty allocations (see module docstring
    of vllm_fl.ops.empty_int_zero). Bridges the gap left by upstream
    FlagGems PR #5438 for consumers that rely on zeroed index buffers."""
    from vllm_fl.ops import empty_int_zero

    if empty_int_zero.register():
        logger.warning("[vllm_fl] aten::empty int-dtype zero-fill installed")



def _patch_topk_indices_buffer_init():
    """Fully initialize DeepseekV4Model.topk_indices_buffer.

    Upstream allocates it with torch.empty (max_num_batched_tokens x
    index_topk, int32) and each step only writes the rows for the current
    tokens (`buffer[:num_tokens] = -1`). Padding rows / unused columns keep
    whatever was in memory, and CUDA-graph replay reads the full captured
    extent -> garbage int32 indices -> out-of-bounds gather in the sparse
    attention (illegal memory access).

    This was masked while FlagGems overrode aten::empty with a zero-filling
    triton kernel; upstream FlagGems PR #5438 removed that override, so the
    buffer must be initialized explicitly. -1 is the sentinel the indexer
    itself uses for "no index".
    """
    import vllm.models.deepseek_v4.nvidia.model as nv_model

    if getattr(nv_model, "_fl_topk_buf_patch", False):
        return

    cls = nv_model.DeepseekV4Model
    orig_init = cls.__init__

    import functools

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        buf = getattr(self, "topk_indices_buffer", None)
        if buf is not None:
            buf.fill_(-1)

    cls.__init__ = __init__
    nv_model._fl_topk_buf_patch = True
    logger.warning("[vllm_fl] topk_indices_buffer fully initialized to -1")


def apply_deepseek_v4_thead_patches():
    """Entry point called from vllm_fl.register_model()."""
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "thead":
        return

    _patch_int8_moe_quant_scheme()
    _patch_int8_moe_deepgemm_backend()
    _patch_flashmla_ops()
    _patch_int8_o_proj()
    _patch_topk_softplus_sqrt()
    _patch_moe_align_block_size()
    _patch_sparse_indexer_ops()
    _patch_indexer_q_quant()
    _patch_sparse_indexer_forward()
    _patch_compressor_cache_insert()
    _patch_indexer_num_sms()
    _patch_asymmetric_capture_sizes()
    _patch_topk_indices_buffer_init()
    _patch_empty_int_zero()
    _patch_torch_compile_model()
    _patch_int8_weights_mapper()
    _patch_disable_cutedsl()
    _patch_dequant_gather()
