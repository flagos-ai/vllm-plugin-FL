# Copyright (c) 2026 BAAI. All rights reserved.

"""DeepSeek-V4-specific Hygon model and attention compatibility patches."""

import functools
import importlib
import logging
from types import MethodType
from typing import Any

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
        if (
            getattr(self, "use_bf16_cache", False)
            or self.k_cache.kv_cache.dtype == torch.bfloat16
        ):
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

        return sparse_indexer.sparse_attn_indexer(
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
patch_compressed_tensors_scale_fallback = _patch_compressed_tensors_scale_fallback


__all__ = [
    "patch_compressed_tensors_scale_fallback",
    "patch_deepseek_v4_bf16_indexer_cache",
    "patch_deepseek_v4_cutedsl_selection",
    "patch_deepseek_v4_qnorm_rope_kv_insert",
    "patch_mhc_ops",
    "patch_sparse_attn_indexer",
]
