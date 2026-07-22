# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

"""
Kunlunxin platform monkey-patches.

Follows the same pattern as ascend/patch.py — all Kunlunxin-specific
overrides are applied here instead of scattering `if platform == "kunlunxin"`
guards across shared code.

Called once during plugin initialization (register_oot_ops).

Note: patches/patch_fla_utils.py (ensure_fla_compat) is NOT called from here.
It must run earlier — in register_model() — before any FLA module import,
to prevent torch.xpu.get_device_name crash. See vllm_fl/__init__.py.
"""

import logging

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_kunlunxin_patches():
    """Apply all Kunlunxin-specific patches. Idempotent."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    # Disable Triton kernels incompatible with Kunlunxin XPU
    import os
    os.environ.setdefault("VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE", "0")

    # RESTORED from old version: Critical Triton kernel compatibility patches
    patch_block_table_slot_mapping()
    patch_attention_backend_registry()
    patch_topk_topp_sampler()
    patch_fused_moe()

    # Existing patches
    patch_causal_conv1d()
    patch_fla_ops()
    patch_fused_gdn_gating()
    patch_ssm_cache_update()
    patch_sampler_rng()
    logger.info("Applied all Kunlunxin patches")


# ── RESTORED: block_table slot_mapping (Triton kernel bypass) ──
def patch_block_table_slot_mapping():
    """Replace Triton-based compute_slot_mapping with CPU numpy implementation.

    vLLM 0.20 moved compute_slot_mapping to a Triton kernel which fails on
    Kunlunxin XPU (err_code -714). Replace with a numpy-based approach.
    """
    try:
        import numpy as np
        import torch
        from vllm.v1.worker.block_table import BlockTable

        PAD_SLOT_ID = -1

        def compute_slot_mapping_cpu(self, num_reqs, query_start_loc, positions):
            query_start_loc_cpu = query_start_loc.cpu().numpy()
            positions_cpu = positions.cpu().numpy()
            num_tokens = positions_cpu.shape[0]

            total_cp_world_size = self.pcp_world_size * self.dcp_world_size
            total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank

            # Build req_indices from query_start_loc
            req_indices = np.zeros(num_tokens, dtype=np.int64)
            for i in range(num_reqs):
                start = int(query_start_loc_cpu[i])
                end = int(query_start_loc_cpu[i + 1])
                req_indices[start:end] = i

            if total_cp_world_size > 1:
                virtual_block_size = self.block_size * total_cp_world_size
                block_table_indices = (
                    req_indices * self.max_num_blocks_per_req
                    + positions_cpu // virtual_block_size
                )
                block_numbers = self.block_table.np.ravel()[block_table_indices]
                virtual_block_offsets = positions_cpu % virtual_block_size
                mask = (
                    virtual_block_offsets // self.cp_kv_cache_interleave_size
                    % total_cp_world_size == total_cp_rank
                )
                block_offsets = (
                    virtual_block_offsets
                    // (total_cp_world_size * self.cp_kv_cache_interleave_size)
                    * self.cp_kv_cache_interleave_size
                    + virtual_block_offsets % self.cp_kv_cache_interleave_size
                )
                slot_mapping = block_numbers * self.block_size + block_offsets
                self.slot_mapping.np[:num_tokens] = np.where(mask, slot_mapping, PAD_SLOT_ID)
            else:
                block_table_indices = (
                    req_indices * self.max_num_blocks_per_req
                    + positions_cpu // self.block_size
                )
                block_numbers = self.block_table.np.ravel()[block_table_indices]
                block_offsets = positions_cpu % self.block_size
                np.add(
                    block_numbers * self.block_size,
                    block_offsets,
                    out=self.slot_mapping.np[:num_tokens],
                )

            # Pad remaining slots
            self.slot_mapping.np[num_tokens:self.max_num_batched_tokens] = PAD_SLOT_ID
            # Copy to GPU
            self.slot_mapping.copy_to_gpu(self.max_num_batched_tokens)

        BlockTable.compute_slot_mapping = compute_slot_mapping_cpu
        logger.info("Patched BlockTable.compute_slot_mapping to CPU numpy path for Kunlunxin")
    except Exception as e:
        logger.warning("Failed to patch compute_slot_mapping: %s", e)


# ── RESTORED: attention backend registry ──
def patch_attention_backend_registry():
    """Register KUNLUNXIN_FL as CUSTOM in AttentionBackendEnum.

    vLLM 0.20+ validates get_name() against AttentionBackendEnum.
    Register our backend under CUSTOM so the lookup succeeds.
    """
    try:
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention.KunlunxinAttentionBackend"
        )
        logger.info("Registered KunlunxinAttentionBackend as CUSTOM attention backend")
    except Exception as e:
        logger.warning("Failed to register attention backend: %s", e)


# ── RESTORED: topk_topp_sampler ──
def patch_topk_topp_sampler():
    """Force PyTorch-native top-k/top-p on Kunlunxin.

    The vLLM Triton top-k/top-p kernel (topk_topp_triton.py) triggers
    kernel launch failures (error 719) on P800. Route through the PyTorch
    path instead.
    """
    try:
        import vllm.v1.sample.ops.topk_topp_sampler as sampler_mod
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

        sampler_mod.apply_top_k_top_p = apply_top_k_top_p_pytorch
        logger.info("Patched apply_top_k_top_p to use PyTorch-native path for Kunlunxin")
    except Exception as e:
        logger.warning("Failed to patch top-k/top-p sampler for Kunlunxin: %s", e)


# ── RESTORED: fused_moe ──
def patch_fused_moe():
    """Replace fused_experts_impl with Kunlunxin implementation."""
    try:
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fused_moe.fused_moe import (
            fused_experts_impl as klx_fused_experts_impl,
        )
        import vllm_fl.ops.fused_moe.fused_moe as fused_moe_lib

        fused_moe_lib.fused_experts_impl = klx_fused_experts_impl
        logger.info("Patched fused_moe for Kunlunxin")
    except Exception as e:
        logger.warning("Failed to patch fused_moe ops: %s", e)


# ── sampler RNG (TP-consistent unseeded sampling) ──
def patch_sampler_rng():
    """Make random sampling TP-consistent on Kunlunxin XPU.

    Root cause: on torch_xmlir the *default/global* RNG does NOT honor
    ``manual_seed`` for ``Tensor.exponential_()`` and is non-deterministic
    across processes. vLLM's ``random_sample`` uses the default RNG (no
    explicit ``generator``) whenever a request has no per-request seed, and
    relies on every tensor-parallel rank drawing *identical* noise (which
    holds on CUDA). On XPU the TP ranks diverge -> each rank's
    ``argmax(probs / q)`` picks a different next token -> per-rank model/KV
    state diverges -> the following all-reduce combines inconsistent states
    -> garbled / repetitive output. Greedy (argmax, no RNG) and TP=1 (single
    rank) are unaffected.

    Fix: keep the per-rank default RNG (so sampling stays varied), then
    broadcast rank-0's sampled token ids across the TP group so every rank
    proceeds with the same next token and can no longer diverge.
    """
    try:
        import vllm.v1.sample.ops.topk_topp_sampler as _sampler_mod

        _orig_random_sample = _sampler_mod.random_sample

        def _broadcast_random_sample(probs, generators):
            sampled_tokens = _orig_random_sample(probs, generators)
            try:
                from vllm.distributed import get_tp_group
                tp_group = get_tp_group()
                if tp_group.world_size > 1:
                    tp_group.broadcast(sampled_tokens, src=0)
            except Exception as e:
                logger.warning(f"Failed to broadcast tokens in TP group: {e}")
            return sampled_tokens

        _sampler_mod.random_sample = _broadcast_random_sample
        logger.info(
            "Patched sampler random_sample for TP-consistent sampling on Kunlunxin (broadcast only)"
        )
    except Exception as e:
        logger.warning("Failed to patch sampler RNG: %s", e)


# ── causal_conv1d ──
def patch_causal_conv1d():
    """Replace causal_conv1d_fn / causal_conv1d_update with Kunlunxin impls.

    Upstream convention:
        causal_conv1d_fn:  x=(dim, cu_seqlen), conv_states=(..., dim, state_len) — NCW
        causal_conv1d_update: conv_state=(..., dim, state_len) — NCW
    Kunlunxin kernel convention:
        x=(cu_seqlen, dim), conv_states=(N, state_len, dim) — NWC, is_ncw=False

    The wrappers bridge NCW ↔ NWC so the model code follows the upstream convention.
    """
    try:
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as _conv1d_lib
        import vllm.model_executor.models.qwen3_next as _qwen3_next_lib

        from vllm_fl.dispatch import resolve_op

        _klx_conv1d_fn = resolve_op("causal_conv1d_fn")
        _klx_conv1d_update = resolve_op("causal_conv1d_update")

        def causal_conv1d_fn_adapter(
            x, weight, bias, conv_states, query_start_loc, **kwargs
        ):
            # NCW → NWC: conv_states view, x transpose
            conv_states_nwc = conv_states.transpose(-1, -2)
            x_nwc = x.transpose(0, 1).contiguous()
            out = _klx_conv1d_fn(
                x_nwc, weight, bias, conv_states_nwc, query_start_loc, **kwargs
            )
            # NWC → NCW: transpose output back
            return out.transpose(0, 1)

        def causal_conv1d_update_adapter(
            x, conv_state, weight, bias=None, activation=None, **kwargs
        ):
            # NCW → NWC: conv_state view
            conv_state_nwc = conv_state.transpose(-1, -2)
            return _klx_conv1d_update(
                x, conv_state_nwc, weight, bias, activation, **kwargs
            )

        _conv1d_lib.causal_conv1d_fn = causal_conv1d_fn_adapter
        _conv1d_lib.causal_conv1d_update = causal_conv1d_update_adapter
        _qwen3_next_lib.causal_conv1d_fn = causal_conv1d_fn_adapter
        _qwen3_next_lib.causal_conv1d_update = causal_conv1d_update_adapter

        # 【修复】关键！patch gdn_linear_attn 模块
        import vllm.model_executor.layers.mamba.gdn_linear_attn as _gdn_lib
        if hasattr(_gdn_lib, "causal_conv1d_fn"):
            _gdn_lib.causal_conv1d_fn = causal_conv1d_fn_adapter
        if hasattr(_gdn_lib, "causal_conv1d_update"):
            _gdn_lib.causal_conv1d_update = causal_conv1d_update_adapter

        logger.info("Patched causal_conv1d ops for Kunlunxin (including gdn_linear_attn)")
    except Exception as e:
        logger.warning("Failed to patch causal_conv1d ops: %s", e)


# ── FLA ops (chunk / fused_recurrent) ──
def patch_fla_ops():
    """Replace chunk_gated_delta_rule and fused_recurrent_gated_delta_rule
    with Kunlunxin top-level implementations.
    """
    try:
        import vllm.model_executor.layers.fla.ops as _fla_ops_lib
        import vllm.model_executor.layers.fla.ops.chunk as _fla_chunk_lib
        import vllm.model_executor.layers.fla.ops.fused_recurrent as _fla_recurrent_lib
        import vllm.model_executor.models.qwen3_next as _qwen3_next_lib

        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fla.chunk import (
            chunk_gated_delta_rule as klx_chunk_gated_delta_rule,
        )
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule as klx_fused_recurrent,
        )

        # Patch top-level chunk_gated_delta_rule
        _fla_ops_lib.chunk_gated_delta_rule = klx_chunk_gated_delta_rule
        _fla_chunk_lib.chunk_gated_delta_rule = klx_chunk_gated_delta_rule
        _qwen3_next_lib.chunk_gated_delta_rule = klx_chunk_gated_delta_rule

        # Patch top-level fused_recurrent_gated_delta_rule
        _fla_ops_lib.fused_recurrent_gated_delta_rule = klx_fused_recurrent
        _fla_recurrent_lib.fused_recurrent_gated_delta_rule = klx_fused_recurrent
        _qwen3_next_lib.fused_recurrent_gated_delta_rule = klx_fused_recurrent

        # Patch on gdn_linear_attn if imported there
        import vllm.model_executor.layers.mamba.gdn_linear_attn as _gdn_lib
        if hasattr(_gdn_lib, "fla_chunk_gated_delta_rule"):
            _gdn_lib.fla_chunk_gated_delta_rule = klx_chunk_gated_delta_rule
        if hasattr(_gdn_lib, "fused_recurrent_gated_delta_rule"):
            _gdn_lib.fused_recurrent_gated_delta_rule = klx_fused_recurrent

        logger.info("Patched FLA ops for Kunlunxin")
    except Exception as e:
        logger.warning("Failed to patch FLA ops: %s", e)


# ── fused_gdn_gating ──
def patch_fused_gdn_gating():
    """Replace the triton fused_gdn_gating kernel with Kunlunxin impl."""
    try:
        import vllm.model_executor.models.qwen3_next as _qwen3_next_lib
        import vllm.model_executor.layers.mamba.gdn_linear_attn as _gdn_lib

        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fused_gdn_gating import (
            fused_gdn_gating_kunlunxin,
        )

        _qwen3_next_lib.fused_gdn_gating = fused_gdn_gating_kunlunxin
        _gdn_lib.fused_gdn_gating = fused_gdn_gating_kunlunxin
        logger.info("Patched fused_gdn_gating for Kunlunxin")
    except Exception as e:
        logger.warning("Failed to patch fused_gdn_gating: %s", e)


# ── SSM cache update (via _forward_core override) ──
def patch_ssm_cache_update():
    """Replace GatedDeltaNetAttention._forward_core with Kunlunxin version.
    See patches/patch_forward_core.py for the implementation and diff markers.
    """
    try:
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_forward_core import apply_ssm_patch
        apply_ssm_patch()
        logger.info("Patched GatedDeltaNetAttention._forward_core for Kunlunxin")
    except Exception as e:
        logger.warning("Failed to patch _forward_core: %s", e)


# ── decode_paged_attention NaN workaround ──
def patch_decode_attention():
    """Replace decode_paged_attention with prefill_attention (prefix_cache mode).

    xtorch_ops.decode_paged_attention produces NaN on certain layers during
    decode (observed on layer 43+ of Qwen3.6-27B). Using prefill_attention
    with is_prefix_cache=True provides correct results.
    """
    try:
        import vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention as attn_mod
        import xtorch_ops

        @staticmethod
        def patched_forward_decode(
            query, key_cache, value_cache, block_tables,
            seq_lens, seq_lens_host, max_seq_len, num_decode_tokens,
            kv_cache_dtype, num_kv_heads, scale, alibi_slopes,
            k_scale, v_scale, max_window_size=-1, output=None
        ):
            """Use prefill_attention in prefix_cache mode for decode."""
            import torch
            if output is None:
                output = torch.empty_like(query)

            decode_query = query[:num_decode_tokens]
            decode_output = output[:num_decode_tokens]

            # Build query_start_loc: each decode token has query_len=1
            query_start_loc_host = torch.arange(
                num_decode_tokens + 1, dtype=torch.int32, device='cpu'
            )
            query_start_loc = query_start_loc_host.to(decode_query.device)

            # Build kv_prefix_start_loc from seq_lens
            sl = seq_lens_host[:num_decode_tokens].to(torch.int32)
            kv_prefix_start_loc_host = torch.zeros(
                num_decode_tokens + 1, dtype=torch.int32, device='cpu'
            )
            kv_prefix_start_loc_host[1:] = torch.cumsum(sl, dim=0)
            kv_prefix_start_loc = kv_prefix_start_loc_host.to(decode_query.device)

            window_left = -1
            window_right = -1
            if max_window_size > 0:
                window_left = max_window_size
                window_right = 0

            xtorch_ops.prefill_attention(
                decode_query,
                key_cache,
                value_cache,
                decode_output,
                is_causal=True,
                is_prefix_cache=True,
                alpha=scale,
                context_qlen_lod_cpu=query_start_loc_host,
                context_qlen_lod_xpu=query_start_loc,
                context_kvlen_lod_cpu=kv_prefix_start_loc_host,
                context_kvlen_lod_xpu=kv_prefix_start_loc,
                block_table=block_tables,
                alibi_slopes=alibi_slopes,
                swa_left=window_left,
                swa_right=window_right,
            )
            return output

        attn_mod.KunlunxinPagedAttention.forward_decode = patched_forward_decode
        logger.info(
            "Patched KunlunxinPagedAttention.forward_decode: "
            "using prefill_attention (prefix_cache) to fix decode NaN"
        )
    except Exception as e:
        logger.warning("Failed to patch decode attention: %s", e)
