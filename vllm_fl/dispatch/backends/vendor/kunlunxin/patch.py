# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

"""
Kunlunxin platform monkey-patches for vllm 0.20+.

Follows the same pattern as ascend/patch.py -- all Kunlunxin-specific
overrides are applied here.

Called once during plugin initialization (register_oot_ops).
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

    # Disable packed recurrent decode fast path (uses triton kernel
    # fused_recurrent_gated_delta_rule_packed_decode which is not available
    # on Kunlunxin). Setting env var before vllm.envs is evaluated.
    import os
    os.environ.setdefault('VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE', '0')

    patch_attention_backend_registry()
    patch_block_table_slot_mapping()
    patch_topk_topp_sampler()
    patch_fused_moe()
    patch_causal_conv1d()
    patch_fla_ops()
    patch_fused_gdn_gating()
    patch_ssm_cache_update()
    logger.info("Applied all Kunlunxin patches")


# -- block_table compute_slot_mapping --
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


# -- attention backend registry --
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


# -- topk_topp_sampler --
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


# -- fused_moe --
def patch_fused_moe():
    """Replace fused_experts_impl with Kunlunxin implementation."""
    try:
        from .impl.fused_moe.fused_moe import fused_experts_impl as klx_fused_experts_impl
        import vllm_fl.ops.fused_moe.fused_moe as fused_moe_lib

        fused_moe_lib.fused_experts_impl = klx_fused_experts_impl
        logger.info("Patched fused_moe for Kunlunxin")
    except Exception as e:
        logger.warning("Failed to patch fused_moe ops: %s", e)


# -- causal_conv1d --
def patch_causal_conv1d():
    """Replace causal_conv1d_fn / causal_conv1d_update with Kunlunxin impls.

    Upstream convention:
        causal_conv1d_fn:  x=(dim, cu_seqlen), conv_states=(..., dim, state_len) -- NCW
        causal_conv1d_update: conv_state=(..., dim, state_len) -- NCW
    Kunlunxin kernel convention:
        x=(cu_seqlen, dim), conv_states=(N, state_len, dim) -- NWC, is_ncw=False
    """
    try:
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as _conv1d_lib
        import vllm.model_executor.layers.mamba.gdn_linear_attn as _gdn_lib

        from .impl.causal_conv1d import (
            causal_conv1d_fn_kunlunxin,
            causal_conv1d_update_kunlunxin,
        )

        def causal_conv1d_fn_adapter(
            x, weight, bias, conv_states, query_start_loc, **kwargs
        ):
            # NCW -> NWC: conv_states view, x transpose
            conv_states_nwc = conv_states.transpose(-1, -2)
            x_nwc = x.transpose(0, 1).contiguous()
            out = causal_conv1d_fn_kunlunxin(
                x_nwc, weight, bias, conv_states_nwc, query_start_loc, **kwargs
            )
            # NWC -> NCW: transpose output back
            return out.transpose(0, 1)

        def causal_conv1d_update_adapter(
            x, conv_state, weight, bias=None, activation=None, **kwargs
        ):
            # NCW -> NWC: conv_state view
            conv_state_nwc = conv_state.transpose(-1, -2)
            return causal_conv1d_update_kunlunxin(
                x, conv_state_nwc, weight, bias, activation, **kwargs
            )

        _conv1d_lib.causal_conv1d_fn = causal_conv1d_fn_adapter
        _conv1d_lib.causal_conv1d_update = causal_conv1d_update_adapter
        # Also patch on gdn_linear_attn if it imports these at module level
        if hasattr(_gdn_lib, "causal_conv1d_fn"):
            _gdn_lib.causal_conv1d_fn = causal_conv1d_fn_adapter
        if hasattr(_gdn_lib, "causal_conv1d_update"):
            _gdn_lib.causal_conv1d_update = causal_conv1d_update_adapter
        logger.info("Patched causal_conv1d ops for Kunlunxin")
    except Exception as e:
        logger.warning("Failed to patch causal_conv1d ops: %s", e)


# -- FLA ops (chunk / fused_recurrent) --
def patch_fla_ops():
    """Replace chunk_gated_delta_rule and fused_recurrent_gated_delta_rule
    with Kunlunxin implementations.
    """
    try:
        import vllm.model_executor.layers.fla.ops as _fla_ops_lib
        import vllm.model_executor.layers.fla.ops.chunk as _fla_chunk_lib
        import vllm.model_executor.layers.fla.ops.fused_recurrent as _fla_recurrent_lib
        import vllm.model_executor.layers.mamba.gdn_linear_attn as _gdn_lib

        from .impl.fla.chunk import (
            chunk_gated_delta_rule as klx_chunk_gated_delta_rule,
        )
        from .impl.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule as klx_fused_recurrent,
        )

        # Patch in fla.ops modules
        _fla_ops_lib.chunk_gated_delta_rule = klx_chunk_gated_delta_rule
        _fla_chunk_lib.chunk_gated_delta_rule = klx_chunk_gated_delta_rule

        _fla_ops_lib.fused_recurrent_gated_delta_rule = klx_fused_recurrent
        _fla_recurrent_lib.fused_recurrent_gated_delta_rule = klx_fused_recurrent

        # Patch on gdn_linear_attn if imported there
        if hasattr(_gdn_lib, "fla_chunk_gated_delta_rule"):
            _gdn_lib.fla_chunk_gated_delta_rule = klx_chunk_gated_delta_rule

        logger.info("Patched FLA ops for Kunlunxin")
    except Exception as e:
        logger.warning("Failed to patch FLA ops: %s", e)



# -- fused_gdn_gating --
def patch_fused_gdn_gating():
    """Replace the triton fused_gdn_gating kernel with Kunlunxin impl."""
    try:
        import vllm.model_executor.layers.mamba.gdn_linear_attn as _gdn_lib

        from .impl.fused_gdn_gating import fused_gdn_gating_kunlunxin

        _gdn_lib.fused_gdn_gating = fused_gdn_gating_kunlunxin
        logger.info("Patched fused_gdn_gating for Kunlunxin")
    except Exception as e:
        logger.warning("Failed to patch fused_gdn_gating: %s", e)


# -- SSM cache update (via _forward_core override) --
def patch_ssm_cache_update():
    """Replace GatedDeltaNetAttention._forward_core ssm_state write.

    The Kunlunxin version replaces the direct ssm_state[indices] = ... write
    with KunlunxinPagedAttention.reshape_and_cache_flash for the Kunlunxin
    memory subsystem.
    """
    try:
        from .patches.patch_forward_core import apply_ssm_patch

        apply_ssm_patch()
    except Exception as e:
        logger.warning("Failed to patch _forward_core: %s", e)
