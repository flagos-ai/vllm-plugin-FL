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

    patch_causal_conv1d()
    patch_fla_ops()
    patch_fused_gdn_gating()
    patch_op_cls()
    patch_ssm_cache_update()
    patch_sampler_rng()
    logger.info("Applied all Kunlunxin patches")


# ── sampler RNG (TP-consistent unseeded sampling) ──
def patch_sampler_rng():
    try:
        import torch
        import vllm.v1.sample.ops.topk_topp_sampler as _sampler_mod

        def _tp_consistent_random_sample(probs, generators):
            q = torch.empty_like(probs)
            if len(generators) != probs.shape[0]:
                q.uniform_()
                q = -torch.log(1-q)
                q = q.clamp(min=1e-12)
            if generators:
                # TODO(woosuk): This can be slow because we handle each request
                # one by one. Optimize this.
                if os.getenv("FAST_RANDOM_SAMPLE") == "1":
                    for i, generator in generators.items():
                        q[i].uniform_(generator=generator)
                    q = -torch.log(1-q)
                    q = q.clamp(min=1e-12)
                else:
                    for i, generator in generators.items():
                        q[i].uniform_(generator=generator)
                        q[i] = -torch.log(1-q[i])
                        q[i] = q[i].clamp(min=1e-12)

            return probs.div_(q).argmax(dim=-1).view(-1)

        _sampler_mod.random_sample = _tp_consistent_random_sample
        logger.info(
            "Patched sampler random_sample for TP-consistent sampling on Kunlunxin"
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
        logger.info("Patched causal_conv1d ops for Kunlunxin")
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

        logger.info("Patched FLA ops for Kunlunxin")
    except Exception as e:
        logger.warning("Failed to patch FLA ops: %s", e)


# ── fused_gdn_gating ──
def patch_fused_gdn_gating():
    """Replace the triton fused_gdn_gating kernel with Kunlunxin impl."""
    try:
        import vllm.model_executor.models.qwen3_next as _qwen3_next_lib

        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fused_gdn_gating import (
            fused_gdn_gating_kunlunxin,
        )

        _qwen3_next_lib.fused_gdn_gating = fused_gdn_gating_kunlunxin
        logger.info("Patched fused_gdn_gating for Kunlunxin")
    except Exception as e:
        logger.warning("Failed to patch fused_gdn_gating: %s", e)


# ── SSM cache update (via _forward_core override) ──
def patch_ssm_cache_update():
    """Replace Qwen3NextGatedDeltaNet._forward_core with Kunlunxin version.
    See patches/patch_forward_core.py for the implementation and diff markers.
    """
    try:
        import vllm.model_executor.models.qwen3_next as _qwen3_next_lib

        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_Qwen3NextGatedDeltaNet_forward_core import _forward_core_kunlunxin

        _qwen3_next_lib.Qwen3NextGatedDeltaNet._forward_core = _forward_core_kunlunxin
        logger.info("Patched Qwen3NextGatedDeltaNet._forward_core for Kunlunxin")
    except Exception as e:
        logger.warning("Failed to patch _forward_core: %s", e)


# ── SharedFusedMoE (via CustomOp.register_oot) ──
def patch_op_cls():
    """Register KunlunxinSharedFusedMoE as OOT replacement for SharedFusedMoE.
    """
    try:
        from vllm.model_executor.custom_op import CustomOp

        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fused_moe import KunlunxinSharedFusedMoE

        REGISTERED_KUNLUNXIN_OPS = {
            "SharedFusedMoE": KunlunxinSharedFusedMoE,
        }
        for name, op_cls in REGISTERED_KUNLUNXIN_OPS.items():
            CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)
        logger.info("Registered Kunlunxin OOT ops: %s",
                     list(REGISTERED_KUNLUNXIN_OPS.keys()))
    except Exception as e:
        logger.warning("Failed to register Kunlunxin OOT ops: %s", e)
