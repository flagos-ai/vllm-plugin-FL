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

import torch

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
    patch_cudagraph_dispatcher()
    patch_topk_topp_sampler()
    patch_fused_moe()

    # Existing patches
    patch_causal_conv1d()
    patch_fla_ops()
    patch_fused_gdn_gating()
    patch_ssm_cache_update()
    patch_sampler_rng()
    patch_decode_attention()
    logger.info("Applied all Kunlunxin patches")


def patch_cudagraph_dispatcher():
    """Use FULL graphs only for exact-size uniform decode batches.

    vLLM normally rounds an uncaptured token count up to the next CUDA Graph
    capture size.  That is unsafe for Kunlunxin recurrent models because the
    synthetic padding rows can participate in GDN/SSM state updates.  Keep the
    profitable FULL graph path for exact-size decode batches, but exclude FULL
    for mixed batches and non-exact decode sizes.  The native dispatcher then
    falls back to PIECEWISE (or NONE when PIECEWISE is not configured).
    """
    try:
        from functools import wraps

        from vllm.config import CUDAGraphMode
        from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

        original_dispatch = CudagraphDispatcher.dispatch
        if getattr(original_dispatch, "_kunlunxin_exact_full", False):
            return

        @wraps(original_dispatch)
        def dispatch_exact_full(
            self,
            num_tokens,
            uniform_decode=False,
            has_lora=False,
            num_active_loras=0,
            valid_modes=None,
            invalid_modes=None,
        ):
            padding_map = getattr(self, "_bs_to_padded_graph_size", None)
            full_is_exact = (
                uniform_decode
                and padding_map is not None
                and 0 <= num_tokens < len(padding_map)
                and padding_map[num_tokens] == num_tokens
            )

            if not full_is_exact:
                invalid_modes = set(invalid_modes or ())
                invalid_modes.add(CUDAGraphMode.FULL)

                # A DP re-dispatch can constrain valid_modes to the mode chosen
                # across ranks.  If that mode is FULL but this local batch is
                # non-exact, choose the safe no-graph fallback rather than
                # leaving the native dispatcher with an empty allowed set.
                if valid_modes is not None and set(valid_modes) <= {
                    CUDAGraphMode.FULL
                }:
                    valid_modes = {CUDAGraphMode.NONE}
                    invalid_modes.discard(CUDAGraphMode.NONE)

            return original_dispatch(
                self,
                num_tokens=num_tokens,
                uniform_decode=uniform_decode,
                has_lora=has_lora,
                num_active_loras=num_active_loras,
                valid_modes=valid_modes,
                invalid_modes=invalid_modes,
            )

        dispatch_exact_full._kunlunxin_exact_full = True
        CudagraphDispatcher.dispatch = dispatch_exact_full
        logger.info(
            "Patched CudagraphDispatcher: FULL only for exact-size uniform "
            "decode batches; non-exact batches fall back safely"
        )
    except Exception as e:
        logger.warning("Failed to patch CudagraphDispatcher: %s", e)


# ── RESTORED: block_table slot_mapping (Triton kernel bypass) ──
def patch_block_table_slot_mapping():
    """Replace Triton-based compute_slot_mapping with CPU numpy implementation.

    vLLM 0.20 moved compute_slot_mapping to a Triton kernel which fails on
    Kunlunxin XPU (err_code -714). Replace with a numpy-based approach.
    """
    try:
        import torch
        from vllm.v1.worker.block_table import BlockTable

        PAD_SLOT_ID = -1

        def compute_slot_mapping_xpu(self, num_reqs, query_start_loc, positions):
            device = positions.device
            num_tokens = positions.shape[0]

            total_cp_world_size = self.pcp_world_size * self.dcp_world_size
            total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank

            # Build req_indices: repeat_interleave on XPU, no CPU copy
            counts = query_start_loc[1:num_reqs + 1] - query_start_loc[:num_reqs]
            req_indices = torch.repeat_interleave(
                torch.arange(num_reqs, device=device), counts
            )

            # Direct GPU indexing: use .gpu tensors, skip .np and copy_to_gpu
            if total_cp_world_size > 1:
                virtual_block_size = self.block_size * total_cp_world_size
                bt_indices = (
                    req_indices * self.max_num_blocks_per_req
                    + positions // virtual_block_size
                )
                block_numbers = self.block_table.gpu.view(-1)[bt_indices]
                virtual_block_offsets = positions % virtual_block_size
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
                slot_vals = block_numbers * self.block_size + block_offsets
                self.slot_mapping.gpu[:num_tokens] = torch.where(
                    mask, slot_vals,
                    torch.full((), PAD_SLOT_ID, dtype=slot_vals.dtype, device=device)
                )
            else:
                bt_indices = (
                    req_indices * self.max_num_blocks_per_req
                    + positions // self.block_size
                )
                block_numbers = self.block_table.gpu.view(-1)[bt_indices]
                block_offsets = positions % self.block_size
                self.slot_mapping.gpu[:num_tokens] = (
                    block_numbers * self.block_size + block_offsets
                )

            # Pad remaining slots
            self.slot_mapping.gpu[num_tokens:self.max_num_batched_tokens] = PAD_SLOT_ID

        BlockTable.compute_slot_mapping = compute_slot_mapping_xpu
        logger.info("Patched BlockTable.compute_slot_mapping to XPU torch path for Kunlunxin")
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

        def _broadcast_random_sample(
            probs, generators, use_fp64_gumbel=False
        ):
            sampled_tokens = _orig_random_sample(
                probs, generators, use_fp64_gumbel
            )
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
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as _gdn_lib

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
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as _gdn_lib

        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fla.chunk import (
            chunk_gated_delta_rule as klx_chunk_gated_delta_rule,
        )
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule as klx_fused_recurrent,
        )

        # Patch top-level chunk_gated_delta_rule
        _fla_ops_lib.chunk_gated_delta_rule = klx_chunk_gated_delta_rule
        _fla_chunk_lib.chunk_gated_delta_rule = klx_chunk_gated_delta_rule
        _gdn_lib.fla_chunk_gated_delta_rule = klx_chunk_gated_delta_rule

        # Patch top-level fused_recurrent_gated_delta_rule
        _fla_ops_lib.fused_recurrent_gated_delta_rule = klx_fused_recurrent
        _fla_recurrent_lib.fused_recurrent_gated_delta_rule = klx_fused_recurrent
        if hasattr(_gdn_lib, "fused_recurrent_gated_delta_rule"):
            _gdn_lib.fused_recurrent_gated_delta_rule = klx_fused_recurrent

        logger.info("Patched FLA ops for Kunlunxin")
    except Exception as e:
        logger.warning("Failed to patch FLA ops: %s", e)


# ── fused_gdn_gating ──
def patch_fused_gdn_gating():
    """Replace the triton fused_gdn_gating kernel with Kunlunxin impl."""
    try:
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as _gdn_lib

        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fused_gdn_gating import (
            fused_gdn_gating_kunlunxin,
        )
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule as klx_fused_recurrent,
        )

        def fused_post_conv_prep_kunlunxin(
            conv_output,
            a,
            b,
            A_log,
            dt_bias,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        ):
            """PyTorch post-conv path matching vLLM 0.24's return layout."""
            num_v_heads = A_log.shape[0]
            q_size = num_k_heads * head_k_dim
            k_size = num_k_heads * head_k_dim
            v_size = num_v_heads * head_v_dim
            q, k, v = torch.split(
                conv_output, (q_size, k_size, v_size), dim=-1
            )
            q = q.reshape(-1, num_k_heads, head_k_dim).contiguous()
            k = k.reshape(-1, num_k_heads, head_k_dim).contiguous()
            v = v.reshape(-1, num_v_heads, head_v_dim).contiguous()
            if apply_l2norm:
                q = torch.nn.functional.normalize(q, p=2, dim=-1)
                k = torch.nn.functional.normalize(k, p=2, dim=-1)

            x = (a + dt_bias).float()
            g = -torch.exp(A_log.float()) * torch.nn.functional.softplus(x)
            if output_g_exp:
                g = torch.exp(g)
            beta_output = torch.sigmoid(b.float())
            return q, k, v, g, beta_output

        def fused_sigmoid_gating_delta_rule_update_kunlunxin(
            A_log,
            a,
            b,
            dt_bias,
            q,
            k,
            v,
            beta=1.0,
            threshold=20.0,
            scale=None,
            initial_state=None,
            inplace_final_state=True,
            cu_seqlens=None,
            ssm_state_indices=None,
            num_accepted_tokens=None,
            use_qk_l2norm_in_kernel=False,
            is_kda=False,
        ):
            if is_kda:
                raise NotImplementedError("KDA is not supported on Kunlunxin")
            # The Kunlunxin 0.20 adaptation used PyTorch gating and L2
            # normalization for decode because the fused/kernel-internal paths
            # lose enough precision to corrupt recurrent state.
            x = (a + dt_bias).float()
            g = (
                -torch.exp(A_log.float())
                * torch.nn.functional.softplus(x, beta=beta, threshold=threshold)
            ).unsqueeze(0)
            beta_output = torch.sigmoid(b.float()).to(q.dtype).unsqueeze(0)
            if use_qk_l2norm_in_kernel:
                q = torch.nn.functional.normalize(q, p=2, dim=-1)
                k = torch.nn.functional.normalize(k, p=2, dim=-1)
                use_qk_l2norm_in_kernel = False

            # vLLM 0.24 keeps decode indices followed by prefill indices for a
            # mixed batch.  The Kunlunxin recurrent ABI requires exactly one
            # state index per variable-length decode sequence.
            if (
                cu_seqlens is not None
                and ssm_state_indices is not None
                and ssm_state_indices.ndim == 1
            ):
                ssm_state_indices = ssm_state_indices[
                    : cu_seqlens.shape[0] - 1
                ]
            return klx_fused_recurrent(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta_output,
                scale=scale,
                initial_state=initial_state,
                inplace_final_state=inplace_final_state,
                cu_seqlens=cu_seqlens,
                ssm_state_indices=ssm_state_indices,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        _gdn_lib.fused_gdn_gating = fused_gdn_gating_kunlunxin
        _gdn_lib.fused_post_conv_prep = fused_post_conv_prep_kunlunxin
        _gdn_lib.fused_sigmoid_gating_delta_rule_update = (
            fused_sigmoid_gating_delta_rule_update_kunlunxin
        )
        logger.info(
            "Patched GDN post-conv, gating, and recurrent update for Kunlunxin"
        )
    except Exception as e:
        logger.warning("Failed to patch fused_gdn_gating: %s", e)


# ── SSM cache update (via _forward_core override) ──
def patch_ssm_cache_update():
    """Replace GatedDeltaNetAttention._forward_core with Kunlunxin version.
    See patches/patch_forward_core.py for the implementation and diff markers.
    """
    try:
        import inspect
        import textwrap

        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn_mod
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_forward_core import (
            _kunlunxin_write_ssm_cache,
        )

        cls = gdn_mod.QwenGatedDeltaNetAttention
        source = textwrap.dedent(inspect.getsource(cls._forward_core))
        original_write = (
            "ssm_state[prefill_state_indices] = "
            "last_recurrent_state.to(ssm_state.dtype)"
        )
        replacement_write = (
            "_kunlunxin_write_ssm_cache(ssm_state, last_recurrent_state, "
            "prefill_state_indices)"
        )
        if source.count(original_write) != 1:
            raise RuntimeError(
                "Unexpected vLLM 0.24 GDN cache-write implementation"
            )
        source = source.replace(original_write, replacement_write)
        namespace = dict(vars(gdn_mod))
        namespace["_kunlunxin_write_ssm_cache"] = _kunlunxin_write_ssm_cache
        exec(source, namespace)
        cls._forward_core = namespace["_forward_core"]
        logger.info(
            "Patched vLLM 0.24 GDN core to use Kunlunxin SSM cache write"
        )
        return
    except ImportError:
        pass

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

        original_forward_decode = attn_mod.KunlunxinPagedAttention.forward_decode
        def use_native_decode_for_cudagraph() -> bool:
            try:
                from vllm.config import CUDAGraphMode
                from vllm.forward_context import (
                    get_forward_context,
                    is_forward_context_available,
                )
                if not is_forward_context_available():
                    return False
                return (
                    get_forward_context().cudagraph_runtime_mode
                    == CUDAGraphMode.FULL
                )
            except Exception:
                # Keep the patch usable across vLLM minor versions that do
                # not expose the runtime-mode field.
                return False

        @staticmethod
        def patched_forward_decode(
            query, key_cache, value_cache, block_tables,
            seq_lens, seq_lens_host, max_seq_len, num_decode_tokens,
            kv_cache_dtype, num_kv_heads, scale, alibi_slopes,
            k_scale, v_scale, max_window_size=-1, output=None
        ):
            """Use prefill_attention in prefix_cache mode for decode."""
            import torch

            if use_native_decode_for_cudagraph():
                return original_forward_decode(
                    query,
                    key_cache,
                    value_cache,
                    block_tables,
                    seq_lens,
                    seq_lens_host,
                    max_seq_len,
                    num_decode_tokens,
                    kv_cache_dtype,
                    num_kv_heads,
                    scale,
                    alibi_slopes,
                    k_scale,
                    v_scale,
                    max_window_size=max_window_size,
                    output=output,
                )

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
            alpha = scale * (float(decode_query.shape[2]) ** 0.5)
            xtorch_ops.prefill_attention(
                decode_query,
                key_cache,
                value_cache,
                decode_output,
                is_causal=True,
                is_prefix_cache=True,
                alpha=alpha,
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
