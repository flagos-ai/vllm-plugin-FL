# Copyright (c) 2026 BAAI. All rights reserved.

import logging

import torch
import torch.nn.functional as F
import vllm

logger = logging.getLogger(__name__)
_patches_applied = False

def apply_ascend_patches():
    """Apply all Ascend-specific patches."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True
    # Patch modules for Ascend platform
    patch_causal_conv1d()
    patch_fla_ops()
    patch_op_cls()
    patch_fused_moe()
    patch_gdn_warmup()
    patch_gdn_triton_ops()
    patch_gdn_state_gather()
    patch_vit_pos_embed()

def patch_mamba_config():
    """Patch HybridAttentionMambaModelConfig for Ascend."""
    from .patches.patch_mamba_config import verify_and_update_config

    vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = verify_and_update_config
    logger.info("Patched HybridAttentionMambaModelConfig for Ascend")

def patch_causal_conv1d():
    """Patch causal_conv1d ops with Ascend implementations.

    Uses the pure-PyTorch causal_conv1d_fn (which internally calls
    causal_conv1d_ref) and a pure-PyTorch causal_conv1d_update to
    avoid the Triton kernel that crashes the NPU driver.
    """
    try:
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as _conv1d_lib
        import vllm.model_executor.models.qwen3_next as _qwen3_next_lib

        from .impl.causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_npu
        from .impl.causal_conv1d import causal_conv1d_ref

        def causal_conv1d_update_torch(
            x, conv_state, weight, bias=None, activation=None,
            conv_state_indices=None, num_accepted_tokens=None,
            query_start_loc=None, max_query_len=-1,
            pad_slot_id=-1, block_idx_last_scheduled_token=None,
            initial_state_idx=None, validate_data=False,
        ):
            """Pure-PyTorch causal_conv1d_update for Ascend NPU.

            Uses causal_conv1d_ref for each sequence. Moves small index
            tensors to CPU once to avoid per-iteration NPU sync.
            """
            if isinstance(activation, bool):
                activation = "silu" if activation else None
            original_dtype = x.dtype
            x = x.to(conv_state.dtype)
            unsqueeze = query_start_loc is None and x.dim() == 2
            if unsqueeze:
                x = x.unsqueeze(-1)

            _, width = weight.shape

            # Move small index tensors to CPU once
            if conv_state_indices is not None:
                csi_cpu = conv_state_indices.cpu().tolist()
            else:
                csi_cpu = None

            if query_start_loc is None:
                batch = x.shape[0]
                for i in range(batch):
                    idx = csi_cpu[i] if csi_cpu is not None else i
                    if pad_slot_id is not None and csi_cpu is not None and idx == pad_slot_id:
                        continue
                    state = conv_state[idx]  # (dim, state_len)
                    xi = x[i]  # (dim, seqlen)
                    seqlen = xi.shape[-1]
                    combined = torch.cat([state[..., -(width-1):], xi], dim=-1)
                    dim_size = weight.shape[0]
                    out_i = F.conv1d(
                        combined.unsqueeze(0),
                        weight.unsqueeze(1), bias,
                        padding=0, groups=dim_size,
                    ).squeeze(0)[..., :seqlen]
                    conv_state[idx, :, -(width-1):] = combined[..., -(width-1):]
                    if activation in ["silu", "swish"]:
                        out_i = F.silu(out_i)
                    x[i] = out_i
            else:
                qsl_cpu = query_start_loc.cpu().tolist()
                batch = len(qsl_cpu) - 1
                for i in range(batch):
                    idx = csi_cpu[i] if csi_cpu is not None else i
                    if pad_slot_id is not None and csi_cpu is not None and idx == pad_slot_id:
                        continue
                    start, end = qsl_cpu[i], qsl_cpu[i+1]
                    if start >= end:
                        continue
                    xi = x[start:end].t().unsqueeze(0)  # (1, dim, seqlen)
                    state = conv_state[idx]
                    dim_size = weight.shape[0]
                    combined = torch.cat([state[..., -(width-1):], xi.squeeze(0)], dim=-1)
                    out_i = F.conv1d(
                        combined.unsqueeze(0),
                        weight.unsqueeze(1), bias,
                        padding=0, groups=dim_size,
                    ).squeeze(0)[..., :(end-start)]
                    conv_state[idx, :, -(width-1):] = combined[..., -(width-1):]
                    if activation in ["silu", "swish"]:
                        out_i = F.silu(out_i)
                    x[start:end] = out_i.t()

            if unsqueeze:
                x = x.squeeze(-1)
            return x.to(original_dtype)

        _conv1d_lib.causal_conv1d_fn = causal_conv1d_fn_npu
        _conv1d_lib.causal_conv1d_update = causal_conv1d_update_torch
        _qwen3_next_lib.causal_conv1d_fn = causal_conv1d_fn_npu
        _qwen3_next_lib.causal_conv1d_update = causal_conv1d_update_torch

        # Also patch the gdn_linear_attn module's local bindings
        import vllm.model_executor.layers.mamba.gdn_linear_attn as _gdn_lib
        _gdn_lib.causal_conv1d_fn = causal_conv1d_fn_npu
        _gdn_lib.causal_conv1d_update = causal_conv1d_update_torch

        logger.info("Patched causal_conv1d ops for Ascend (pure-PyTorch)")
    except Exception as e:
        logger.warning("Failed to patch causal_conv1d ops: %s", e)

def patch_gdn_warmup():
    """Disable GDN prefill kernel warmup on Ascend NPU.

    The Triton FLA chunk_gated_delta_rule kernel uses tl.insert_slice which
    is not available on NPU. The warmup failure corrupts the NPU stream,
    causing all subsequent kernel calls to fail with 'Inner error'.
    Skipping the warmup avoids the stream corruption entirely.
    """
    try:
        import vllm.model_executor.layers.mamba.gdn_linear_attn as gdn_lib

        def _noop_warmup(self, mixed_qkv):
            pass

        gdn_lib.GatedDeltaNetAttention._warmup_prefill_kernels = _noop_warmup
        logger.info("Disabled GDN prefill kernel warmup for Ascend NPU")
    except Exception as e:
        logger.warning("Failed to patch GDN warmup: %s", e)

def patch_fused_moe():
    """Patch fused MoE ops with Ascend implementations."""
    # TODO ops' triton implementation is not ready yet
    from .impl.fused_moe import fused_experts_impl
    try:
        import vllm_fl.ops.fused_moe.fused_moe as fused_moe_lib

        fused_moe_lib.fused_experts_impl = fused_experts_impl

        logger.info("Patched fused_moe for Ascend")
    except Exception as e:
        logger.warning("Failed to patch fused_moe ops: %s", e)

def patch_fla_ops():
    """Patch FLA ops for Ascend NPU.

    The FlagGems FLA Triton kernels (chunk_gated_delta_rule_fwd, solve_tril,
    etc.) use tl.insert_slice and other ops not available in triton 3.2.0.
    Replace chunk_gated_delta_rule_fwd with a no-op that returns zeros,
    allowing the model to load and serve. The linear_attention layers
    will produce zero output (degraded quality) but the model will be
    functional for testing.
    """
    try:
        import vllm.model_executor.layers.fla.ops as _fla_ops_lib
        import vllm.model_executor.layers.fla.ops.chunk as _fla_chunk_lib
        import vllm.model_executor.layers.fla.ops.fused_recurrent as _fla_recurrent_lib
        import vllm.model_executor.layers.fla.ops.layernorm_guard as _fla_layernorm_lib
        import vllm.model_executor.models.qwen3_next as _qwen3_next_lib
        from flag_gems.runtime.backend._ascend.fla import (
            fused_recurrent_gated_delta_rule_fwd,
        )
        from flag_gems.runtime.backend._ascend.fla.layernorm_guard import (
            LayerNormFn as ascend_LayerNormFn,
        )

        from .impl.fla.gdn_torch_ops import chunk_gated_delta_rule_torch

        def chunk_gated_delta_rule_fwd_proper(
            q, k, v, g, beta, scale, initial_state,
            output_final_state, cu_seqlens=None, **kwargs
        ):
            if scale is None:
                scale = q.shape[-1] ** -0.5
            o, final_state = chunk_gated_delta_rule_torch(
                q=q, k=k, v=v, g=g, beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )
            return g, o, None, final_state, None, None, v

        def chunk_gated_delta_rule_proper(
            q, k, v, g, beta, scale=None, initial_state=None,
            output_final_state=False, cu_seqlens=None,
            head_first=False, use_qk_l2norm_in_kernel=False,
        ):
            if use_qk_l2norm_in_kernel:
                q = l2norm_fwd_torch(q)
                k = l2norm_fwd_torch(k)
            if scale is None:
                scale = q.shape[-1] ** -0.5
            o, fs = chunk_gated_delta_rule_torch(
                q=q, k=k, v=v, g=g, beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )
            return o, fs

        _fla_ops_lib.chunk_gated_delta_rule_fwd = chunk_gated_delta_rule_fwd_proper
        _fla_chunk_lib.chunk_gated_delta_rule_fwd = chunk_gated_delta_rule_fwd_proper
        _fla_chunk_lib.chunk_gated_delta_rule = chunk_gated_delta_rule_proper
        _fla_recurrent_lib.fused_recurrent_gated_delta_rule_fwd = fused_recurrent_gated_delta_rule_fwd
        _fla_layernorm_lib.LayerNormFn = ascend_LayerNormFn
        _qwen3_next_lib.chunk_gated_delta_rule = chunk_gated_delta_rule_proper

        # Also patch the FlagGems module and every import site
        import flag_gems.runtime.backend._ascend.fla as _fg_fla
        import flag_gems.runtime.backend._ascend.fla.chunk as _fg_chunk
        _fg_fla.chunk_gated_delta_rule_fwd = chunk_gated_delta_rule_fwd_proper
        _fg_chunk.chunk_gated_delta_rule_fwd = chunk_gated_delta_rule_fwd_proper

        # Patch the plugin's own FLA chunk module
        try:
            from vllm_fl.dispatch.backends.vendor.ascend.impl.fla import chunk as _ascend_chunk
            _ascend_chunk.chunk_gated_delta_rule_fwd = chunk_gated_delta_rule_fwd_proper
        except ImportError:
            pass

        # Patch the gdn_linear_attn module's local name binding
        import vllm.model_executor.layers.mamba.gdn_linear_attn as _gdn_lib
        _gdn_lib.fla_chunk_gated_delta_rule = chunk_gated_delta_rule_proper

        # Patch forward_native and forward_oot on ChunkGatedDeltaRule
        def _proper_forward_native(
            self, q, k, v, g, beta, initial_state, output_final_state,
            cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
            use_qk_l2norm_in_kernel=True,
        ):
            return chunk_gated_delta_rule_proper(
                q=q, k=k, v=v, g=g, beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
        _gdn_lib.ChunkGatedDeltaRule.forward_native = _proper_forward_native
        _gdn_lib.ChunkGatedDeltaRule.forward_oot = _proper_forward_native


        # No error-swallowing wrapper — errors propagate for proper debugging.

        logger.info("Patched FLA ops for Ascend NPU (proper recurrent chunk_gated_delta_rule)")
    except Exception as e:
        logger.warning("Failed to patch FLA ops: %s", e)

def patch_gdn_triton_ops():
    """Replace all GDN Triton kernels with pure-PyTorch implementations.

    The Triton LLVM backend crashes on ARM/Ascend NPU with an assertion
    failure in PointerUnion::get(). This replaces every Triton kernel
    used on the GDN inference path:
      - fused_sigmoid_gating_delta_rule_update (decode path gating+recurrence)
      - fused_post_conv_prep (prefill path post-conv preparation)
      - fused_recurrent_gated_delta_rule_packed_decode (packed decode)
      - fused_gdn_gating (gating computation in gdn_linear_attn.py)
      - l2norm_fwd (L2 normalization)
    """
    try:
        from .impl.fla.gdn_torch_ops import (
            chunk_gated_delta_rule_torch,
            fused_gdn_gating_torch,
            fused_post_conv_prep_torch,
            l2norm_fwd_torch,
        )

        # 1. Patch fused_sigmoid_gating_delta_rule_update (decode path)
        # Use the proper vectorized bmm implementation
        def _proper_sigmoid_gating_update(
            A_log, a, b, dt_bias, q, k, v,
            beta=1.0, threshold=20.0, scale=None,
            initial_state=None, inplace_final_state=True,
            cu_seqlens=None, ssm_state_indices=None,
            num_accepted_tokens=None, use_qk_l2norm_in_kernel=False,
            is_kda=False,
        ):
            """Pure-PyTorch fused sigmoid gating delta rule update.

            Combines gating computation + recurrent delta rule in one function.
            This is used for the DECODE path (T=1 typically).
            """
            B, T, H, K_dim = q.shape
            V_dim = v.shape[-1]
            HV = v.shape[2]
            groups = HV // H

            if scale is None:
                scale = K_dim ** -0.5

            # Compute gating: g = -exp(A_log) * softplus(a + dt_bias)
            # a, b: [num_tokens, HV]
            a_f = a.float()
            b_f = b.float()
            x = a_f + dt_bias.float().unsqueeze(0)
            sp = F.softplus(x, beta=beta, threshold=threshold)
            g_vals = -torch.exp(A_log.float()).unsqueeze(0) * sp
            beta_vals = torch.sigmoid(b_f)  # [tokens, HV]

            N = B if cu_seqlens is None else len(cu_seqlens) - 1
            o = torch.zeros_like(v)

            if inplace_final_state:
                final_state = initial_state
            else:
                final_state = initial_state.clone()

            if cu_seqlens is not None:
                cu_cpu = cu_seqlens.cpu().tolist()
            if ssm_state_indices is not None:
                ssi_cpu = ssm_state_indices.cpu()

            # Expand q/k to match HV heads
            if groups > 1:
                q_exp = q.repeat_interleave(groups, dim=2)
                k_exp = k.repeat_interleave(groups, dim=2)
            else:
                q_exp = q
                k_exp = k

            for i_n in range(N):
                if cu_seqlens is not None:
                    bos = cu_cpu[i_n]
                    eos = cu_cpu[i_n + 1]
                else:
                    bos, eos = i_n * T, (i_n + 1) * T

                if bos >= eos:
                    continue

                # Get state index
                if ssm_state_indices is not None:
                    if num_accepted_tokens is not None:
                        i_t_init = num_accepted_tokens[i_n].item() - 1
                    else:
                        i_t_init = 0
                    if ssm_state_indices.ndim == 1:
                        state_idx = ssi_cpu[i_n].item()
                    else:
                        state_idx = ssi_cpu[i_n, i_t_init].item()
                    if state_idx < 0:
                        continue
                else:
                    state_idx = bos

                h = final_state[state_idx].float()  # [HV, V, K]

                for t_offset in range(eos - bos):
                    t = bos + t_offset
                    qt = q_exp[0, t].float() if cu_seqlens is not None else q_exp[i_n, t_offset].float()
                    kt = k_exp[0, t].float() if cu_seqlens is not None else k_exp[i_n, t_offset].float()
                    vt = v[0, t].float() if cu_seqlens is not None else v[i_n, t_offset].float()

                    t_flat = t if cu_seqlens is not None else i_n * T + t_offset
                    gt = g_vals[t_flat].float()
                    bt = beta_vals[t_flat].float()

                    if use_qk_l2norm_in_kernel:
                        qt = qt * torch.rsqrt(torch.sum(qt * qt, dim=-1, keepdim=True) + 1e-6)
                        kt = kt * torch.rsqrt(torch.sum(kt * kt, dim=-1, keepdim=True) + 1e-6)
                    qt = qt * scale

                    # h *= exp(g)
                    h = h * torch.exp(gt).unsqueeze(-1).unsqueeze(-1)
                    # hk = h @ k
                    hk = torch.bmm(h, kt.unsqueeze(-1)).squeeze(-1)
                    # v' = beta * (v - hk)
                    vp = (vt - hk) * bt.unsqueeze(-1)
                    # h += v' outer k
                    h = h + torch.bmm(vp.unsqueeze(-1), kt.unsqueeze(-2))
                    # o = h @ q
                    ot = torch.bmm(h, qt.unsqueeze(-1)).squeeze(-1)

                    if cu_seqlens is not None:
                        o[0, t] = ot.to(o.dtype)
                    else:
                        o[i_n, t_offset] = ot.to(o.dtype)

                    # Store final state
                    if inplace_final_state and ssm_state_indices is not None:
                        if ssm_state_indices.ndim == 1:
                            fidx = ssi_cpu[i_n].item()
                        else:
                            fidx = ssi_cpu[i_n, t_offset].item()
                        if fidx >= 0:
                            final_state[fidx] = h.to(final_state.dtype)
                    else:
                        final_state[state_idx] = h.to(final_state.dtype)

            return o, final_state

        import vllm.model_executor.layers.fla.ops.fused_sigmoid_gating as _fsg_lib
        _fsg_lib.fused_sigmoid_gating_delta_rule_update = _proper_sigmoid_gating_update

        import vllm.model_executor.layers.mamba.gdn_linear_attn as _gdn_lib
        _gdn_lib.fused_sigmoid_gating_delta_rule_update = _proper_sigmoid_gating_update

        # 2. Patch fused_post_conv_prep
        import vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv as _fpc_lib
        _fpc_lib.fused_post_conv_prep = fused_post_conv_prep_torch

        import vllm.model_executor.layers.fla.ops as _fla_ops_lib
        _fla_ops_lib.fused_post_conv_prep = fused_post_conv_prep_torch
        _gdn_lib.fused_post_conv_prep = fused_post_conv_prep_torch

        # 3. Patch fused_recurrent_gated_delta_rule_packed_decode (decode path)
        # Use proper vectorized bmm implementation
        def _proper_packed_decode(
            mixed_qkv, a, b, A_log, dt_bias, scale,
            initial_state, out, ssm_state_indices,
            use_qk_l2norm_in_kernel=False,
        ):
            """Pure-PyTorch packed decode for GDN recurrence.

            Implements the gated delta rule recurrent update:
              g = -exp(A_log) * softplus(a + dt_bias)
              beta = sigmoid(b)
              h = h * exp(g)
              h = h + beta * (v - h @ k) ⊗ k
              o = h @ q
            """
            # Sync on first call to ensure prefill NPU ops have completed
            if not hasattr(_proper_packed_decode, '_synced'):
                _proper_packed_decode._synced = True
                import torch_npu
                torch.npu.synchronize()
            B_batch = mixed_qkv.shape[0]
            HV, V_dim, K_dim = initial_state.shape[-3:]
            qkv_dim = mixed_qkv.shape[1]
            qk_dim = qkv_dim - HV * V_dim
            H = qk_dim // (2 * K_dim)
            groups = HV // H
            ssi_cpu = ssm_state_indices.cpu().tolist()

            # Compute gating for all tokens at once
            x = a.float() + dt_bias.float().unsqueeze(0)
            sp = F.softplus(x, beta=1.0, threshold=20.0)
            g_vals = (-torch.exp(A_log.float()).unsqueeze(0) * sp)  # [B, HV]
            beta_vals = torch.sigmoid(b.float())  # [B, HV]

            for i_n in range(B_batch):
                state_idx = ssi_cpu[i_n]
                if state_idx < 0:
                    out[i_n, 0, :, :] = 0
                    continue

                # Extract q, k, v from packed mixed_qkv
                q_start, k_start, v_start = 0, H * K_dim, 2 * H * K_dim
                b_q = mixed_qkv[i_n, q_start:q_start + H * K_dim].view(H, K_dim).float()
                b_k = mixed_qkv[i_n, k_start:k_start + H * K_dim].view(H, K_dim).float()
                b_v = mixed_qkv[i_n, v_start:v_start + HV * V_dim].view(HV, V_dim).float()

                if use_qk_l2norm_in_kernel:
                    b_q = b_q * torch.rsqrt(torch.sum(b_q * b_q, dim=-1, keepdim=True) + 1e-6)
                    b_k = b_k * torch.rsqrt(torch.sum(b_k * b_k, dim=-1, keepdim=True) + 1e-6)
                b_q = b_q * scale

                # Expand q/k heads to match value heads (multi-query attention)
                if groups > 1:
                    b_q = b_q.repeat_interleave(groups, dim=0)
                    b_k = b_k.repeat_interleave(groups, dim=0)

                h = initial_state[state_idx].float()  # [HV, V, K]
                # Safety net: zero out if state contains garbage
                # (NaN/Inf from uninitialized memory)
                if torch.isnan(h).any() or torch.isinf(h).any():
                    h = torch.zeros_like(h)

                gt = g_vals[i_n]  # [HV]
                bt = beta_vals[i_n]  # [HV]

                # Gated decay
                h = h * torch.exp(gt).unsqueeze(-1).unsqueeze(-1)
                # Delta rule update: h += beta * (v - h@k) ⊗ k
                hk = torch.bmm(h, b_k.unsqueeze(-1)).squeeze(-1)  # [HV, V]
                vp = (b_v - hk) * bt.unsqueeze(-1)
                h = h + torch.bmm(vp.unsqueeze(-1), b_k.unsqueeze(-2))
                # Output: o = h @ q
                b_o = torch.bmm(h, b_q.unsqueeze(-1)).squeeze(-1)  # [HV, V]

                out[i_n, 0] = b_o.to(out.dtype)
                initial_state[state_idx] = h.to(initial_state.dtype)

            return out, initial_state

        import vllm.model_executor.layers.fla.ops.fused_recurrent as _fr_lib
        _fr_lib.fused_recurrent_gated_delta_rule_packed_decode = _proper_packed_decode
        _fla_ops_lib.fused_recurrent_gated_delta_rule_packed_decode = _proper_packed_decode
        _gdn_lib.fused_recurrent_gated_delta_rule_packed_decode = _proper_packed_decode

        # 4. Patch fused_gdn_gating (in gdn_linear_attn.py)
        _gdn_lib.fused_gdn_gating = fused_gdn_gating_torch

        # 5. Patch l2norm_fwd
        import vllm.model_executor.layers.fla.ops.chunk as _chunk_lib
        import vllm.model_executor.layers.fla.ops.l2norm as _l2norm_lib
        _l2norm_lib.l2norm_fwd = l2norm_fwd_torch
        _chunk_lib.l2norm_fwd = l2norm_fwd_torch
        _gdn_lib.l2norm_fwd = l2norm_fwd_torch

        # Also patch the plugin's own l2norm if present
        try:
            from vllm_fl.dispatch.backends.vendor.ascend.impl.fla import (
                l2norm as _ascend_l2norm,
            )
            _ascend_l2norm.l2norm_fwd = l2norm_fwd_torch
        except ImportError:
            pass

        logger.info(
            "Patched all GDN Triton kernels with pure-PyTorch for Ascend NPU"
        )
    except Exception as e:
        logger.warning("Failed to patch GDN Triton ops: %s", e)


def patch_op_cls():
    """Patch MMEncoderAttention to use manual matmul attention on NPU.

    The NPU npu_fused_infer_attention_score kernel only supports head_dim
    in {64, 128, 192}. The vision encoder may have non-standard head_dim
    (e.g. 72 for Qwen3.5). F.scaled_dot_product_attention on NPU may also
    dispatch to the same problematic kernel. Use pure-PyTorch matmul
    attention instead.
    """
    try:
        from vllm.model_executor.custom_op import CustomOp

        from .impl.mm_encoder_attention import AscendMMEncoderAttention
        from .impl.vocab_parallel_embedding import AscendVocabParallelEmbedding
        REGISTERED_ASCEND_OPS = {
            "VocabParallelEmbedding": AscendVocabParallelEmbedding,
            "MMEncoderAttention": AscendMMEncoderAttention,
        }
        for name, op_cls in REGISTERED_ASCEND_OPS.items():
            CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)
        logger.info("Patched MMEncoderAttention for NPU (matmul attention)")
    except Exception as e:
        logger.warning("Failed to patch MMEncoderAttention: %s", e)

def refresh_block_size(vllm_config, block_size = 128):
    """
    Refresh the block size in cache config.
    """
    cache_config = vllm_config.cache_config
    scheduler_config = vllm_config.scheduler_config
    model_config = vllm_config.model_config

    if not cache_config:
        return

    if cache_config.block_size is None:
        cache_config.block_size = block_size

    if not scheduler_config or not model_config:
        return

    # TODO(MengqingCao): Remove the model_type check, after resolving the hidden error in get_kv_cache_groups.
    if model_config.hf_text_config.model_type != "qwen3_next" and cache_config.block_size != block_size:
        if cache_config.enable_prefix_caching or scheduler_config.enable_chunked_prefill:
            logger.info(f"Block size is set to {block_size} if prefix cache or chunked prefill is enabled.")
            cache_config.block_size = block_size


def patch_gdn_state_gather():
    """Fix GDN ssm_state gather for Ascend NPU.

    torch_npu's advanced-index gather ``ssm_state[index_tensor]`` returns
    wrong/stale values on NPU (for both int32 and int64 index tensors), while
    ``torch.index_select`` reads correctly. In GatedDeltaNetAttention._forward_core
    the broken gather corrupts the initial state of decode sequences batched with
    a prefill (mixed prefill+decode step), blowing up the recurrent state and
    producing garbage tokens under concurrency.

    We rebind _forward_core to a copy of the installed method source with that one
    gather rewritten to torch.index_select, so the patch tracks the vLLM version.
    """
    try:
        import inspect
        import textwrap
        import vllm.model_executor.layers.mamba.gdn_linear_attn as _gdn

        cls = _gdn.GatedDeltaNetAttention
        src = inspect.getsource(cls._forward_core)
        broken = "ssm_state[non_spec_state_indices_tensor].contiguous()"
        fixed = ("torch.index_select(ssm_state, 0, "
                 "non_spec_state_indices_tensor.long()).contiguous()")
        if broken not in src:
            logger.warning(
                "GDN _forward_core gather pattern not found; skipping patch "
                "(vLLM may already be fixed or changed).")
            return
        src = textwrap.dedent(src).replace(broken, fixed)
        # Exec in the module's namespace so all globals resolve correctly.
        ns = {}
        exec(src, _gdn.__dict__, ns)
        cls._forward_core = ns["_forward_core"]
        logger.info("Patched GDN _forward_core ssm_state gather for Ascend "
                    "(index_select).")
    except Exception as e:
        logger.warning("Failed to patch GDN state gather: %s", e)


def patch_vit_pos_embed():
    """Force native (pure-torch) ViT position-embedding interpolation on NPU.

    Qwen3-VL's fast_pos_embed_interpolate picks a Triton kernel when HAS_TRITON
    is True, but Triton is unreliable on Ascend NPU and produces wrong learned
    2D position embeddings, so the vision encoder loses spatial layout (image
    understanding degrades badly). The function reads the module-level HAS_TRITON
    at call time, so flipping it to False routes to pos_embed_interpolate_native.
    """
    try:
        import vllm.model_executor.models.qwen3_vl as _vl
        if getattr(_vl, "HAS_TRITON", False):
            _vl.HAS_TRITON = False
            logger.info("Forced native ViT pos-embed interpolation for Ascend.")
    except Exception as e:
        logger.warning("Failed to patch ViT pos-embed interpolation: %s", e)



