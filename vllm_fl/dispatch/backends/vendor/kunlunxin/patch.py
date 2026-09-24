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
import os

import torch

logger = logging.getLogger(__name__)
_patches_applied = False


def _is_full_graph_runtime() -> bool:
    """Return whether the current forward is using a FULL CUDA graph."""
    try:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        return (
            is_forward_context_available()
            and get_forward_context().cudagraph_runtime_mode
            == CUDAGraphMode.FULL
        )
    except Exception:
        return False


def apply_kunlunxin_patches():
    """Apply all Kunlunxin-specific patches. Idempotent."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    # Disable Triton kernels incompatible with Kunlunxin XPU
    os.environ.setdefault("VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE", "0")

    # RESTORED from old version: Critical Triton kernel compatibility patches
    patch_block_table_slot_mapping()
    patch_attention_backend_registry()
    patch_cudagraph_dispatcher()
    patch_breakable_cudagraph_mode()
    patch_breakable_private_pools()
    patch_breakable_full_only()
    patch_graph_all_reduce()
    patch_eager_all_gather()
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
    """Keep uniform decode on FULL graphs, including padded batches.

    Padding rows are isolated by the attention cache-write mask and vLLM's
    reserved NULL recurrent-state slot. Tensor-parallel collectives use the
    graph-aware ProcessGroup path, so padding is no longer a reason to drop
    decode to PIECEWISE. Mixed/non-uniform batches remain outside FULL because
    their control flow is not shape-stable.
    """
    try:
        from functools import wraps

        from vllm.config import CUDAGraphMode
        from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

        original_dispatch = CudagraphDispatcher.dispatch
        if getattr(original_dispatch, "_kunlunxin_uniform_full", False):
            return

        @wraps(original_dispatch)
        def dispatch_uniform_full(
            self,
            num_tokens,
            uniform_decode=False,
            has_lora=False,
            num_active_loras=0,
            valid_modes=None,
            invalid_modes=None,
        ):
            if not uniform_decode:
                invalid_modes = set(invalid_modes or ())
                invalid_modes.add(CUDAGraphMode.FULL)

                # A DP re-dispatch can constrain valid_modes to FULL after the
                # cross-rank decision.  Keep the allowed set non-empty; NONE is
                # the only safe local fallback in that special case.
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

        dispatch_uniform_full._kunlunxin_uniform_full = True
        CudagraphDispatcher.dispatch = dispatch_uniform_full
        logger.info(
            "Patched CudagraphDispatcher: uniform padded decode keeps FULL; "
            "mixed batches exclude FULL"
        )
    except Exception as e:
        logger.warning("Failed to patch CudagraphDispatcher: %s", e)


def patch_breakable_cudagraph_mode():
    """Enable vLLM's breakable wrapper without a process-wide env switch.

    Setting ``VLLM_USE_BREAKABLE_CUDAGRAPH`` changes compilation behavior for
    every platform.  Patch only the already imported Kunlunxin execution path
    so PIECEWISE remains the normal compiled path and FULL can break around
    collectives that cannot be captured safely by FlagCX.
    """
    try:
        import vllm.compilation.breakable_cudagraph as breakable
        import vllm.v1.worker.gpu_model_runner as gpu_model_runner
        import vllm_fl.worker.model_runner as fl_model_runner

        def enabled() -> bool:
            return True

        breakable.is_breakable_cudagraph_enabled = enabled
        gpu_model_runner.is_breakable_cudagraph_enabled = enabled
        fl_model_runner.is_breakable_cudagraph_enabled = enabled
        logger.info("Enabled breakable cudagraph for Kunlunxin FULL graphs")
    except Exception as e:
        logger.warning("Failed to enable breakable cudagraph: %s", e)


def patch_breakable_private_pools():
    """Give each FULL batch descriptor an independent graph memory pool.

    Kunlunxin can replay one FULL graph reliably, but graphs with different
    static shapes cannot safely alias allocations from vLLM's global graph
    pool. Leaving ``graph_pool`` unset makes each CUDAGraph own its captured
    activation storage while model parameters remain shared.
    """
    try:
        from functools import wraps

        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

        original_init = BreakableCUDAGraphWrapper.__init__
        if getattr(original_init, "_kunlunxin_private_pools", False):
            return

        @wraps(original_init)
        def init_with_private_pools(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.graph_pool = None

        init_with_private_pools._kunlunxin_private_pools = True
        BreakableCUDAGraphWrapper.__init__ = init_with_private_pools
        logger.info("Enabled per-descriptor FULL graph memory pools")
    except Exception as e:
        logger.warning("Failed to isolate FULL graph memory pools: %s", e)


def patch_breakable_full_only():
    """Keep PIECEWISE on its compiled runner; break only FULL graphs."""
    try:
        from functools import wraps

        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
        original_call = BreakableCUDAGraphWrapper.__call__
        if getattr(original_call, "_kunlunxin_full_only", False):
            return

        @wraps(original_call)
        def call_full_only(self, *args, **kwargs):
            if not is_forward_context_available():
                return original_call(self, *args, **kwargs)

            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
                return self.runnable(*args, **kwargs)

            return original_call(self, *args, **kwargs)

        call_full_only._kunlunxin_full_only = True
        BreakableCUDAGraphWrapper.__call__ = call_full_only
        logger.info("Restricted breakable cudagraph capture to FULL mode")
    except Exception as e:
        logger.warning("Failed to restrict breakable cudagraph mode: %s", e)


def patch_graph_all_reduce():
    """Capture FULL decode in bounded ProcessGroup graph segments.

    The direct FlagCX ctypes path is retained for eager and PIECEWISE modes.
    FULL capture uses the registered FlagCX ProcessGroup so the collective is
    recorded in the decode graph instead of producing eager communication
    breaks. FlagCX initializes that ProcessGroup lazily, but XCCL cannot create
    its communicator while a graph capture is active. End the first segment,
    initialize the communicator once outside capture, then resume capture and
    record the real collective. The one-time warm-up is not part of replay.

    A full model forward contains roughly 128 all-reduces, which exceeds
    XCCL's per-graph allocation limit when the leaking KL3 event mode is
    disabled. End the current graph after a bounded number of collectives and
    immediately begin the next graph-only segment. Replay remains entirely
    captured and preserves stream ordering between segments.
    """
    try:
        from functools import wraps

        import torch.distributed as dist

        from vllm.compilation.breakable_cudagraph import (
            BreakableCUDAGraphCapture,
        )
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
        from vllm_fl.distributed.communicator import CommunicatorFL

        original_all_reduce = CommunicatorFL.all_reduce
        if getattr(original_all_reduce, "_kunlunxin_graph_safe", False):
            return

        @wraps(original_all_reduce)
        def graph_safe_all_reduce(self, input_):
            use_full_graph_collective = (
                torch.cuda.is_current_stream_capturing()
                and is_forward_context_available()
                and get_forward_context().cudagraph_runtime_mode
                == CUDAGraphMode.FULL
            )
            if not use_full_graph_collective:
                return original_all_reduce(self, input_)

            capture = BreakableCUDAGraphCapture.current()
            if (
                not getattr(self, "_kunlunxin_graph_pg_warmed", False)
                and capture is not None
                and capture._capturing
            ):
                # ProcessGroupFlagCX creates its XCCL communicator on the
                # first collective. That allocation fails inside capture, so
                # initialize it once between graph segments. Do not register
                # this warm-up as an eager replay segment: the actual
                # all-reduce below is still captured and replay remains FULL.
                capture._end_segment()
                warmup = input_.clone()
                dist.all_reduce(warmup, group=self.device_group)
                torch.cuda.synchronize()
                del warmup
                self._kunlunxin_graph_pg_warmed = True
                capture._begin_segment()

            output = input_.clone()
            dist.all_reduce(output, group=self.device_group)

            if capture is not None and capture._capturing:
                collectives = (
                    getattr(capture, "_kunlunxin_segment_all_reduces", 0) + 1
                )
                if collectives >= 16:
                    capture._kunlunxin_segment_all_reduces = 0
                    capture._end_segment()
                    capture._begin_segment()
                else:
                    capture._kunlunxin_segment_all_reduces = collectives
            return output

        graph_safe_all_reduce._kunlunxin_graph_safe = True
        CommunicatorFL.all_reduce = graph_safe_all_reduce
        logger.info(
            "Patched FULL graph all-reduce into bounded FlagCX graph segments"
        )
    except Exception as e:
        logger.warning("Failed to patch graph-time all-reduce: %s", e)


def patch_eager_all_gather():
    """Keep post-graph TP all-gather off the ProcessGroup event path.

    FULL replay itself needs the graph-aware ProcessGroup for collectives, but
    logits all-gather runs after replay on every decode step.  FlagCX's
    ProcessGroup creates a CUDA event for each such call and eventually
    exhausts the device event pool during a long-running concurrency sweep.
    Use the existing direct FlagCX communicator outside capture; retain the
    ProcessGroup implementation if capture is active or direct FlagCX is not
    available.
    """
    try:
        from functools import wraps

        from vllm_fl.distributed.communicator import CommunicatorFL

        original_all_gather = CommunicatorFL.all_gather
        if getattr(original_all_gather, "_kunlunxin_direct_eager", False):
            return

        @wraps(original_all_gather)
        def direct_eager_all_gather(self, input_, dim=-1):
            pyflagcx_comm = getattr(self, "pyflagcx_comm", None)
            if (
                torch.cuda.is_current_stream_capturing()
                or pyflagcx_comm is None
                or pyflagcx_comm.disabled
            ):
                return original_all_gather(self, input_, dim)

            if self.world_size == 1:
                return input_
            if dim < 0:
                dim += input_.dim()
            if not 0 <= dim < input_.dim():
                raise IndexError(
                    f"Invalid dim ({dim}) for input shape {input_.shape}"
                )

            input_tensor = input_.movedim(dim, 0).contiguous()
            output_shape = (
                input_tensor.shape[0] * self.world_size,
                *input_tensor.shape[1:],
            )
            output_tensor = torch.empty(
                output_shape,
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
            pyflagcx_comm.all_gather(output_tensor, input_tensor)
            return output_tensor.movedim(0, dim).contiguous()

        direct_eager_all_gather._kunlunxin_direct_eager = True
        CommunicatorFL.all_gather = direct_eager_all_gather
        logger.info("Patched eager TP all-gather to use direct FlagCX")
    except Exception as e:
        logger.warning("Failed to patch eager TP all-gather: %s", e)


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
    """Select a numerically stable decode path for each execution mode.

    Prefix attention avoids a numerical issue in eager and PIECEWISE decode.
    Its host LoD launch metadata cannot be updated by a captured FULL graph,
    so FULL decode uses the graph-safe paged kernel and its device lengths.
    """
    try:
        import vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention as attn_mod
        import xtorch_ops

        @staticmethod
        def patched_forward_decode(
            query, key_cache, value_cache, block_tables,
            seq_lens, seq_lens_host, max_seq_len, num_decode_tokens,
            kv_cache_dtype, num_kv_heads, scale, alibi_slopes,
            k_scale, v_scale, max_window_size=-1, output=None,
            query_start_loc=None, query_start_loc_host=None,
            kv_prefix_start_loc=None, kv_prefix_start_loc_host=None,
        ):
            """Use prefill_attention in prefix_cache mode for decode."""
            import torch

            if output is None:
                output = torch.empty_like(query)

            decode_query = query[:num_decode_tokens]
            decode_output = output[:num_decode_tokens]

            if any(
                value is None
                for value in (
                    query_start_loc,
                    query_start_loc_host,
                    kv_prefix_start_loc,
                    kv_prefix_start_loc_host,
                )
            ):
                raise RuntimeError(
                    "Kunlunxin prefix-decode requires persistent LoD metadata"
                )

            window_left = -1
            window_right = -1
            if max_window_size > 0:
                window_left = max_window_size
                window_right = 0
            alpha = scale * (float(decode_query.shape[2]) ** 0.5)
            if _is_full_graph_runtime():
                xtorch_ops.decode_paged_attention(
                    decode_query,
                    key_cache,
                    value_cache,
                    seq_lens_host[:num_decode_tokens],
                    seq_lens[:num_decode_tokens],
                    block_tables,
                    decode_output,
                    alpha=scale,
                    k_perchannel_scale=k_scale,
                    v_perchannel_scale=v_scale,
                    alibi_slopes=alibi_slopes,
                    sink=None,
                )
            else:
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
            "using prefix decode outside FULL and paged decode in FULL"
        )
    except Exception as e:
        logger.warning("Failed to patch decode attention: %s", e)
