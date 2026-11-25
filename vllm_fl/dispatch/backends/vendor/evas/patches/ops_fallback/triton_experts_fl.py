# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU patch: fork ``TritonExpertsFL.apply`` to accept uint8 E4M3 activations.

The patched ``scaled_fp8_quant`` yields no native fp8 dtype on LPU, so DSV4
W8A8 MoE feeds ``apply`` uint8 E4M3 bytes the upstream dtype assert rejects.
The verbatim fork adds ``torch.uint8`` to that chain, declares
``expects_unquantized_inputs = True`` (the kernel then quantizes A1 once), and
forwards the gemm1 clamp args as upstream does; non-uint8 inputs run the
byte-identical upstream body.  Re-sync against ``fused_moe_utils.py`` on an
upstream refactor.

"""

from __future__ import annotations

import os
import sys

import torch

from ...local.double_die import (
    is_double_die_enabled,
    to_split_dim_logical,
)

from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
from vllm.model_executor.layers.fused_moe.utils import _resize_cache, moe_kernel_quantize_input
from vllm.triton_utils import tl

from vllm_fl.ops.fused_moe.activation import apply_moe_activation
from vllm_fl.ops.fused_moe.fused_moe_utils import (
    _invoke_fused_moe_triton_kernel,
    _prepare_expert_assignment,
)

_PATCHED = False
_ORIG_APPLY = None

_LOG = "[evas-adoption-moe]"
_SEEN = {"logged": False, "fallback": set(), "moe_sum_stale": False}

# The dual-die GEMM1 can fuse the SwiGLU epilogue into the projection (the
# upstream default) or run a plain gate/up projection plus a separate SiLU.
# The fused form widens the accumulator to 2 * BLOCK_SIZE_N and round-trips
# fp32 twice per tile, and on this backend it costs about 2.6x more per MAC
# than the plain shape used by the down projection (microbenchmarked: TTFT
# -45%, TPOT -36% with the plain form, and per-layer fingerprints identical,
# so the relocation is numerically inert).
# The plain form pays one extra [routes, 2, I] bf16 buffer and a separate
# activation pass instead.  Default off; set the knob to 1 to restore the
# upstream fused epilogue.
_FUSE_SILU = os.environ.get("VLLM_FL_MOE_FUSE_SILU", "0").strip() == "1"


def _kernel_cannot_serve(
    hidden_states,
    w1,
    topk_ids,
    *,
    quant_config,
    per_channel_quant,
    block_shape,
    a1_scale,
    a2_scale,
) -> list[str]:
    """Why the evas fused-experts kernel cannot serve this call, as labels.

    The kernel implements exactly one quantization scheme (fp8 W8A8 with
    dynamic block-128 activation quant, no caller scales) and one layout
    (dual die, no in-place write).  Everything else is served by the upstream
    ``TritonExpertsFL.apply`` body below, which handles every scheme vLLM
    declares -- so an unsupported declaration falls through to it instead of
    reaching the kernel.  Keeping the predicate here rather than in FlagGems
    ties it to the declaration that has to satisfy it (``fp8_methods``), and
    leaves the FlagGems export a plain re-export.
    """
    reasons = []
    use_fp8_w8a8 = bool(quant_config.use_fp8_w8a8)
    if quant_config.use_int8_w8a8:
        reasons.append("int8 W8A8")
    if quant_config.use_int8_w8a16:
        reasons.append("int8 W8A16")
    if quant_config.use_int4_w4a16:
        reasons.append("int4 W4A16")
    block_shape = None if block_shape is None else list(block_shape)
    if use_fp8_w8a8:
        if block_shape is None and not per_channel_quant:
            reasons.append("W8A16 fp8 (float activations, no activation quant)")
        elif block_shape != [128, 128] or per_channel_quant:
            reasons.append(
                f"fp8 activation quant is per_channel_quant="
                f"{bool(per_channel_quant)}, block_shape={block_shape} "
                f"(the kernel implements block-128 only)"
            )
    else:
        if per_channel_quant:
            reasons.append("per_channel_quant without fp8")
        if block_shape is not None:
            reasons.append(f"block_shape={block_shape} without fp8")
    if a1_scale is not None:
        reasons.append(
            f"caller-quantized A1 {tuple(a1_scale.shape)} {a1_scale.dtype} over "
            f"a {hidden_states.dtype} A (the kernel quantizes A1 itself)"
        )
    if a2_scale is not None:
        reasons.append(f"caller-quantized A2 {tuple(a2_scale.shape)} {a2_scale.dtype}")
    if not is_double_die_enabled():
        reasons.append("double die is off (the kernel assumes the dual-die layout)")
    if hidden_states.size(0) != topk_ids.size(0):
        reasons.append(
            f"rows differ: hidden {hidden_states.size(0)} vs "
            f"topk_ids {topk_ids.size(0)}"
        )
    alignment = 256 if use_fp8_w8a8 else 32
    if hidden_states.size(1) % alignment:
        reasons.append(f"H={hidden_states.size(1)} not a multiple of {alignment}")
    intermediate = w1.shape[1] // 2
    if intermediate % alignment:
        reasons.append(f"I={intermediate} not a multiple of {alignment}")
    return reasons


def _triton_experts_fl_apply_lpu(
    self,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    a1q_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
    apply_router_weight_on_input: bool,
):
    """Fork of ``TritonExpertsFL.apply`` (upstream body + LPU adaptations).

    The ``torch.uint8`` entries below are the LPU W8A8 contract: the patched
    ``scaled_fp8_quant`` emits uint8 E4M3 + fp32 scale, no native fp8 dtype.
    The module docstring lists ALL differences vs upstream (fast-path gate,
    kunlunxin branch) — re-sync from there.
    """
    # Fast path (no LoRA, standard SiLU): single fused FlagGems call.  Upstream
    # gates this on CUDA; here it runs on LPU too, because
    # ``flag_gems.fused_experts_impl`` is the evas fused-experts kernel.
    #
    # The kernel implements one quantization scheme and one layout, so the
    # declaration in ``fp8_methods`` has to match it; ``_kernel_cannot_serve``
    # names the mismatch and the call falls through to the upstream body
    # below, which serves every scheme.  That keeps the contract next to the
    # declaration that has to satisfy it (this used to be an adapter in
    # FlagGems ``_wire_moe.py``).
    #
    # No ``config`` is passed: the kernel derives its own block sizes from the
    # FlagGems tune tables (one per GEMM stage), so a vLLM-computed config
    # cannot reach it and the kernel does not declare the parameter.
    fast_path = (
        self._lora_context is None
        and activation == MoEActivation.SILU
        and self.gemm1_clamp_limit is None
    )
    if fast_path:
        reasons = _kernel_cannot_serve(
            hidden_states,
            w1,
            topk_ids,
            quant_config=self.quant_config,
            per_channel_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
            a1_scale=a1q_scale,
            a2_scale=a2_scale,
        )
        if reasons:
            fast_path = False
            key = tuple(reasons)
            if key not in _SEEN["fallback"]:
                _SEEN["fallback"].add(key)
                # Straight to stderr: under vLLM our INFO records are dropped by
                # the worker, and this line is the discriminator between "the
                # kernel ran" and "the upstream body ran".
                print(
                    f"{_LOG} fused_experts_impl: upstream body "
                    f"({'; '.join(reasons)})",
                    file=sys.stderr,
                    flush=True,
                )
    if fast_path:
        import flag_gems

        E, _num_tokens, _N, _K, _top_k = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        if global_num_experts == -1:
            global_num_experts = E
        if not _SEEN["logged"]:
            _SEEN["logged"] = True
            # Reports what actually crosses the boundary -- the incoming
            # flags, not a claim about them -- so the log alone says which
            # scheme the MoE ran.
            print(
                f"{_LOG} fused_experts_impl fast path: hidden "
                f"{tuple(hidden_states.shape)} {hidden_states.dtype} split_dim="
                f"{hidden_states.split_dim()} -> logical [M, H] "
                f"(kernel replicates); per_channel_quant "
                f"{self.per_act_token_quant}, block_shape {self.block_shape} "
                f"(as declared), fuse_silu={_FUSE_SILU}",
                file=sys.stderr,
                flush=True,
            )
        output.copy_(flag_gems.fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=False,
            activation=activation.value,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
            use_int8_w8a8=self.quant_config.use_int8_w8a8,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            per_channel_quant=self.per_act_token_quant,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=self.w1_scale,
            w2_scale=self.w2_scale,
            a1_scale=a1q_scale,
            a2_scale=a2_scale,
            block_shape=self.block_shape,
            w1_bias=self.w1_bias,
            w2_bias=self.w2_bias,
            fuse_silu=_FUSE_SILU,
        ))
        return

    # LoRA path: step-by-step pipeline (call_op dispatch) so LoRA
    # adapters can be injected between GEMM1/activation/GEMM2.
    # Check constraints.
    if self.quant_config.use_int4_w4a16:
        assert hidden_states.size(-1) // 2 == w1.size(2), "Hidden size mismatch"
    else:
        assert hidden_states.size(-1) == w1.size(2), (
            f"Hidden size mismatch {hidden_states.size(-1)} != {w1.size(2)}"
        )

    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert hidden_states.dim() == 2
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.uint8,  # LPU: E4M3 bytes in a uint8 container (no fp8 dtype)
    ]

    # LPU declares ``expects_unquantized_inputs`` (the fused export below
    # quantizes A1 itself), so the prepare step deferred the first activation
    # quantization to the experts kernel and A arrives float.  Do what the
    # prepare step would have done -- same entry point, same flags -- and
    # leave the already-quantized contract (bytes + scale) untouched for
    # callers that still hand it over pre-quantized.
    #
    # The float tensor is kept for the LoRA shrink input, which needs
    # unquantized activations (upstream does the same).
    lora_unquantized_hidden_states = None
    if self.expects_unquantized_inputs and a1q_scale is None:
        lora_unquantized_hidden_states = hidden_states
        hidden_states, a1q_scale = moe_kernel_quantize_input(
            hidden_states,
            self.a1_scale or self.a1_gscale,
            self.quant_dtype,
            self.per_act_token_quant,
            self.block_shape,
            quantization_emulation=self.quantization_emulation,
        )

    E, num_tokens, N, K, top_k_num = self.moe_problem_size(
        hidden_states, w1, w2, topk_ids
    )

    if global_num_experts == -1:
        global_num_experts = E

    config = try_get_optimal_moe_config(
        w1.size(),
        w2.size(),
        top_k_num,
        self.quant_config.config_name(hidden_states.dtype),
        num_tokens,
        block_shape=self.block_shape,
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    elif (
        hidden_states.dtype == torch.float8_e4m3fn
        or hidden_states.dtype == torch.float8_e4m3fnuz
        or hidden_states.dtype == torch.uint8  # LPU E4M3 bytes
    ):
        compute_type = tl.bfloat16
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    # Note that the output tensor might be in workspace1
    intermediate_cache1 = _resize_cache(workspace2, (num_tokens, top_k_num, N))
    cache2_dim = self.adjust_N_for_activation(N, activation)
    intermediate_cache2 = _resize_cache(
        workspace13, (num_tokens * top_k_num, cache2_dim)
    )
    intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))

    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        _prepare_expert_assignment(
            topk_ids,
            config,
            num_tokens,
            top_k_num,
            global_num_experts,
            expert_map,
            use_int8_w8a16=self.quant_config.use_int8_w8a16,
            use_int4_w4a16=self.quant_config.use_int4_w4a16,
            block_shape=self.block_shape,
        )
    )

    _invoke_fused_moe_triton_kernel(
        hidden_states,
        w1,
        intermediate_cache1,
        a1q_scale,
        self.w1_scale,
        None,  # topk_weights
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,  # mul_routed_weights
        top_k_num,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
        use_int8_w8a8=self.quant_config.use_int8_w8a8,
        use_int8_w8a16=self.quant_config.use_int8_w8a16,
        use_int4_w4a16=self.quant_config.use_int4_w4a16,
        per_channel_quant=self.per_act_token_quant,
        block_shape=self.block_shape,
        B_bias=self.w1_bias,
    )

    # LoRA w13: applied to intermediate_cache1 before activation, using the
    # unquantized activation as the lora_a input (float when this call
    # quantized A1 itself).  moe_lora_align_block_size is called once here and
    # results reused for the w2 LoRA below.
    sorted_token_ids_lora = None
    expert_ids_lora = None
    num_tokens_post_padded_lora = None
    token_lora_mapping = None
    lora_context = self._lora_context
    if lora_context is not None:
        # The shrink kernel needs unquantized activations: prefer the float
        # tensor this call quantized itself, else the stash the caller left on
        # the context (same shape guard as upstream, so a dispatched layout
        # never indexes it out of bounds).
        if lora_unquantized_hidden_states is not None:
            lora_x = lora_unquantized_hidden_states
        elif (
            getattr(lora_context, "original_hidden_states", None) is not None
            and lora_context.original_hidden_states.shape[0] == hidden_states.shape[0]
        ):
            lora_x = lora_context.original_hidden_states
        else:
            lora_x = hidden_states
        (
            sorted_token_ids_lora,
            expert_ids_lora,
            num_tokens_post_padded_lora,
            token_lora_mapping,
        ) = self.apply_w13_lora(
            lora_context,
            y=intermediate_cache1,
            x=lora_x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expert_map=expert_map,
            w1=w1,
            w2=w2,
            num_tokens=num_tokens,
            top_k_num=top_k_num,
        )

    apply_moe_activation(
        activation,
        intermediate_cache2,
        intermediate_cache1.view(-1, N),
        clamp_limit=self.gemm1_clamp_limit,
        alpha=self.gemm1_alpha,
        beta=self.gemm1_beta,
    )

    a2q_scale: torch.Tensor | None = None

    qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
        intermediate_cache2,
        a2_scale,
        self.quant_dtype,
        self.per_act_token_quant,
        self.block_shape,
        quantization_emulation=self.quantization_emulation,
    )

    _invoke_fused_moe_triton_kernel(
        qintermediate_cache2,
        w2,
        intermediate_cache3,
        a2q_scale,
        self.w2_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=self.quant_config.use_fp8_w8a8,
        use_int8_w8a8=self.quant_config.use_int8_w8a8,
        use_int8_w8a16=self.quant_config.use_int8_w8a16,
        use_int4_w4a16=self.quant_config.use_int4_w4a16,
        per_channel_quant=self.per_act_token_quant,
        block_shape=self.block_shape,
        B_bias=self.w2_bias,
    )

    # LoRA w2: applied to intermediate_cache3 before moe_sum, using the
    # unquantized intermediate_cache2 as the lora_a input.  Reuses the
    # sorted_token_ids_lora computed above.
    if lora_context is not None:
        self.apply_w2_lora(
            lora_context,
            y=intermediate_cache3,
            x=intermediate_cache2,
            topk_weights=topk_weights,
            sorted_token_ids_lora=sorted_token_ids_lora,
            expert_ids_lora=expert_ids_lora,
            num_tokens_post_padded_lora=num_tokens_post_padded_lora,
            token_lora_mapping=token_lora_mapping,
            num_tokens=num_tokens,
            w1=w1,
            w2=w2,
            top_k_num=top_k_num,
        )

    # separate function is required for MoE + LoRA.
    #
    # The dual-die moe_sum kernel writes through the die shards of its OUTPUT,
    # and it obtains those shards from ``unzip_logical``, which repairs a stale
    # split descriptor by cloning.  A clone built inside that call is dropped on
    # return, so the whole result is lost whenever the output still carries the
    # descriptor of its parent buffer -- which is exactly what a view-derived
    # output does (``empty_like`` of a row slice, a reshape of one).  Rebuilding
    # the descriptor here makes the write reach a tensor this frame owns, and
    # ``to_split_dim_logical`` returns the tensor unchanged when the descriptor
    # already matches, so the ordinary call site pays nothing extra.  Split
    # descriptors only exist on LPU storage, so the repair is skipped for host
    # tensors (the CPU unit tests drive this fork on plain CPU storage).
    moe_sum_out = output
    if is_double_die_enabled() and output.device.type == "lpu":
        moe_sum_out = to_split_dim_logical(output, 0)
    self.moe_sum(intermediate_cache3, moe_sum_out)
    if moe_sum_out is not output:
        output.copy_(moe_sum_out)
        if not _SEEN["moe_sum_stale"]:
            _SEEN["moe_sum_stale"] = True
            print(
                f"{_LOG} moe_sum output carried a stale double-die descriptor; "
                "writing through a repaired copy instead"
            )


def _lpu_expects_unquantized_inputs(self) -> bool:
    """LPU: the experts kernel owns the first activation quantization.

    ``FusedMoEExperts.expects_unquantized_inputs`` tells the prepare step to
    skip ``moe_kernel_quantize_input`` and hand the raw activation to the
    experts kernel.  On LPU both experts bodies quantize A1 themselves (the
    adopted evas kernel per chunk, the pure twin and this fork through
    ``moe_kernel_quantize_input``), so the caller-side quantization plus the
    A1 dequant it used to require are both unnecessary -- one quantization,
    in the kernel that consumes it.
    """
    return True


def apply_lpu_triton_experts_fl_patch() -> None:
    """Rebind ``TritonExpertsFL.apply`` to the uint8-capable fork. Idempotent."""
    global _PATCHED, _ORIG_APPLY
    if _PATCHED:
        return
    from vllm_fl.ops.fused_moe.fused_moe_utils import TritonExpertsFL

    _ORIG_APPLY = TritonExpertsFL.apply
    TritonExpertsFL.apply = _triton_experts_fl_apply_lpu
    # Class-level like ``apply``: this module is evas-only, so the rebind
    # cannot reach the kunlunxin consumers of the same class.
    TritonExpertsFL.expects_unquantized_inputs = property(
        _lpu_expects_unquantized_inputs
    )
    _PATCHED = True
