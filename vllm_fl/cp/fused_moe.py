# SPDX-License-Identifier: Apache-2.0
"""Restricted Qwen PCP prefill fusion around the native Triton experts."""

from contextlib import contextmanager
from types import MethodType

import torch
import triton
import triton.language as tl


@triton.jit
def _sum_gate_add(EXPERTS, GATE, SHARED, OUT, SIZE: tl.constexpr,
                 WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < SIZE
    base = i // WIDTH * (8 * WIDTH) + i % WIDTH
    # ATen Reduce.cuh, strided top-8 BF16 sum: four FP32 accumulators,
    # each accumulating k and k+4, then combined left to right (vt0=4).
    a0 = tl.load(EXPERTS + base, mask, other=0).to(tl.float32)
    a1 = tl.load(EXPERTS + base + WIDTH, mask, other=0).to(tl.float32)
    a2 = tl.load(EXPERTS + base + 2 * WIDTH, mask, other=0).to(tl.float32)
    a3 = tl.load(EXPERTS + base + 3 * WIDTH, mask, other=0).to(tl.float32)
    a4 = tl.load(EXPERTS + base + 4 * WIDTH, mask, other=0).to(tl.float32)
    a5 = tl.load(EXPERTS + base + 5 * WIDTH, mask, other=0).to(tl.float32)
    a6 = tl.load(EXPERTS + base + 6 * WIDTH, mask, other=0).to(tl.float32)
    a7 = tl.load(EXPERTS + base + 7 * WIDTH, mask, other=0).to(tl.float32)
    total = (((a0 + a4) + (a1 + a5)) + (a2 + a6)) + (a3 + a7)
    routed = total.to(tl.bfloat16).to(tl.float32)
    gate = tl.load(GATE + i // WIDTH, mask, other=0).to(tl.float32)
    shared = tl.load(SHARED + i, mask, other=0).to(tl.float32)
    product = (gate * shared).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + i, product + routed, mask)


def sum_gate_add(experts, gate, shared, out):
    """Fuse the audited contiguous top-8/2048 BF16 sum and gated addition.

    Return False without writing output for unsupported layouts.
    """
    if (not shared.is_cuda or shared.dtype != torch.bfloat16
            or shared.ndim != 2 or shared.shape[1] != 2048
            or experts.shape != (shared.shape[0], 8, 2048)
            or gate.shape != (shared.shape[0], 1) or out.shape != shared.shape
            or any(x.dtype != shared.dtype or x.device != shared.device
                   or not x.is_contiguous() for x in (experts, gate, shared, out))):
        return False
    if out.numel():
        _sum_gate_add[(triton.cdiv(out.numel(), 1024),)](
            experts, gate, shared, out, out.numel(), 2048, BLOCK=1024,
            enable_fp_fusion=False,
        )
    return True


@contextmanager
def _fused_moe_sum(mlp, gate_output, out=None):
    """Temporarily fuse Triton's top-k sum with the shared-expert add."""
    runner = mlp.experts.runner
    impl = runner.quant_method.moe_kernel.impl
    experts = impl.fused_experts
    original = experts.moe_sum
    sentinel = object()
    previous = experts.__dict__.get("moe_sum", sentinel)
    fused = {"used": False, "output": None}

    def fused_sum(self, expert_output, output):
        # _eligible requires NO_OVERLAP: this call follows shared execution
        # on the same stream, before the runner consumes/reset its output.
        shared = runner._shared_experts._output[runner._shared_experts._output_idx]
        destination = output if out is None else out
        if (shared is not None and sum_gate_add(
                expert_output, gate_output, shared, destination)):
            fused["used"] = True
            fused["output"] = destination
            return
        original(expert_output, output)

    experts.moe_sum = MethodType(fused_sum, experts)
    try:
        yield fused
    finally:
        if previous is sentinel:
            del experts.moe_sum
        else:
            experts.moe_sum = previous


@triton.jit
def _gate_add(GATE, SHARED, ROUTED, OUT, SIZE: tl.constexpr,
              WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < SIZE
    gate = tl.load(GATE + i // WIDTH, mask, other=0).to(tl.float32)
    shared = tl.load(SHARED + i, mask, other=0).to(tl.float32)
    routed = tl.load(ROUTED + i, mask, other=0).to(tl.float32)
    # Preserve the BF16 product rounding before the original output addition.
    product = (gate * shared).to(OUT.dtype.element_ty).to(tl.float32)
    tl.store(OUT + i, product + routed, mask)


def gate_add(gate, shared, routed, *, out=None):
    if (not shared.is_cuda or shared.dtype != torch.bfloat16
            or shared.ndim != 2 or routed.shape != shared.shape
            or gate.shape != (shared.shape[0], 1)
            or any(x.dtype != shared.dtype or x.device != shared.device
                   or not x.is_contiguous() for x in (gate, shared, routed))):
        value = gate * shared + routed
        return value if out is None else out.copy_(value)
    if out is None:
        out = torch.empty_like(shared)
    if out.numel():
        _gate_add[(triton.cdiv(out.numel(), 1024),)](
            gate, shared, routed, out, out.numel(), out.shape[1],
            BLOCK=1024, enable_fp_fusion=False,
        )
    return out


def _eligible(mlp, full_input):
    """Only the audited BF16, no-DP/EP, synchronous Qwen expert path."""
    if not full_input.is_cuda or full_input.dtype != torch.bfloat16:
        return False
    from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        FusedTopKRouter,
    )
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExpertsOrder,
    )
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    runner = mlp.experts.runner
    shared = getattr(mlp, "shared_expert", None)
    if (type(runner) is not MoERunner or type(shared) is not Qwen2MoeMLP
            or type(runner.quant_method) is not UnquantizedFusedMoEMethod
            or type(runner.router) is not FusedTopKRouter
            or shared.expert_gate is None or shared.down_proj.reduce_results
            or runner.routed_input_transform is not None
            or runner.routed_output_transform is not None
            or runner.routed_scaling_factor != 1.0 or runner.enable_dbo
            or runner.moe_config.dp_size != 1 or runner.moe_config.ep_size != 1
            or runner.moe_config.pcp_size != 1
            or runner.moe_config.is_sequence_parallel
            or runner.moe_config.hidden_dim != full_input.shape[-1]
            or runner._shared_experts is None
            or runner._shared_experts._layer is not shared):
        return False
    kernel = runner.quant_method.moe_kernel
    if kernel is None or type(kernel.impl) is not FusedMoEKernelModularImpl:
        return False
    impl = kernel.impl
    return (type(impl.prepare_finalize) is MoEPrepareAndFinalizeNoDPEPModular
            and type(impl.fused_experts) is TritonExperts
            and not impl.inplace and impl.shared_experts is None
            and runner._shared_experts._determine_shared_experts_order(full_input)
            == SharedExpertsOrder.NO_OVERLAP)


@contextmanager
def _ungated_no_copy(shared, impl, fused=None):
    """Temporarily customize only this instance for one synchronous forward."""
    gate = shared.expert_gate
    sentinel = object()
    previous = impl.__dict__.get("_finalize", sentinel)

    def already_finalized(output, fused_out, *args, **kwargs):
        # Exact NoDPEP + TritonExperts path: weighting/sum already completed.
        # An owned destination bypasses the borrowed native output entirely.
        if fused is not None and fused["used"]:
            return fused["output"]
        return fused_out

    shared.expert_gate = None
    impl._finalize = already_finalized
    try:
        yield gate
    finally:
        shared.expert_gate = gate
        if previous is sentinel:
            del impl._finalize
        else:
            impl._finalize = previous


def try_prefill_moe(mlp, full_input, *, out=None):
    """Return combined TP partials, or None to retain the general runner.

    With ``out``, write and return that caller-owned contiguous tensor. It must
    be independent of native workspace and remain alive until its consumers
    finish. Without it, the fused result borrows the native MoE workspace.
    """
    if not _eligible(mlp, full_input):
        return None
    if out is not None and (
            out.shape != full_input.shape or out.dtype != full_input.dtype
            or out.device != full_input.device or not out.is_contiguous()):
        raise ValueError("MoE output must match the input and be contiguous")
    runner = mlp.experts.runner
    if mlp.experts.is_internal_router:
        router_logits = full_input
    else:
        router_logits, _ = mlp.gate(full_input)
    gate = mlp.shared_expert.expert_gate
    gate_output = torch.sigmoid(gate(full_input)[0])
    with _fused_moe_sum(mlp, gate_output, out) as fused:
        with _ungated_no_copy(
                mlp.shared_expert, runner.quant_method.moe_kernel.impl, fused):
            # Keep the native custom-op boundary and its routing, expert
            # kernels, bookkeeping and stream handling. Guards above exclude
            # outer runner transforms, hidden padding and scaling.
            shared_output, routed_output = runner._forward_entry(
                full_input, router_logits, full_input, None,
                runner._encode_layer_name(), 0,
            )
    if fused["used"]:
        # With no owned destination this still aliases native workspace.
        return routed_output
    return gate_add(gate_output, shared_output, routed_output, out=out)
