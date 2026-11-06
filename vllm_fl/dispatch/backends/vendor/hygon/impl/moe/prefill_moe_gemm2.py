# Copyright (c) 2026 BAAI. All rights reserved.
"""Large-prefill INT8 down GEMM, using unchanged LightOp packed weights."""
import functools
import os
import types

import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["rows"])
def _prefill_down(q, weight, a_scale, b_scale, routed_weight, sorted_ids,
                  expert_ids, padded_count, output, rows):
    # Reuse G1's 128-row route plan; process two 64-row tiles per expert block.
    pid = tl.program_id(0)
    padded = tl.load(padded_count)
    m_tiles = tl.cdiv(padded, 64)
    if pid >= m_tiles * 32:
        return
    tile_m, tile_n = pid // 32, pid % 32
    positions = tile_m * 64 + tl.arange(0, 64)
    route = tl.load(sorted_ids + positions, positions < padded, other=rows)
    expert = tl.load(expert_ids + tile_m // 2)
    n = tile_n * 128 + tl.arange(0, 128)
    kk = tl.arange(0, 64)
    sa = tl.load(a_scale + route, route < rows, other=0)
    rw = tl.load(routed_weight + route, route < rows, other=0)
    # Match the LightOp down epilogue's FP32 multiplication order.
    effective_scale = sa * rw
    accumulator = tl.zeros((64, 128), tl.int32)
    for block_k in range(4):
        k = block_k * 64 + kk
        a = tl.load(q + route[:, None] * 256 + k[None, :],
                    route[:, None] < rows, other=0)
        # Packed [expert, K/64, N, K%64]; no extra weight copy or unpack buffer.
        offset = (expert * 1048576 + block_k * 262144
                  + n[None, :] * 64 + kk[:, None])
        b = tl.load(weight + offset)
        accumulator += tl.dot(a, b)
    sb = tl.load(b_scale + expert * 4096 + n)
    value = (accumulator.to(tl.float32) * effective_scale[:, None]) * sb[None, :]
    tl.store(output + route[:, None] * 4096 + n[None, :],
             value.to(tl.bfloat16), route[:, None] < rows)


def supports_prefill_down(q, w, output, sa, sb, rw, sorted_ids, experts, padded,
                          top_k, config):
    if top_k != 1 or output.ndim != 3:
        return False
    m = output.shape[0]
    if not (4096 <= m <= 8192 and output.shape[1:] == (6, 4096)):
        return False
    # LightOp G1 uses alignment 128; G2's 32*DELTA4 describes the same plan.
    if not all(config.get(k) == v for k, v in
               (("BLOCK_SIZE_M", 32), ("MODE", 568), ("DELTA", 4))):
        return False
    if (q.shape != (m * 6, 256) or w.shape != (256, 4, 262144)
            or sa.shape != (m * 6, 1) or sb.shape != (256, 4096, 1)
            or rw is None or rw.shape != (m, 6)
            or sorted_ids.ndim != 1 or experts.ndim != 1 or padded.numel() != 1):
        return False
    tensors = (q, w, output, sa, sb, rw, sorted_ids, experts, padded)
    dtypes = (torch.int8, torch.int8, torch.bfloat16, torch.float32,
              torch.float32, torch.float32, torch.int32, torch.int32, torch.int32)
    return q.is_cuda and all(t.device == q.device and t.dtype == dtype
                            and t.is_contiguous()
                            for t, dtype in zip(tensors, dtypes))


def prefill_down(q, w, output, sa, sb, rw, sorted_ids, experts, padded):
    """Keep INT32 dot and BF16((float(acc)*(sa*router_weight))*sb) semantics."""
    blocks = triton.cdiv(sorted_ids.numel(), 128)
    _prefill_down[(blocks * 2 * 32,)](
        q, w, sa, sb, rw, sorted_ids, experts, padded, output, q.shape[0],
        num_warps=4, num_stages=2, enable_fp_fusion=False,
    )
    return output


@functools.lru_cache(maxsize=4)
def _adapt(original):
    if not isinstance(original, types.FunctionType):
        return original
    original_gemm = original.__globals__.get("moe_gemm_marlin_w8a8")
    if original_gemm is None:
        return original

    def gemm(q, w, output, sa, sb, rw, sorted_ids, experts, padded, top_k, config):
        if supports_prefill_down(q, w, output, sa, sb, rw, sorted_ids, experts,
                                 padded, top_k, config):
            return prefill_down(q, w, output, sa, sb, rw, sorted_ids, experts, padded)
        return original_gemm(q, w, output, sa, sb, rw, sorted_ids, experts,
                             padded, top_k, config)

    # Preserve the private sum adapter; do not change LightOp's global bindings.
    bindings = dict(original.__globals__)
    bindings["moe_gemm_marlin_w8a8"] = gemm
    adapted = types.FunctionType(original.__code__, bindings, original.__name__,
                                 original.__defaults__, original.__closure__)
    adapted.__kwdefaults__ = original.__kwdefaults__
    functools.update_wrapper(adapted, original)
    return adapted


def adapt_prefill_moe_gemm2(original):
    if os.getenv("VLLM_FL_HYGON_PREFILL_MOE_GEMM2", "0").lower() in ("0", "off", "false", "no"):
        return original
    return _adapt(original)
