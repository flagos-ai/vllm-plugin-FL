"""Ordinary TP8 experts: use the lossless packed layout and paired FP4 kernels.

TP8 keeps all 896 experts with I384. BF16 GEMM boundaries, FP32 SiTU
and the existing top16 reduction remain unchanged.
"""

import torch
import triton
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.marlin_mxfp4 import pack_matrix, _marlin_tail_gemv
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.expert_gemm import _paired_arithmetic_mm, meta
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.decode_experts import (
    _paired_scheduled_mm,
    _route_gemv_g2,
)
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.grouped_moe import _alignment, _situ, _weighted_sum
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.small_alignment import align_small
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.tp8_expert_g1 import _tp8_g1_situ


from vllm_fl.ops.kimi_k3.workspace_contract import (
    MIN_PREFILL_TOKENS,
    MAX_PREFILL_TOKENS,
    ROUTE_COLUMNS,
)


def prepare(w13, s13, w2, s2):
    shapes = ((896, 768, 1792), (896, 768, 112), (896, 3584, 192), (896, 3584, 12))
    values = (w13, s13, w2, s2)
    if any(
        tuple(t.shape) != s
        or t.dtype != torch.uint8
        or not t.is_contiguous()
        or t.device != w13.device
        for t, s in zip(values, shapes)
    ):
        raise ValueError("Expected ordinary TP8 I384 checkpoint weights")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Weight preparation must precede Graph capture")
    ranges = [(int(t.min().item()), int(t.max().item())) for t in (s13, s2)]
    if any(hi == 255 for lo, hi in ranges):
        raise ValueError("Nonfinite E8M0 scale")
    a, sa = pack_matrix(w13, s13, 128)
    b, sb = pack_matrix(w2, s2, 32)
    packed = (a, sa, b, sb)
    assert sum(t.numel() * t.element_size() for t in values) == sum(
        t.numel() * t.element_size() for t in packed
    )
    return packed, {
        "normal_flags": tuple(lo >= 2 and hi <= 252 for lo, hi in ranges),
        "scale_ranges": ranges,
        "pointers": tuple(t.data_ptr() for t in packed),
        "sms": torch.cuda.get_device_properties(w13.device).multi_processor_count,
    }


def run(x, packed, weights, ids, prepared, record=None, *, tiles=None, workspace=None):
    m = x.shape[0]
    if (
        x.shape != (m, 3584)
        or x.dtype != torch.bfloat16
        or ids.shape != (m, 16)
        or weights.shape != (m, 16)
        or weights.dtype != torch.float32
    ):
        raise ValueError("Expected BF16 H3584 and FP32 top16")
    if ids.dtype not in (torch.int32, torch.int64) or not all(
        t.is_contiguous() and t.device == x.device for t in (x, weights, ids, *packed)
    ):
        raise ValueError("Expected contiguous tensors on one GPU")
    if tuple(t.data_ptr() for t in packed) != prepared["pointers"]:
        raise ValueError("Prepared weight identity changed")
    if m == 0:
        return torch.empty_like(x)
    # Preserve baseline tiles for tails and untuned shapes.
    # Qualified middle batches select scheduled G2 below.
    bm = 16 if m < 256 else 32 if m < 2048 else 64
    bn1, bk1, w1, bn2, w2, tail = (
        64 if m == 1 else 128,
        128,
        4,
        64 if m <= 8 else 128,
        4,
        False,
    )
    if tiles is not None:
        bm, bn1, bk1, w1, bn2, w2, tail = tiles
    r = m * 16
    si, ei, total = align_small(ids, bm, 896) if m <= 64 else _alignment(ids, bm, 896)
    cap = min(triton.cdiv(si.numel(), bm), min(896, r) + (r - min(896, r)) // bm)
    # Bound the prefill routing scratch; each output column retains the same
    # FP32 dot, BF16 rounding and top16 reduction as the full-width path.
    bounded = tiles is None and MIN_PREFILL_TOKENS <= m <= MAX_PREFILL_TOKENS
    down_width = ROUTE_COLUMNS if bounded else 3584
    if bounded and not torch.cuda.is_current_stream_capturing():
        if workspace is None:
            raise RuntimeError("TP8 prefill requires a reserved model workspace")
        scratch, arena_act = workspace.get(x.device)
        act = arena_act[:r]
    else:
        scratch = torch.empty(
            r * max(768, down_width), dtype=torch.bfloat16, device=x.device
        )
        act = torch.empty((r, 384), dtype=torch.bfloat16, device=x.device)
    gu = scratch[: r * 768].view(r, 768)
    down = scratch[: r * down_width].view(r, down_width)
    out = torch.empty_like(x)
    a, sa, b, sb = packed
    fused_g1 = tiles is None and m in (2, 8, 12, 16, 24, 32)
    if fused_g1:
        first = _tp8_g1_situ[(cap * 12,)](
            x,
            a,
            sa,
            act,
            si,
            ei,
            total,
            r,
            prepared["normal_flags"][0],
            num_warps=2,
            num_stages=1,
            enable_fp_fusion=False,
        )
    elif tail:
        first = _marlin_tail_gemv[(r, 192)](
            x,
            a,
            sa,
            gu,
            ids,
            r,
            768,
            3584,
            128,
            4,
            512,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        first = _paired_arithmetic_mm[(cap * triton.cdiv(768, bn1),)](
            x,
            a,
            sa,
            gu,
            si,
            ei,
            total,
            768,
            3584,
            cap * bm,
            r,
            16,
            bm,
            bn1,
            bk1,
            1,
            PACK_K=128,
            MODE=2 if prepared["normal_flags"][0] else 1,
            num_warps=w1,
            num_stages=2,
        )
    if not fused_g1:
        _situ[(triton.cdiv(r * 384, 256),)](
            gu, act, r * 384, 384, 4.0, 25.0, 256, num_warps=4, enable_fp_fusion=False
        )
    if bounded:
        slice_meta = []
        for start in range(0, 3584, down_width):
            width = min(down_width, 3584 - start)
            down_part = scratch[: r * width].view(r, width)
            second = _paired_arithmetic_mm[(cap * triton.cdiv(width, bn2),)](
                act,
                b,
                sb,
                down_part,
                si,
                ei,
                total,
                3584,
                384,
                cap * bm,
                r,
                1,
                bm,
                bn2,
                32,
                1,
                PACK_K=32,
                MODE=2 if prepared["normal_flags"][1] else 1,
                N_START=start,
                OUTPUT_N=width,
                num_warps=w2,
                num_stages=2,
            )
            _weighted_sum[(m, triton.cdiv(width, 256))](
                down_part,
                weights,
                out,
                3584,
                16,
                16,
                256,
                ROUTE_H=width,
                OUTPUT_START=start,
                num_warps=4,
                enable_fp_fusion=False,
            )
            if record is not None:
                slice_meta.append(meta(second))
        if record is not None:
            record.update(
                G1=meta(first),
                G2=meta(second),
                G2_slices=slice_meta,
                G1_fused_situ=fused_g1,
                tiles=(bm, bn1, bk1, w1, bn2, w2, tail),
                bounded_route_columns=down_width,
                scratch_bytes=scratch.numel() * 2,
            )
        return out
    if tail:
        programs = min(r * 56, prepared["sms"] * 16)
        second = _route_gemv_g2[(programs,)](
            act,
            b,
            sb,
            down,
            ids,
            r,
            3584,
            64,
            programs,
            prepared["normal_flags"][1],
            K=384,
            num_warps=1,
            num_stages=1,
            enable_fp_fusion=False,
        )
    elif tiles is None and m in (12, 16, 24, 32):
        programs = min(cap * 56, prepared["sms"] * 48)
        second = _paired_scheduled_mm[(programs,)](
            act,
            b,
            sb,
            down,
            si,
            ei,
            total,
            3584,
            384,
            r,
            1,
            16,
            64,
            32,
            1,
            programs,
            False,
            0,
            2 if prepared["normal_flags"][1] else 1,
            num_warps=4,
            num_stages=2,
        )
    else:
        second = _paired_arithmetic_mm[(cap * triton.cdiv(3584, bn2),)](
            act,
            b,
            sb,
            down,
            si,
            ei,
            total,
            3584,
            384,
            cap * bm,
            r,
            1,
            bm,
            bn2,
            32,
            1,
            PACK_K=32,
            MODE=2 if prepared["normal_flags"][1] else 1,
            num_warps=w2,
            num_stages=2,
        )
    _weighted_sum[(m, 14)](
        down, weights, out, 3584, 16, 16, 256, num_warps=4, enable_fp_fusion=False
    )
    if record is not None:
        record.update(
            G1=meta(first),
            G2=meta(second),
            G1_fused_situ=fused_g1,
            tiles=(bm, bn1, bk1, w1, bn2, w2, tail),
        )
    return out
