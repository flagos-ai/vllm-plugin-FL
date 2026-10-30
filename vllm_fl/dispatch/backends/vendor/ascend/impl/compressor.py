# Copyright (c) 2026 BAAI. All rights reserved.
"""Ordered FP32 state projection and paged DeepSeek Compressor kernels."""

import torch
import triton
import triton.language as tl

from .triton_utils import (
    get_aicore_num,
    get_vectorcore_num,
    init_device_properties_triton,
)


@triton.jit
def _scatter_projection(
    value,
    APE,
    POS,
    SLOT,
    STATE,
    m0,
    c0,
    N: tl.constexpr,
    C: tl.constexpr,
    NP: tl.constexpr,
    BS: tl.constexpr,
    P0: tl.constexpr,
    SL0: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    R: tl.constexpr,
    GATE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    f = tl.arange(0, BM * BN)
    n = (m0 + f // BN).to(tl.int64)
    c = c0 + f % BN
    slot = tl.load(SLOT + n * SL0, n < N, other=-1).to(tl.int64)
    valid = (n < N) & (c < C) & (slot >= 0) & (slot < NP * BS)
    safe_slot = tl.where(valid, slot, 0)
    safe_c = tl.where(c < C, c, 0).to(tl.int64)
    offset = (safe_slot // BS) * S0 + (safe_slot % BS) * S1 + safe_c * S2
    result = value.reshape(BM * BN)
    if GATE:
        pos = tl.load(POS + n * P0, n < N, other=0).to(tl.int64)
        # R is a power of two: this is floor-mod even for negative positions.
        ape = tl.load(APE + (pos & (R - 1)) * A0 + safe_c * A1, valid, other=0)
        result = result + ape.to(tl.float32)
        offset = offset + C * S2
    tl.store(STATE + offset, result, valid)


@triton.jit
def _project_cube(
    X,
    W,
    APE,
    POS,
    SLOT,
    STATE,
    N: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    NP: tl.constexpr,
    BS: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    P0: tl.constexpr,
    SL0: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    R: tl.constexpr,
    GATE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    nc: tl.constexpr = triton.cdiv(C, BN)
    total: tl.constexpr = triton.cdiv(N, BM) * nc
    for work in range(tl.program_id(0), total, tl.num_programs(0)):
        m0 = (work // nc) * BM
        c0 = (work % nc) * BN
        m = (m0 + tl.arange(0, BM)).to(tl.int64)
        c = (c0 + tl.arange(0, BN)).to(tl.int64)
        slot = tl.load(SLOT + m * SL0, m < N, other=-1).to(tl.int64)
        valid = (m < N) & (slot >= 0) & (slot < NP * BS)
        acc = tl.zeros((BM, BN), tl.float32)
        for start in range(triton.cdiv(H, BK)):
            h = (start * BK + tl.arange(0, BK)).to(tl.int64)
            x = tl.load(
                X + m[:, None] * X0 + h[None, :] * X1,
                (m[:, None] < N) & (h[None, :] < H),
                other=0,
            )
            # Keep the memory bounds rectangular. Combining the runtime slot
            # mask here makes Ascend lower a tail tile to a full BM-row copy
            # followed by a select, which reads beyond the input allocation.
            x = tl.where(valid[:, None], x, 0)
            w = tl.load(
                W + c[:, None] * W0 + h[None, :] * W1,
                (c[:, None] < C) & (h[None, :] < H),
                other=0,
            )
            acc = tl.dot(x, tl.trans(w), acc)
        _scatter_projection(
            acc,
            APE,
            POS,
            SLOT,
            STATE,
            m0,
            c0,
            N,
            C,
            NP,
            BS,
            P0,
            SL0,
            A0,
            A1,
            S0,
            S1,
            S2,
            R,
            GATE,
            BM,
            BN,
        )


@triton.jit
def _project_vector(
    X,
    W,
    APE,
    POS,
    SLOT,
    STATE,
    N: tl.constexpr,
    H: tl.constexpr,
    C: tl.constexpr,
    NP: tl.constexpr,
    BS: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    P0: tl.constexpr,
    SL0: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    R: tl.constexpr,
    GATE: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    nc: tl.constexpr = triton.cdiv(C, BN)
    for work in range(tl.program_id(0), N * nc, tl.num_programs(0)):
        n = (work // nc).to(tl.int64)
        c0 = (work % nc) * BN
        slot = tl.load(SLOT + n * SL0).to(tl.int64)
        if (slot >= 0) & (slot < NP * BS):
            acc = tl.full((BN,), 0, tl.float32)
            for start in range(triton.cdiv(H, BK)):
                h = (start * BK + tl.arange(0, BK)).to(tl.int64)
                x = tl.load(X + n * X0 + h * X1, h < H, other=0).to(tl.float32)
                # Keep strided gathers flat: Ascend's 2D gather lowering can
                # misread noncontiguous weights before the reduction.
                f = tl.arange(0, BN * BK)
                wc = (c0 + f // BK).to(tl.int64)
                wh = (start * BK + f % BK).to(tl.int64)
                w = (
                    tl.load(
                        W + wc * W0 + wh * W1,
                        (wc < C) & (wh < H),
                        other=0,
                    )
                    .to(tl.float32)
                    .reshape(BN, BK)
                )
                acc = acc + tl.sum(w * x[None, :], axis=1)
            _scatter_projection(
                acc,
                APE,
                POS,
                SLOT,
                STATE,
                n,
                c0,
                N,
                C,
                NP,
                BS,
                P0,
                SL0,
                A0,
                A1,
                S0,
                S1,
                S2,
                R,
                GATE,
                1,
                BN,
            )


@triton.jit
def _history(
    STATE,
    TABLE,
    REQUEST,
    POSITION,
    SCRATCH,
    HAS,
    U: tl.constexpr,
    D: tl.constexpr,
    NP: tl.constexpr,
    BS: tl.constexpr,
    NR: tl.constexpr,
    NC: tl.constexpr,
    CS_ROWS: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    R0: tl.constexpr,
    P0: tl.constexpr,
    R: tl.constexpr,
    COFF: tl.constexpr,
    CD: tl.constexpr,
):
    tiles: tl.constexpr = triton.cdiv(D, CD)
    L: tl.constexpr = R * COFF
    for work in range(tl.program_id(0), U * tiles, tl.num_programs(0)):
        row = (work // tiles).to(tl.int64)
        tile = work % tiles
        req = tl.load(REQUEST + row * R0).to(tl.int64)
        pos = tl.load(POSITION + row * P0).to(tl.int64)
        rope_pos = pos - (pos & (R - 1))
        active = (req >= 0) & (req < NR) & (pos >= 0)
        active = active & ((pos & (R - 1)) == R - 1) & (rope_pos < CS_ROWS)
        has_source = False
        if active:
            f = tl.arange(0, L * CD)
            history = f // CD
            channel = tile * CD + f % CD
            source = pos - (L - 1) + history.to(tl.int64)
            valid = (source >= 0) & (source // BS < NC)
            safe_source = tl.where(valid, source, 0)
            page = tl.load(
                TABLE + req * B0 + (safe_source // BS) * B1,
                valid,
                other=-1,
            ).to(tl.int64)
            valid = valid & (page >= 0) & (page < NP)
            safe_page = tl.where(valid, page, 0)
            safe_row = tl.where(valid, source % BS, 0)
            safe_channel = tl.where(channel < D, channel, 0).to(tl.int64)
            half = (history // R).to(tl.int64)
            offset = safe_page * S0 + safe_row * S1
            valid_channel = valid & (channel < D)
            kv = tl.load(
                STATE + offset + (half * D + safe_channel) * S2,
                valid_channel,
                other=0,
            ).reshape(L, CD)
            gate = tl.load(
                STATE + offset + ((COFF + half) * D + safe_channel) * S2,
                valid_channel,
                other=0,
            ).reshape(L, CD)
            valid2 = valid.reshape(L, CD)
            has_source = tl.sum(valid.to(tl.int32), 0) > 0
            if has_source:
                scores = tl.where(valid2, gate, -float("inf"))
                maximum = tl.max(scores, axis=0)
                # Explicitly retain valid NaNs, independent of max's NaN policy.
                has_nan = tl.sum((scores != scores).to(tl.int32), axis=0) > 0
                maximum = tl.where(has_nan, float("nan"), maximum)
                exponent = tl.where(valid2, tl.exp(scores - maximum[None, :]), 0)
                probability = exponent / tl.sum(exponent, axis=0)[None, :]
                compressed = tl.sum(kv * probability, axis=0)
                d = tile * CD + tl.arange(0, CD)
                tl.store(SCRATCH + row * D + d, compressed, d < D)
        if tile == 0:
            tl.store(HAS + row, has_source.to(tl.int32))


@triton.jit
def _norm_rope(
    SCRATCH,
    HAS,
    POSITION,
    WEIGHT,
    CS,
    OUT,
    U: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    P0: tl.constexpr,
    W0: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    R: tl.constexpr,
    EPS: tl.constexpr,
    BD: tl.constexpr,
):
    for row in range(tl.program_id(0), U, tl.num_programs(0)):
        rr = row.to(tl.int64)
        d = tl.arange(0, BD)
        has = tl.load(HAS + rr) != 0
        z = tl.full((BD,), 0, tl.float32)
        if has:
            x = tl.load(SCRATCH + rr * D + d, d < D, other=0)
            inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
            weight = tl.load(WEIGHT + d.to(tl.int64) * W0, d < D, other=0).to(
                tl.float32
            )
            z = (x * inv) * weight
            if RD > 0:
                prefix: tl.constexpr = D - RD
                rotate = (d >= prefix) & (d < D)
                local = d - prefix
                partner = tl.where(rotate, prefix + (local ^ 1), d)
                other = tl.gather(z, partner, axis=0)
                pos = tl.load(POSITION + rr * P0).to(tl.int64)
                rope_pos = pos - (pos & (R - 1))
                freq = tl.where(rotate, local // 2, 0).to(tl.int64)
                cos = tl.load(CS + rope_pos * C0 + freq * C1, rotate, other=0).to(
                    tl.float32
                )
                sin = tl.load(
                    CS + rope_pos * C0 + (freq + RD // 2) * C1, rotate, other=0
                ).to(tl.float32)
                rotated = tl.where(
                    (local & 1) == 0, z * cos - other * sin, z * cos + other * sin
                )
                z = tl.where(rotate, rotated, z)
        tl.store(OUT + rr * D + d, z, d < D)


def _validate(
    hidden_states,
    wkv,
    wgate,
    ape,
    state_cache,
    positions,
    slot_mapping,
    state_block_table,
    output_request_ids,
    output_positions,
    rms_weight,
    cos_sin_cache,
    compress_ratio,
):
    if compress_ratio not in (4, 128):
        raise ValueError("Compression ratio must be 4 or 128")
    if hidden_states.ndim != 2 or wkv.ndim != 2 or wgate.shape != wkv.shape:
        raise ValueError(
            "Expected [tokens, hidden] and matching [width, hidden] weights"
        )
    n, h = hidden_states.shape
    d = rms_weight.numel()
    coff = 2 if compress_ratio == 4 else 1
    if d < 1 or rms_weight.ndim != 1 or h != wkv.shape[1] or wkv.shape[0] != coff * d:
        raise ValueError("Projection and compressor dimensions do not match")
    if (
        state_cache.ndim != 3
        or state_cache.dtype != torch.float32
        or state_cache.shape[-1] != 2 * coff * d
        or state_cache.shape[1] < 1
    ):
        raise ValueError("Compressor state must be FP32 [pages, block, packed width]")
    if ape.shape != (compress_ratio, coff * d):
        raise ValueError("APE shape must match compression ratio and projection width")
    if positions.shape != (n,) or slot_mapping.shape != (n,):
        raise ValueError("Expected separate input positions and state slots")
    if (
        state_block_table.ndim != 2
        or output_request_ids.ndim != 1
        or output_positions.shape != output_request_ids.shape
    ):
        raise ValueError("Expected output request/position vectors and a block table")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] % 2
        or cos_sin_cache.shape[1] > d
    ):
        raise ValueError("RoPE width must be even and no larger than the head")
    floats = (hidden_states, wkv, wgate, ape, rms_weight, cos_sin_cache)
    if any(
        x.dtype not in (torch.float16, torch.bfloat16, torch.float32) for x in floats
    ):
        raise TypeError("Compressor requires floating inputs and unquantized weights")
    metadata = (
        positions,
        slot_mapping,
        state_block_table,
        output_request_ids,
        output_positions,
    )
    if any(x.dtype not in (torch.int32, torch.int64) for x in metadata):
        raise TypeError("Compressor metadata must be INT32 or INT64")
    if any(x.device != hidden_states.device for x in (*floats, state_cache, *metadata)):
        raise ValueError("All compressor tensors must share a device")


def _compressor_impl(
    hidden_states,
    wkv,
    wgate,
    ape,
    state_cache,
    positions,
    slot_mapping,
    state_block_table,
    output_request_ids,
    output_positions,
    rms_weight,
    cos_sin_cache,
    compress_ratio=4,
    rms_eps=1e-6,
):
    _validate(
        hidden_states,
        wkv,
        wgate,
        ape,
        state_cache,
        positions,
        slot_mapping,
        state_block_table,
        output_request_ids,
        output_positions,
        rms_weight,
        cos_sin_cache,
        compress_ratio,
    )
    with torch.npu.device(hidden_states.device):
        n, h = hidden_states.shape
        d = rms_weight.numel()
        c = wkv.shape[0]
        u = output_positions.numel()
        output = torch.empty(
            (u, d), device=hidden_states.device, dtype=hidden_states.dtype
        )
        init_device_properties_triton()
        if n and state_cache.shape[0]:
            for weight, gate in ((wkv, False), (wgate, True)):
                cube = (
                    hidden_states.dtype == weight.dtype
                    and weight.dtype in (torch.bfloat16, torch.float16)
                    and hidden_states.stride(1) == weight.stride(1) == 1
                    and hidden_states.stride(0) >= h
                    and weight.stride(0) >= h
                    and (n - 1) * hidden_states.stride(0) + h < 2**31
                    and (c - 1) * weight.stride(0) + h < 2**31
                )
                common = (
                    hidden_states,
                    weight,
                    ape,
                    positions,
                    slot_mapping,
                    state_cache,
                    n,
                    h,
                    c,
                    state_cache.shape[0],
                    state_cache.shape[1],
                    *hidden_states.stride(),
                    *weight.stride(),
                    positions.stride(0),
                    slot_mapping.stride(0),
                    *ape.stride(),
                    *state_cache.stride(),
                    compress_ratio,
                    gate,
                )
                if cube:
                    work = triton.cdiv(n, 16) * triton.cdiv(c, 32)
                    _project_cube[(min(int(get_aicore_num()), work),)](
                        *common, 16, 32, 64, enable_fp_fusion=False
                    )
                else:
                    work = n * triton.cdiv(c, 32)
                    _project_vector[(min(int(get_vectorcore_num()), work),)](
                        *common, 32, 64, enable_fp_fusion=False
                    )
        if u:
            scratch = torch.empty(
                (u, d), device=hidden_states.device, dtype=torch.float32
            )
            has = torch.empty((u,), device=hidden_states.device, dtype=torch.int32)
            cores = int(get_vectorcore_num())
            # C128 keeps 128 historical rows live; bound its channel tile to
            # fit FP32 values plus INT64 gather metadata in the A2/A3 UB.
            channel_tile = 16 if compress_ratio == 128 else 32
            _history[(min(cores, u * triton.cdiv(d, channel_tile)),)](
                state_cache,
                state_block_table,
                output_request_ids,
                output_positions,
                scratch,
                has,
                u,
                d,
                state_cache.shape[0],
                state_cache.shape[1],
                *state_block_table.shape,
                cos_sin_cache.shape[0],
                *state_cache.stride(),
                *state_block_table.stride(),
                output_request_ids.stride(0),
                output_positions.stride(0),
                compress_ratio,
                2 if compress_ratio == 4 else 1,
                channel_tile,
                enable_fp_fusion=False,
            )
            _norm_rope[(min(cores, u),)](
                scratch,
                has,
                output_positions,
                rms_weight,
                cos_sin_cache,
                output,
                u,
                d,
                cos_sin_cache.shape[1],
                output_positions.stride(0),
                rms_weight.stride(0),
                *cos_sin_cache.stride(),
                compress_ratio,
                rms_eps,
                triton.next_power_of_2(d),
                enable_fp_fusion=False,
            )
        return output


_compressor_op = torch.library.custom_op(
    "flagos::deepseek_v4_compressor",
    _compressor_impl,
    mutates_args=("state_cache",),
    device_types="npu",
    schema="(Tensor hidden_states, Tensor wkv, Tensor wgate, Tensor ape, Tensor(a!) state_cache, Tensor positions, Tensor slot_mapping, Tensor state_block_table, Tensor output_request_ids, Tensor output_positions, Tensor rms_weight, Tensor cos_sin_cache, int compress_ratio=4, float rms_eps=1e-6) -> Tensor",
)


@_compressor_op.register_fake
def _compressor_fake(
    hidden_states,
    wkv,
    wgate,
    ape,
    state_cache,
    positions,
    slot_mapping,
    state_block_table,
    output_request_ids,
    output_positions,
    rms_weight,
    cos_sin_cache,
    compress_ratio=4,
    rms_eps=1e-6,
):
    return torch.empty(
        (output_positions.numel(), rms_weight.numel()),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )


def deepseek_v4_compressor(
    hidden_states,
    wkv,
    wgate,
    ape,
    state_cache,
    positions,
    slot_mapping,
    state_block_table,
    output_request_ids,
    output_positions,
    rms_weight,
    cos_sin_cache,
    compress_ratio=4,
    rms_eps=1e-6,
):
    return _compressor_op(
        hidden_states,
        wkv,
        wgate,
        ape,
        state_cache,
        positions,
        slot_mapping,
        state_block_table,
        output_request_ids,
        output_positions,
        rms_weight,
        cos_sin_cache,
        compress_ratio,
        rms_eps,
    )
