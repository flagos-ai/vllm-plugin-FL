# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from vLLM v0.32.0:
# https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/worker/gpu/sample/gumbel.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Preserve Murmur3 logical right shifts on Ascend's integer lowering.

The masks remove sign extension if the backend lowers an unsigned shift as a
signed shift. Multiplication, addition, seed/position mixing and Gumbel noise
remain the vLLM 0.32 algorithm, including the full 32-bit winning-tail entropy.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def _murmur3_rotl32(value, shift: tl.constexpr):
    return (value << shift) | ((value >> (32 - shift)) & ((1 << shift) - 1))


@triton.jit
def _murmur3_mix(h, key):
    key *= 0xCC9E2D51
    key = _murmur3_rotl32(key, 15)
    key *= 0x1B873593
    h ^= key
    h = _murmur3_rotl32(h, 13)
    return h * 5 + 0xE6546B64


@triton.jit
def _murmur3_fmix32(h):
    h ^= (h >> 16) & 0xFFFF
    h *= 0x85EBCA6B
    h ^= (h >> 13) & 0x7FFFF
    h *= 0xC2B2AE35
    return h ^ ((h >> 16) & 0xFFFF)


@triton.jit
def murmur3_hash32(seed, pos, offset, domain: tl.constexpr = 0):
    seed = seed.to(tl.int64)
    pos = pos.to(tl.int64)
    offset = offset.to(tl.uint32)
    # Keep the request-wide prefix scalar until the token offset is mixed in.
    h = (seed ^ seed).to(tl.uint32)
    h ^= domain
    h = _murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, (pos & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, offset)
    return _murmur3_fmix32(h ^ 16)


@triton.jit
def murmur3_uniform32(seed, pos, offset):
    random32 = murmur3_hash32(seed, pos, offset)
    # Split the uint32 before converting to fp32 so backends without a native
    # uint32-to-float conversion can still use all 32 source bits. Both 16-bit
    # halves convert exactly; their sum is the correctly rounded fp32 value of
    # (random32 + 0.5) * 2**-32. In particular, the u -> 0 winning tail keeps
    # the full 32-bit source resolution instead of being truncated to 24 bits.
    hi16 = ((random32 >> 16) & 0xFFFF).to(tl.int32)
    lo16 = (random32 & 0xFFFF).to(tl.int32)
    return (
        hi16.to(tl.float32) * 1.52587890625e-05
        + (lo16.to(tl.float32) + 0.5) * 2.3283064365386963e-10
    )


@triton.jit
def _uniform64_from_random53(random53):
    uniform = (random53.to(tl.float64) + 0.5) * 1.1102230246251565e-16
    # The largest midpoint rounds to 1.0 in fp64; keep the uniform open without
    # relying on a near-one literal that Triton may materialize in fp32.
    return tl.where(uniform == 1.0, uniform - 1.1102230246251565e-16, uniform)


@triton.jit
def murmur3_uniform64(seed, pos, offset):
    lo = murmur3_hash32(seed, pos, offset).to(tl.uint64)
    hi = murmur3_hash32(seed, pos, offset, domain=0x9E3779B9).to(tl.uint64)
    random53 = (((hi << 32) | lo) >> 11) & 0x1FFFFFFFFFFFFF
    return _uniform64_from_random53(random53)
