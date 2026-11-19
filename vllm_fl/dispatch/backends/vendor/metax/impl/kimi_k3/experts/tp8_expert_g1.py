# SPDX-License-Identifier: Apache-2.0
"""TP8 small-batch gate/up projection fused with SiTU.

Packed FP4/E8M0 storage and the intermediate BF16 rounding are unchanged.
Each output tile pairs 32 gate columns with their matching 32 up columns.
"""

import triton
import triton.language as tl
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.expert_gemm import _decode_words_math
from vllm_fl.dispatch.backends.vendor.metax.impl.kimi_k3.experts.marlin_mxfp4 import _concat_k
from flag_gems.utils import tl_extra_shim

_tanh = tl_extra_shim.tanh
_exp2 = tl_extra_shim.exp2


@triton.jit
def _tp8_g1_situ(
    X, W, S, Activated, Sorted, Experts, Total, R: tl.constexpr, NORMAL: tl.constexpr
):
    task = tl.program_id(0)
    row_block, column_block = task // 12, task % 12
    if row_block * 16 >= tl.load(Total):
        return
    route = tl.load(Sorted + row_block * 16 + tl.arange(0, 16)).to(tl.int32)
    expert = tl.load(Experts + row_block).to(tl.int32)
    if (expert < 0) | (expert >= 896):
        return
    valid = route < R
    index = tl.arange(0, 64)
    column = column_block * 32 + index % 32 + index // 32 * 384
    word_k = tl.arange(0, 16)
    k = tl.arange(0, 128)
    weight_ptr = W + (expert * 448 + word_k[:, None]) * 768 + column[None, :]
    scale_ptr = S + expert * 112 * 768 + column
    acc = tl.zeros((16, 64), tl.float32)
    for block in range(28):
        a = tl.load(
            X + route[:, None] // 16 * 3584 + block * 128 + k[None, :],
            valid[:, None],
            0,
        )
        word = tl.load(weight_ptr).to(tl.uint32)
        base = scale_ptr + block * 4 * 768
        s0 = tl.load(base).to(tl.int32)
        s1 = tl.load(base + 768).to(tl.int32)
        s2 = tl.load(base + 1536).to(tl.int32)
        s3 = tl.load(base + 2304).to(tl.int32)
        parts = _decode_words_math(
            word, s0[None, :], s1[None, :], s2[None, :], s3[None, :], 128, NORMAL
        )
        b = _concat_k(parts, 64, 128)
        acc = tl.dot(a, b, acc)
        weight_ptr += 16 * 768
    # Preserve G1's BF16 materialization before all FP32 SiTU arithmetic.
    rounded = tl.reshape(acc.to(tl.bfloat16), (16, 2, 32))
    gate, up = tl.split(tl.permute(rounded, (0, 2, 1)))
    gate, up = gate.to(tl.float32), up.to(tl.float32)
    sigmoid = 1.0 / (1.0 + _exp2(-gate * 1.4426950408889634))
    gate = 4.0 * _tanh(gate / 4.0) * sigmoid
    up = 25.0 * _tanh(up / 25.0)
    value = tl.fma(gate, up, 0.0).to(tl.bfloat16)
    column = column_block * 32 + tl.arange(0, 32)
    tl.store(Activated + route[:, None] * 384 + column[None, :], value, valid[:, None])
