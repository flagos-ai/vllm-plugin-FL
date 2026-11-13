# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Translate selected QSA token indices into native virtual-page metadata."""

import triton as tr
import triton.language as tl
from flag_gems import cumsum_out


@tr.jit
def selected_table(
    IDX,
    TAB,
    MAP,
    OUT,
    LENS,
    CU,
    SI: tl.constexpr,
    ST: tl.constexpr,
    PAGE: tl.constexpr,
    STEP: tl.constexpr,
    NP: tl.constexpr,
):
    r = tl.program_id(0)
    if r == 0:
        tl.store(CU, 0)
    n = tl.arange(0, 1024)
    raw = tl.load(IDX + r * SI + n * 4, n <= 512, -1)
    req = tl.load(MAP + r)
    page = tl.maximum(raw, 0) // PAGE
    off = tl.maximum(raw, 0) % PAGE
    phys = tl.load(TAB + req * ST + page, (n <= 512) & (raw >= 0) & (page < ST), -1)
    valid = (n <= 512) & (raw >= 0) & (phys >= 0) & (phys < NP)
    # The fused producer compacts a partial tail immediately after valid full groups.
    offsets = n[:, None] * 4 + tl.arange(0, 4)[None, :]
    group_tokens = tl.load(IDX + r * SI + offsets, offsets < 2051, -1)
    count = tl.sum((group_tokens >= 0).to(tl.int32), 1)
    valid &= count > 0
    tl.store(OUT + r * 513 + n, 0, n < 513)
    tl.debug_barrier()
    dst = tl.cumsum(valid.to(tl.int32), 0) - 1
    virtual_page = phys * STEP + off // 4
    tl.store(OUT + r * 513 + dst, virtual_page, valid)
    tl.store(LENS + r, tl.sum(tl.where(valid, count, 0), 0))


def prepare_native_qsa_metadata(indices, table, mapping, k, workspace):
    m = indices.shape[0]
    selected_table[(m,)](
        indices,
        table,
        mapping,
        workspace["table"],
        workspace["lengths"],
        workspace["cu"],
        indices.stride(0),
        table.stride(0),
        k.shape[1],
        k.stride(0) // 1024,
        k.shape[0],
        num_warps=4,
    )
    cumsum_out(
        workspace["lengths"], 0, dtype=workspace["cu"].dtype, out=workspace["cu"][1:]
    )
