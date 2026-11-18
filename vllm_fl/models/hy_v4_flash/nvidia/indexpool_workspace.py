# SPDX-License-Identifier: Apache-2.0
"""Bounds for Flash IndexPool's sequential forward temporaries."""

MAX_SELECTION_ROWS = 8192


def scoring_chunk_rows(logits_elems: int, width: int) -> int:
    if width <= 0 or logits_elems < width:
        raise ValueError("IndexPool logits budget cannot hold a single scoring row")
    return min(MAX_SELECTION_ROWS, logits_elems // width)


def profile_workspace_bytes(
    *,
    tokens: int,
    batch: int,
    ratio: int,
    tail_ring: int,
    heads: int,
    budget: int,
    width: int,
    page_size: int,
    logits_elems: int,
    num_sms: int,
) -> int:
    """Conservative transient peak, excluding already-budgeted persistent caches."""
    rows = min(tokens, scoring_chunk_rows(logits_elems, width))
    groups = (tokens + batch * (ratio - 1)) // ratio

    # Plan prefix, positions, source/destination maps and live metadata.
    plan = batch * 4 + groups * (32 + 16 * ratio) + batch * tail_ring * 16
    token_metadata = tokens * 21
    # BF16 key projection + norm, BF16 raw and FP32 projection input,
    # both FP32 projected tensors, and pooled/rotated/quantized rows.
    pool = tokens * 128 * 8 + groups * 128 * (14 * ratio + 16)

    # Padded FP8 Q, rotary/projection conversions, FP32 scales/head weights.
    padded_heads = 32 if heads <= 32 else 64
    query = tokens * (heads * 128 * 16 + padded_heads * 136)
    # Selector's wide path uses at most two 256-bin arrays per radix part.
    # Small row chunks may choose up to eight parts (8192 int32 words after
    # rounding); chunks of at least 1024 rows use two (2048 words).
    radix = 0
    if width > budget:
        wide_large = rows * 2048 if rows >= 1024 else 0
        wide_small = min(rows, 1023) * 8192 if width > 8192 else 0
        radix = 4 * max(wide_large, wide_small)
    block_columns = (width + page_size - 1) // page_size
    # Paged block table (plus a possible contiguous clone), one logits
    # buffer, selected and expanded indices, and small schedule temporaries.
    logits_rows = (rows + 63) // 64 * 64
    scoring = 4 * logits_rows * width + rows * (8 * block_columns + 4 * budget)
    scoring += rows * (4 * (budget + 1) * ratio + 32) + (num_sms + 1) * 32
    return token_metadata + max(plan + pool, query + scoring + radix)


def profile_peak_bytes(*, max_width: int, page_size: int, **kwargs) -> int:
    """Smaller eager widths may fit more rows than the maximum Graph width."""
    if max_width < page_size or max_width % page_size:
        raise ValueError("Invalid IndexPool scoring width")
    return max(
        profile_workspace_bytes(width=width, page_size=page_size, **kwargs)
        for width in range(page_size, max_width + 1, page_size)
    )
