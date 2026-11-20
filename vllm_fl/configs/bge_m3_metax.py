# Copyright (c) 2026 BAAI. All rights reserved.
"""Measured BGE-M3 FP32 Linear tuning region on MetaX; M is total rows."""

LINEAR_MAX_ROWS = 1024
LINEAR_FEATURES = frozenset({(3072, 1024), (1024, 1024), (4096, 1024), (1024, 4096)})
LINEAR_EXTRA_CONFIGS = (
    {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "num_warps": 8,
        "num_stages": 4,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "num_warps": 8,
        "num_stages": 4,
    },
)
