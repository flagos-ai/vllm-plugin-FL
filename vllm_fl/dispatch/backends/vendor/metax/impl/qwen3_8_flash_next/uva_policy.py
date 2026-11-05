# SPDX-License-Identifier: Apache-2.0
"""Measured MetaX PLE lookup strategy and row-gather launch settings."""

from functools import lru_cache

from .shape_limits import load_tuning_config


@lru_cache(maxsize=1)
def lookup_settings():
    settings = load_tuning_config()["ngram_uva"]
    if (
        type(settings["broadcast_max_tokens"]) is not int
        or settings["broadcast_max_tokens"] < 0
        or any(
            settings[key] not in (1, 4, 8, 16)
            for key in ("broadcast_block_rows", "sharded_block_rows")
        )
        or settings["num_warps"] not in (2, 4, 8)
    ):
        raise ValueError("Invalid MetaX UVA lookup settings")
    return settings
