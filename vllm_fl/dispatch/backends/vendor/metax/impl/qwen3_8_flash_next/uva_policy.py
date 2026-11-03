# SPDX-License-Identifier: Apache-2.0
"""Measured MetaX PLE lookup strategy and row-gather launch settings."""

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def lookup_settings():
    path = (
        Path(__file__).resolve().parents[5]
        / "config/qwen3_8_flash_next_metax_tuning.json"
    )
    settings = json.loads(path.read_text())["ngram_uva"]
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
