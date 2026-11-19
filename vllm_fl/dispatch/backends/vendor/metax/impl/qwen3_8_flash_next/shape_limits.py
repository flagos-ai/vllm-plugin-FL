# SPDX-License-Identifier: Apache-2.0
"""Shared access to the validated Qwen MetaX tuning configuration."""

import json
from functools import lru_cache
from pathlib import Path


def load_tuning_config():
    """Return a fresh mapping; each policy owns its existing cache."""
    path = (
        Path(__file__).resolve().parents[5]
        / "config/qwen3_8_flash_next_metax_tuning.json"
    )
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def max_prefill_rows():
    value = load_tuning_config().get("prefill_max_rows", 2048)
    if type(value) is not int or value not in (2048, 4096, 8192):
        raise ValueError("Unsupported Qwen MetaX prefill row limit")
    return value
