# SPDX-License-Identifier: Apache-2.0
"""Validated row range for the Qwen MetaX prefill operator policies."""

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def max_prefill_rows():
    path = Path(__file__).resolve().parents[5] / "config/qwen3_8_flash_next_metax_tuning.json"
    value = json.loads(path.read_text()).get("prefill_max_rows", 2048)
    if type(value) is not int or value not in (2048, 4096, 8192):
        raise ValueError("Unsupported Qwen MetaX prefill row limit")
    return value
