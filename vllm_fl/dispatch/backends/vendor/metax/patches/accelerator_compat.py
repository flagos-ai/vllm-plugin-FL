# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# --------------------------------------------------------------------------------
# Hotfix: torch 2.8+metax is missing several torch.accelerator memory APIs that
# were added in PyTorch 2.9+. Patch them to the equivalent torch.cuda calls.
#
# Guard: only applied when torch version is < 2.9 AND the API is actually absent.
# - Version check avoids silently overriding APIs that exist (and may have changed
#   signatures) in torch >= 2.9.
# - hasattr check is the authoritative guard: if the API exists we never touch it,
#   regardless of version.
#
# Reference: https://github.com/MetaX-MACA/vLLM-metax/blob/releases/v0.21.0/vllm_metax/patch/torch_fix/fix_standalone_compile.py
# TODO: remove when MetaX ships torch >= 2.9 with full accelerator API support.
# --------------------------------------------------------------------------------

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def _torch_version_tuple() -> Tuple[int, ...]:
    """Return (major, minor) of torch version, ignoring vendor suffixes like '+metax'."""
    ver = torch.__version__.split("+")[0]  # strip "+metax", "+cu124", etc.
    parts = ver.split(".")[:2]
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return (0, 0)


_torch_ver = _torch_version_tuple()

# Only apply this patch on torch < 2.9. On torch >= 2.9, these APIs should exist
# natively; patching them would risk silently overriding a potentially changed
# signature or implementation.
if _torch_ver < (2, 9):
    _MISSING_APIS = {
        "empty_cache": torch.cuda.empty_cache,
        "memory_stats": torch.cuda.memory_stats,
        "memory_reserved": torch.cuda.memory_reserved,
        "memory_allocated": torch.cuda.memory_allocated,
        "reset_peak_memory_stats": torch.cuda.reset_peak_memory_stats,
        "max_memory_allocated": torch.cuda.max_memory_allocated,
    }

    for _name, _impl in _MISSING_APIS.items():
        if not hasattr(torch.accelerator, _name):
            setattr(torch.accelerator, _name, _impl)
            logger.debug(
                "accelerator_compat: patched torch.accelerator.%s -> torch.cuda.%s "
                "(torch %s lacks this API)",
                _name, _name, torch.__version__,
            )
else:
    # Sanity check: warn if any expected API is still missing on torch >= 2.9,
    # which would indicate an unexpected regression.
    _EXPECTED_APIS = [
        "empty_cache", "memory_stats", "memory_reserved",
        "memory_allocated", "reset_peak_memory_stats", "max_memory_allocated",
    ]
    for _name in _EXPECTED_APIS:
        if not hasattr(torch.accelerator, _name):
            logger.warning(
                "accelerator_compat: torch.accelerator.%s is missing on torch %s "
                "(expected it to exist on >= 2.9). "
                "Please update this patch file.",
                _name, torch.__version__,
            )
