# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from vLLM-Ascend v0.20.2rc1.
# SPDX-License-Identifier: Apache-2.0

"""Ascend attention backend selection helpers.

This module deliberately reads FL's ``VllmConfig.additional_config`` instead
of importing vLLM-Ascend's process-global ``AscendConfig``.
"""

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from vllm.config import CUDAGraphMode, VllmConfig


@lru_cache(maxsize=1)
def _is_ascend_a5() -> bool:
    """Detect A5 through FL's local device-family helper when available."""
    try:
        from vllm_fl.cpu_binding import (
            AscendDeviceType,
            get_ascend_device_type,
        )

        return get_ascend_device_type() == AscendDeviceType.A5
    except (ImportError, RuntimeError):
        return False


def using_paged_attention(
    runtime_shape: int,
    vllm_config: VllmConfig | Any,
) -> bool:
    """Return whether a full-decode graph shape explicitly selects PA.

    The target vLLM-Ascend release defaults an absent or empty
    ``pa_shape_list`` to FIA. Paged attention is limited to explicitly listed
    shapes in ``FULL_DECODE_ONLY`` mode and is disabled for speculative decode.
    """
    if getattr(vllm_config, "speculative_config", None) is not None:
        return False
    if _is_ascend_a5():
        return False

    compilation_config = getattr(vllm_config, "compilation_config", None)
    if (
        compilation_config is None
        or compilation_config.cudagraph_mode
        != CUDAGraphMode.FULL_DECODE_ONLY
    ):
        return False

    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if not isinstance(additional_config, Mapping):
        return False
    pa_shape_list = additional_config.get("pa_shape_list", []) or []
    return runtime_shape in pa_shape_list
