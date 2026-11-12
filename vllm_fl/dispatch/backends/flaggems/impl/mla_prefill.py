# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.

from importlib import import_module

from vllm.platforms import current_platform

import vllm_fl.envs as fl_envs


def _provider():
    library = "flaggems_vllm" if fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM else "flag_gems"
    return getattr(import_module(library), "flash_attn_varlen_func")


def is_available():
    if (
        getattr(current_platform, "vendor_name", None) != "thead"
        or current_platform.device_type != "cuda"
    ):
        return False
    try:
        operator = _provider()
    except (ImportError, AttributeError, OSError):
        return False
    return callable(operator)


def mla_prefill_flaggems(**kwargs):
    return _provider()(**kwargs)
