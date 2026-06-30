# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd.

"""Compatibility patches for PyTorch functorch config keys.

vLLM 0.20.2 patches ``torch._functorch.config.autograd_cache_normalize_inputs``
during compilation. Some Metax PyTorch builds based on 2.8 do not define that
config key yet, so PyTorch's config module raises AttributeError before the
model can finish initialization.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_AUTOGRAD_CACHE_NORMALIZE_INPUTS = "autograd_cache_normalize_inputs"


def _make_torch_config_entry(name: str, default: bool) -> Any | None:
    try:
        from torch.utils import _config_module as torch_config_module
    except Exception:
        logger.debug("Unable to import torch config module.", exc_info=True)
        return None

    config_factory = getattr(torch_config_module, "Config", None)
    config_entry_cls = getattr(torch_config_module, "_ConfigEntry", None)
    if config_factory is None or config_entry_cls is None:
        return None

    config = config_factory(default=default, value_type=bool)

    try:
        return config_entry_cls(config, name)
    except TypeError:
        return config_entry_cls(config)


def _mark_config_dirty(config_module: Any, name: str) -> None:
    if hasattr(config_module, "_is_dirty"):
        config_module._is_dirty = True

    hash_dirty_var = getattr(config_module, "_hash_dirty_var", None)
    if hash_dirty_var is not None:
        hash_dirty_var.set(True)

    mark_dirty = getattr(config_module, "_mark_get_dict_dirty", None)
    if mark_dirty is not None:
        mark_dirty(name)


def patch_functorch_config() -> None:
    try:
        import torch._functorch.config as functorch_config
    except Exception:
        logger.debug("Unable to import torch._functorch.config.", exc_info=True)
        return

    config_map = getattr(functorch_config, "_config", None)
    if not isinstance(config_map, dict):
        return

    if _AUTOGRAD_CACHE_NORMALIZE_INPUTS in config_map:
        return

    entry = _make_torch_config_entry(
        _AUTOGRAD_CACHE_NORMALIZE_INPUTS,
        default=True,
    )
    if entry is None:
        logger.warning(
            "Unable to register PyTorch functorch config key '%s'.",
            _AUTOGRAD_CACHE_NORMALIZE_INPUTS,
        )
        return

    config_map[_AUTOGRAD_CACHE_NORMALIZE_INPUTS] = entry
    _mark_config_dirty(functorch_config, _AUTOGRAD_CACHE_NORMALIZE_INPUTS)


patch_functorch_config()
