# SPDX-License-Identifier: Apache-2.0
"""Hy4 Flash IndexPool calls through the shared backend policy."""

from vllm_fl.dispatch import CachedOp

_fwht = CachedOp("hy4_flash_fwht_quant_fp8")
_pool = CachedOp("hy4_flash_gated_pool")


def fwht_quantize(vectors):
    return _fwht(vectors)


def gated_pool(values, gates, ape, norm_weight, eps):
    return _pool(values, gates, ape, norm_weight, eps)


_rotate_indexer = CachedOp("hy4_flash_rotate_indexer")


def rotate_indexer(*args, **kwargs):
    return _rotate_indexer(*args, **kwargs)


_select_groups = CachedOp("hy4_flash_select_groups")


def select_groups(*args, **kwargs):
    return _select_groups(*args, **kwargs)


_expand_groups = CachedOp("hy4_flash_expand_groups")


def expand_groups(*args, **kwargs):
    return _expand_groups(*args, **kwargs)


_write_pool = CachedOp("hy4_flash_write_pool")


def write_pool(*args, **kwargs):
    return _write_pool(*args, **kwargs)


# Immutable reference constants are initialized before inference and reused.
_rotary_coefficients = CachedOp("hy4_flash_rotary_coefficients")


def rotary_coefficients(*args, **kwargs):
    return _rotary_coefficients(*args, **kwargs)
