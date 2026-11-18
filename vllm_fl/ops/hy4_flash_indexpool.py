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


_write_pool = CachedOp("hy4_flash_write_pool")


def write_pool(*args, **kwargs):
    return _write_pool(*args, **kwargs)


_gather_groups = CachedOp("hy4_flash_gather_groups")


def gather_groups(*args, **kwargs):
    return _gather_groups(*args, **kwargs)


_write_tails = CachedOp("hy4_flash_write_tails")


def write_tails(*args, **kwargs):
    return _write_tails(*args, **kwargs)
