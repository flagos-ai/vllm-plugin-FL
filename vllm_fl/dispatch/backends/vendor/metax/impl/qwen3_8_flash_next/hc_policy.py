# SPDX-License-Identifier: Apache-2.0
"""MetaX HC small-op selection with unchanged model projections and rounding."""

import json
from functools import lru_cache
from pathlib import Path

import torch

from .shape_limits import max_prefill_rows


@lru_cache(maxsize=1)
def _settings():
    path = (
        Path(__file__).resolve().parents[5]
        / "config/qwen3_8_flash_next_metax_tuning.json"
    )
    return json.loads(path.read_text())["hc_small_ops"]


def _eligible(x, groups):
    return (
        x.device.type == "cuda"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and 1 <= x.shape[0] <= max_prefill_rows()
        and x.shape[1] == 10240
        and groups == 4
        and x.is_contiguous()
    )


def grouped_norm(x, weight, groups, eps):
    if not (
        _eligible(x, groups)
        and weight.shape == (10240,)
        and weight.dtype == x.dtype
        and weight.device == x.device
        and weight.is_contiguous()
    ):
        return None
    from flaggems_vllm.runtime.backend._metax.ops.hyperconnection_bf16 import (
        grouped_norm,
    )

    return grouped_norm(x, weight, groups, eps, **_settings()["norm"])


@lru_cache(maxsize=64)
def _row_settings(name, rows):
    return next(
        {k: v for k, v in entry.items() if k != "max_rows"}
        for entry in _settings()[name]
        if rows <= entry["max_rows"]
    )


def gate_reduce(logits, normed, groups):
    if not (
        _eligible(normed, groups)
        and logits.shape == normed.shape
        and logits.dtype == normed.dtype
        and logits.device == normed.device
        and logits.is_contiguous()
    ):
        return None
    from flaggems_vllm.runtime.backend._metax.ops.hyperconnection_bf16 import (
        gate_reduce,
    )

    return gate_reduce(logits, normed, groups, **_row_settings("gate", normed.shape[0]))


def inject_combine(logits, block_output, residual, groups):
    if not (
        _eligible(residual, groups)
        and logits.shape == (residual.shape[0], groups)
        and block_output.shape == (residual.shape[0], 2560)
        and all(
            x.dtype == residual.dtype
            and x.device == residual.device
            and x.is_contiguous()
            for x in (logits, block_output)
        )
    ):
        return None
    from flaggems_vllm.runtime.backend._metax.ops.hyperconnection_bf16 import inject

    return inject(
        logits,
        block_output,
        residual,
        groups,
        **_row_settings("inject", residual.shape[0]),
    )


def down_activation(x, groups):
    if not (
        x.device.type == "cuda"
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[1] == 320
        and 1 <= x.shape[0] <= max_prefill_rows()
        and groups == 4
        and x.is_contiguous()
    ):
        return None
    from flaggems_vllm.runtime.backend._metax.ops.hyperconnection_bf16 import (
        divide_silu_bf16,
    )

    return divide_silu_bf16(x, groups, **_settings()["down_activation"])
