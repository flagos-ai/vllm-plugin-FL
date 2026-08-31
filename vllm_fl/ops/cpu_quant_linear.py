"""Shared lifecycle for ARM CPU quantized-linear backends."""

from __future__ import annotations

import logging
import pathlib
import platform
from collections.abc import Callable
from threading import Lock
from typing import Any


_INSTALL_LOCK = Lock()
_ACTIVE_BACKEND: str | None = None


def require_arm_quant_extensions(backend: str) -> None:
    """Require the AArch64 extensions used by the packaged native kernels."""
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        raise RuntimeError(f"{backend} requires AArch64")
    cpuinfo = pathlib.Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return
    feature_sets = [
        set(line.partition(":")[2].split())
        for line in cpuinfo.read_text(encoding="utf-8").splitlines()
        if line.lower().startswith("features")
    ]
    features = set.intersection(*feature_sets) if feature_sets else set()
    required = {"asimddp", "i8mm", "bf16"}
    missing = required - features
    if missing:
        raise RuntimeError(
            f"{backend} requires dotprod, i8mm, and BF16 CPU extensions; "
            f"missing: {', '.join(sorted(missing))}"
        )


def install_cpu_quantized_linear(
    *,
    backend: str,
    prepare_linear: Callable[[Any], Callable[..., Any]],
    supports_shape: Callable[[int, int], bool],
    include_lm_head: bool,
    strict: bool,
    logger: logging.Logger,
    initialize: Callable[[], None] | None = None,
) -> bool:
    """Install one process-wide CPU quantized-linear dispatch implementation.

    Backend adapters own packing and compute. This module owns the shared layer
    eligibility, fallback, weight-removal, and process-lifetime invariants.
    """
    import torch
    import vllm.model_executor.layers.utils as layer_utils
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    global _ACTIVE_BACKEND
    with _INSTALL_LOCK:
        if _ACTIVE_BACKEND == backend:
            return False
        if _ACTIVE_BACKEND is not None:
            raise RuntimeError(
                "ARM CPU quantized-linear backend is already active as "
                f"{_ACTIVE_BACKEND}; cannot activate {backend} in the same process. "
                "Select one backend and restart."
            )

        if initialize is not None:
            initialize()
        original_dispatch = layer_utils.dispatch_cpu_unquantized_gemm

        def dispatch(layer, remove_weight):
            weight = getattr(layer, "weight", None)
            prefix = getattr(layer, "prefix", "") or ""
            is_lm_head = isinstance(layer, ParallelLMHead)
            is_input_embedding = (
                isinstance(layer, VocabParallelEmbedding) and not is_lm_head
            )
            if (
                weight is not None
                and weight.ndim == 2
                and supports_shape(int(weight.shape[0]), int(weight.shape[1]))
                and not is_input_embedding
                and (include_lm_head or not is_lm_head)
            ):
                try:
                    layer.cpu_linear = prepare_linear(weight)
                    if remove_weight:
                        layer.weight = torch.nn.Parameter(
                            torch.empty(0), requires_grad=False
                        )
                    return
                except Exception as exc:
                    message = (
                        f"failed to prepare {backend} weight {prefix} "
                        f"{tuple(weight.shape)}"
                    )
                    if strict:
                        raise RuntimeError(message) from exc
                    logger.warning("%s; falling back to BF16: %s", message, exc)
            return original_dispatch(layer, remove_weight)

        layer_utils.dispatch_cpu_unquantized_gemm = dispatch
        _ACTIVE_BACKEND = backend
        return True
