# Copyright (c) 2026 BAAI. All rights reserved.

"""TXDA RMSNorm fallback built on the shared reference contract."""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.reference.impl.normalization import rms_norm_torch


def rms_norm_txda(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """RMS normalization for TXDA.

    The FlagGems Triton kernel exceeds the TXDA shared-memory limit, so the
    reference implementation is used instead. Delegating keeps the upstream
    numerical contract: residual accumulation, the variance reduction and the
    output all follow ``rms_norm_torch`` (FP32 accumulation, honor
    ``variance_size_override``, output in the input dtype), which a local
    reimplementation would have to replicate exactly.
    """
    return rms_norm_torch(obj, x, residual)
