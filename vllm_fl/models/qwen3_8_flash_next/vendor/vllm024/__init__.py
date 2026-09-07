# SPDX-License-Identifier: Apache-2.0
"""Adapters that make the vendored math obey the vLLM 0.24 ABI."""

from .dispatch import (
    qwen4_hc_backend,
    qwen4_qsa_pre_indexer_status,
    use_official_hc,
)

__all__ = [
    "qwen4_hc_backend",
    "qwen4_qsa_pre_indexer_status",
    "use_official_hc",
]
