# Copyright (c) 2026 BAAI. All rights reserved.

"""
Txda (Tsingmicro) backend for vllm-plugin-FL dispatch.
"""

from .txda import TxdaBackend

__all__ = ["TxdaBackend"]

from .impl import patchs  # noqa: F401 — apply Txda patches at backend load time
