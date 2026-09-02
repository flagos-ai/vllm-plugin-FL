# Copyright (c) 2026 BAAI. All rights reserved.

"""
Txda (tsingmicro) backend for vllm-plugin-FL dispatch.

This backend provides operator implementations for Tsingmicro TX devices.
"""

from .txda import TxdaBackend

__all__ = ["TxdaBackend"]
