# Copyright (c) 2026 BAAI. All rights reserved.

"""
Txda (Tsingmicro) backend for vllm-plugin-FL dispatch.
"""

from .txda import TxdaBackend

__all__ = ["TxdaBackend"]
