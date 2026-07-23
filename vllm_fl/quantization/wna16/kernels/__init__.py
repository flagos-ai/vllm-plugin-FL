# Copyright (c) 2026 BAAI. All rights reserved.
"""Fixed kernel entry points used by the WNA16 quantization adapters.

Implementations live in this package and are called directly. They are not
registered with vllm-fl's general operator dispatch.
"""

from .gemm import is_wna16_gemm_available, wna16_gemm
from .moe import is_wna16_moe_available, wna16_moe

__all__ = [
    "is_wna16_gemm_available",
    "is_wna16_moe_available",
    "wna16_gemm",
    "wna16_moe",
]
