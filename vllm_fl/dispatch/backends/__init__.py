# Copyright (c) 2026 BAAI. All rights reserved.

"""
Backend implementations for vllm-plugin-FL dispatch.
"""

from .base import Backend
from .flaggems import FlagGemsBackend
from .reference import ReferenceBackend

__all__ = ["Backend", "FlagGemsBackend", "ReferenceBackend"]

# Try to import vendor backends
try:
    from .vendor.ascend import AscendBackend
    __all__.append("AscendBackend")
except ImportError:
    AscendBackend = None

# Add more vendor backends here as they become available
try:
    from .vendor.cuda import CudaBackend
    __all__.append("CudaBackend")
except ImportError:
    CudaBackend = None
