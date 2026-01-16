# Copyright (c) 2026 BAAI. All rights reserved.

"""
Backend implementations for vllm-plugin-FL dispatch.
"""

from .base import Backend
from .flaggems import FlagGemsBackend
from .reference import ReferenceBackend

__all__ = ["Backend", "FlagGemsBackend", "ReferenceBackend"]

# Add vendor backends here as they become available
# try:
#     from .vendor.my_vendor import MyVendorBackend
#     __all__.append("MyVendorBackend")
# except ImportError:
#     MyVendorBackend = None
