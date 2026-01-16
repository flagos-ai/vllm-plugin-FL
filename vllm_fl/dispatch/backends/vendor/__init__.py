# Copyright (c) 2026 BAAI. All rights reserved.

"""
Vendor backends for vllm-plugin-FL dispatch.

This package contains vendor-specific backend implementations.

To add a new vendor backend:
1. Create a subdirectory: vendor/<vendor_name>/
2. Implement the backend class inheriting from Backend
3. Create register_ops.py with registration function
4. The backend will be auto-discovered by builtin_ops.py

See the template/ directory for a starting point and detailed instructions.
"""

__all__ = []

# Add vendor backends here as they become available:
# try:
#     from .my_vendor import MyVendorBackend
#     __all__.append("MyVendorBackend")
# except ImportError:
#     pass
