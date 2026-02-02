# Copyright (c) 2026 BAAI. All rights reserved.

"""
Vendor backends for vllm-plugin-FL dispatch.

This package contains vendor-specific backend implementations.

Available vendor backends:
- ascend: Huawei Ascend NPU backend

To add a new vendor backend:
1. Create a subdirectory: vendor/<vendor_name>/
2. Implement the backend class inheriting from Backend
3. Create register_ops.py with registration function
4. The backend will be auto-discovered by builtin_ops.py

See the "Adding Vendor Backends" section in dispatch/README.md for detailed instructions.
"""

__all__ = []

# Import Ascend backend
try:
    from .ascend import AscendBackend
    __all__.append("AscendBackend")
except ImportError:
    pass

# Import CUDA backend
try:
    from .cuda import CudaBackend
    __all__.append("CudaBackend")
except ImportError:
    pass

# Add more vendor backends here as they become available:
# try:
#     from .rocm import RocmBackend
#     __all__.append("RocmBackend")
# except ImportError:
#     pass
