# Copyright (c) 2026 BAAI. All rights reserved.

"""
Vendor backends for vllm-plugin-FL dispatch.

This package contains vendor-specific backend implementations.

Available vendor backends:
- ascend: Huawei Ascend NPU backend

This package intentionally avoids eager imports of vendor subpackages.
Importing a specific backend such as ``vllm_fl.dispatch.backends.vendor.ascend``
should not pull in other vendor branches.

To add a new vendor backend:
1. Create a subdirectory: vendor/<vendor_name>/
2. Implement the backend class inheriting from Backend
3. Create register_ops.py with registration function
4. The backend will be auto-discovered by builtin_ops.py

See the "Adding Vendor Backends" section in dispatch/README.md for detailed instructions.
"""

__all__ = []
