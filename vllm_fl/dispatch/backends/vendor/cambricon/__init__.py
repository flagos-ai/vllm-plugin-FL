# Copyright (c) 2026 BAAI. All rights reserved.

"""
Cambricon MLU vendor patches for vllm-plugin-FL dispatch.

There is no dedicated cambricon dispatch backend (ops route through flag_gems,
see dispatch/config/cambricon.yaml); this package exists only to host
cambricon-scoped source patches.
"""

__all__ = []
