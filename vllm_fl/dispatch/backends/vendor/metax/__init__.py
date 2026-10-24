# Copyright (c) 2026 BAAI. All rights reserved.

"""
MACA backend for vllm-plugin-FL dispatch.
"""

from .metax import MacaBackend

__all__ = [
    "MacaBackend",
]

from vllm_fl.patches.kimi_k3.profile import is_kimi_k3_profile

# The open K3 profile installs only its explicit MACA/MCCL ABI hooks.
# Other models retain the upstream MetaX startup patches.
if not is_kimi_k3_profile():
    from . import patches  # noqa: F401
