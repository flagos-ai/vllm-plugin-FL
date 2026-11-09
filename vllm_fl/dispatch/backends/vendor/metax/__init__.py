# Copyright (c) 2026 BAAI. All rights reserved.

"""
MACA backend for vllm-plugin-FL dispatch.
"""

from .metax import MacaBackend

__all__ = [
    "MacaBackend",
]

from vllm_fl.patches.metax_open import uses_open_metax_runtime

# The open ABI route installs its explicit MACA/MCCL hooks.
# Unconfigured runtimes retain the original MetaX startup patches.
if not uses_open_metax_runtime():
    from . import patches  # noqa: F401
