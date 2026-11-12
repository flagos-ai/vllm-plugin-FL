# Copyright (c) 2026 BAAI. All rights reserved.

"""MetaX (MACA) backend for the vllm-plugin-FL dispatch layer.

Everything MetaX-specific lives under this package:

- ``impl/``    operator implementations for the dispatch layer
- ``patches/`` vLLM-level replacements applied when the backend loads
- ``shims/``   import-time compatibility shims for the vLLM 0.28 empty wheel

The imports below stay lazy on purpose. ``vllm_fl.register()`` loads ``shims``
while vLLM is still initialising; importing ``MacaBackend`` (which pulls
``vllm.v1.attention.backends.registry``) or ``patches`` (which pulls the mamba
model layers) at that moment re-creates the circular import the shims exist to
avoid.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .metax import MacaBackend

__all__ = ["MacaBackend"]


def __getattr__(name: str):
    if name == "MacaBackend":
        # Backend-load time: the model layers are importable here.
        from vllm_fl.patches.metax_open import uses_open_metax_runtime

        if not uses_open_metax_runtime():
            from . import patches  # noqa: F401 — apply MetaX kernel patches

        from .metax import MacaBackend

        return MacaBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
