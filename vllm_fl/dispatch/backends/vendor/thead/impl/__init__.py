"""Copyright (c) 2026 BAAI. All rights reserved."""

__all__ = ["TheadFlashAttentionBackend", "TheadFlashAttentionImpl"]


def __getattr__(name):
    # MLA must not import an unrelated ordinary attention implementation.
    if name in __all__:
        from . import attention

        return getattr(attention, name)
    raise AttributeError(name)
