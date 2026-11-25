"""T-Head backend implementation exports.

Keep the attention compatibility exports lazy.  Importing an unrelated backend
such as ``impl.mla`` must not import the T-Head attention implementation first:
its vLLM-facing symbols can differ between supported vLLM releases.
"""

from typing import Any

__all__ = ["TheadFlashAttentionBackend", "TheadFlashAttentionImpl"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .attention import TheadFlashAttentionBackend, TheadFlashAttentionImpl

        return {
            "TheadFlashAttentionBackend": TheadFlashAttentionBackend,
            "TheadFlashAttentionImpl": TheadFlashAttentionImpl,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
