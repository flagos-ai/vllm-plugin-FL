# Copyright (c) 2026 BAAI. All rights reserved.

"""T-Head backend exports, kept lazy for early native-schema initialization."""

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    # Make the public export visible to type checkers and static analysis while
    # keeping the runtime import lazy. Importing the backend eagerly can load
    # fallback schemas before the optional native bundle is initialized.
    from .thead import TheadBackend as TheadBackend


__all__ = ["TheadBackend"]


def __getattr__(name: str) -> Any:
    if name == "TheadBackend":
        from .thead import TheadBackend

        globals()[name] = TheadBackend
        return TheadBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
