# Copyright (c) 2026 BAAI. All rights reserved.

"""
Txda (tsingmicro) backend operator implementations.

Each operator category lives in its own module (e.g. attention.py). Modules
here are imported lazily by the dispatcher via the dotted paths returned from
TxdaBackend's accessors.
"""

__all__ = []
