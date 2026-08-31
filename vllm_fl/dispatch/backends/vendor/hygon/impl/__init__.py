# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon implementations grouped by their integration boundary.

``native_ops`` supplies empty-install fallbacks for direct ``torch.ops._C``
and ``torch.ops._moe_C`` call sites. ``custom_ops`` contains vLLM CustomOp
implementations, while ``attention``, ``moe``, and ``other`` hold backend
implementations for their respective domains.
"""
