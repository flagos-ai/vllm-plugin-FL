# Copyright (c) 2026 BAAI. All rights reserved.
"""Per-symbol import for vLLM fused ops: prefer ``flaggems_vllm``, fall back to ``flag_gems``.

Why this exists
---------------
FlagGems >= 5.5 upstreamed the vLLM-specific fused operators into
``flag_gems.fused`` (``moe_align_block_size_triton``, ``topk_softmax``,
``invoke_fused_moe_triton_kernel``, ``grouped_topk``, ``moe_sum``,
``flash_attn_varlen_func``, ``flash_mla`` ...), so the standalone
``flaggems_vllm`` package is optional on those stacks.

The impl modules under ``dispatch/backends/flaggems/impl/`` used to import
``flaggems_vllm`` at **module top level**. ``builtin_ops.register_builtins``
wraps the whole FlagGems registration in ``try/except Exception`` and only logs
a warning, so a single missing symbol silently disabled **every** FlagGems
operator -- which pushed all five MoE ops to ``reference.torch`` (correct but
slow, and the source of several int64/bool pitfalls on gcu300).

``gems_import`` keeps the old provider preferred (so behaviour is unchanged on
stacks that do ship ``flaggems_vllm``) and falls back per symbol. Missing names
now only affect the single operator that needs them instead of the whole
backend.
"""

import importlib
import logging

logger = logging.getLogger(__name__)

__all__ = ["gems_import"]


def gems_import(name: str):
    """Return ``flaggems_vllm.<name>`` if available, else ``flag_gems.<name>``.

    Raises ``ImportError`` when neither provider exposes the symbol, so call
    sites keep the same failure mode they had before.
    """
    try:
        package = importlib.import_module("flaggems_vllm")
        return getattr(package, name)
    except (ImportError, AttributeError):
        pass

    try:
        package = importlib.import_module("flag_gems")
    except ImportError as exc:  # pragma: no cover - FlagGems is a hard dependency
        raise ImportError(
            f"cannot resolve '{name}': neither 'flaggems_vllm' nor 'flag_gems' is importable"
        ) from exc

    try:
        return getattr(package, name)
    except AttributeError as exc:
        raise ImportError(
            f"cannot resolve '{name}': not provided by 'flaggems_vllm' or 'flag_gems'"
        ) from exc

