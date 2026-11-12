# Copyright (c) 2026 BAAI. All rights reserved.
"""Import compatibility needed by the Iluvatar worker on vLLM 0.32."""

import importlib
import sys

from vllm_fl.patches.worker_symbols_032 import apply_worker_symbols_032


def apply_import_compat() -> None:
    # Use the shared #737 shim as-is; do not replace its definitions.
    apply_worker_symbols_032()
    # Shared workers still support older vLLM releases. This module moved in
    # 0.32; alias only when running the Iluvatar worker on that release.
    old = "vllm.model_executor.layers.batch_invariant"
    try:
        importlib.import_module(old)
    except ModuleNotFoundError as exc:
        if exc.name != old:
            raise
        sys.modules[old] = importlib.import_module(
            "vllm.model_executor.determinism.batch_invariant"
        )
