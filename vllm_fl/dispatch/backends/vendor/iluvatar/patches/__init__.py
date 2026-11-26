# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Iluvatar CoreX thin-backend runtime patch entry point.

Imported by `PlatformFL.import_kernels()` and `iluvatar.py` to idempotently
activate platform config, model-loader quantization, `DG-R2-K8V4` KV-cache,
step-prep, locality block-queue, and sort-free sampler hooks.
"""

from __future__ import annotations

from .runtime_hooks import install_iluvatar_runtime_hooks, install_worker_runtime_hooks

install_iluvatar_runtime_hooks()

__all__ = ["install_iluvatar_runtime_hooks", "install_worker_runtime_hooks"]
