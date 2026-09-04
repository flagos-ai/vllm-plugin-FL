# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Bridge the dynamic FL dispatch control plane to a compiled runner."""

from __future__ import annotations

import logging
from typing import Any, Optional

from vllm_fl.dispatch import FrozenDispatchManifest, freeze_dispatch
from vllm_fl.dispatch.logger_manager import get_logger

logger = get_logger(__name__)

_HASH_KEY = "vllm_fl_dispatch_fingerprint"


def _install_logger_method_compile_guard() -> None:
    """Permit Dynamo to no-op logging.Logger calls in FlagGems wrappers.

    FlagGems' public Python wrappers (``flag_gems.ops.*`` /
    ``flag_gems.fused.*`` / ``flag_gems.modules.*``) contain plain
    ``logger.debug(...)`` calls on their entry paths — e.g.
    ``flag_gems/ops/rms_norm.py::rms_norm_forward`` emits
    ``"GEMS RMS_NORM FORWARD"``. Torch Dynamo cannot trace
    ``logging.Logger`` methods and raises
    ``torch._dynamo.exc.Unsupported: logging.Logger method not supported for
    non-export cases`` the moment any of those wrappers is reached from a
    compiled region (e.g. with ``custom_ops=all``). Registering the four
    unbound level methods in
    ``torch._dynamo.config.ignore_logging_functions`` makes Dynamo treat such
    calls as no-ops *inside compiled graphs only*; eager logging behavior is
    unchanged. This is the PyTorch-documented escape hatch (it exists
    precisely for library code that logs on hot paths) and is idempotent.
    """

    try:
        import torch._dynamo.config as _dynamo_config
    except Exception:  # pragma: no cover - torch always present in practice
        return
    ignore = getattr(_dynamo_config, "ignore_logging_functions", None)
    if ignore is None or not hasattr(ignore, "add"):
        return
    for method_name in ("debug", "info", "warning", "error"):
        ignore.add(getattr(logging.Logger, method_name))


def _is_compiled_execution(vllm_config: Any) -> bool:
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is None:
        return False
    if getattr(compilation_config, "backend", None) == "eager":
        return False

    mode = getattr(compilation_config, "mode", None)
    mode_name = getattr(mode, "name", str(mode)).upper()
    return mode is not None and mode_name not in {"NONE", "COMPILATIONMODE.NONE"}


def _add_cache_fingerprint(
    vllm_config: Any, manifest: FrozenDispatchManifest
) -> None:
    """Make backend choices part of vLLM's compilation cache identity."""

    additional_config = getattr(vllm_config, "additional_config", None)
    if isinstance(additional_config, dict):
        previous = additional_config.get(_HASH_KEY)
        if previous is not None and previous != manifest.fingerprint:
            raise RuntimeError(
                "vLLM-FL dispatch selection changed after the vLLM config was "
                "prepared. Rebuild the model runner before compiling."
            )
        additional_config[_HASH_KEY] = manifest.fingerprint
    else:
        logger.warning(
            "VllmConfig.additional_config is not a dict; the frozen FL dispatch "
            "fingerprint could not be added to the outer vLLM cache key"
        )

    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is not None:
        # These fields are lazily recomputed by vLLM.  Clear any path that may
        # have been derived before the dispatch fingerprint was attached.
        for attribute in ("cache_dir", "local_cache_dir"):
            if hasattr(compilation_config, attribute):
                setattr(compilation_config, attribute, "")


def freeze_dispatch_for_compile(
    vllm_config: Any,
) -> Optional[FrozenDispatchManifest]:
    """Freeze all imported ``CachedOp`` sites before the first Dynamo trace.

    Optional operator modules can be imported by model registries without ever
    being executed, so unresolved entries are recorded in the manifest rather
    than failing model load.  Calling one of them after freeze still raises a
    deterministic error before entering Dynamo.
    """

    if not _is_compiled_execution(vllm_config):
        return None

    _install_logger_method_compile_guard()
    manifest = freeze_dispatch(strict=False)
    _add_cache_fingerprint(vllm_config, manifest)

    if manifest.unresolved:
        logger.warning(
            "Frozen vLLM-FL dispatch with %d unresolved optional op(s): %s",
            len(manifest.unresolved),
            ", ".join(op_name for op_name, _ in manifest.unresolved),
        )
    logger.info(
        "Frozen vLLM-FL dispatch for torch.compile: %d op(s), fingerprint=%s",
        len(manifest.selections),
        manifest.fingerprint[:12],
    )
    return manifest


__all__ = ["freeze_dispatch_for_compile"]
