# Copyright (c) 2026 BAAI. All rights reserved.

"""
Cambricon MLU vendor patches.

Hosts cambricon-scoped runtime patches that rewrite vllm sources on disk.
Kept out of ``vllm_fl.platform`` and ``vllm_fl/__init__.py`` so the patch logic
stays self-contained and testable, mirroring the iluvatar vendor layout.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def patch_triton_chained_or_for_cambricon() -> None:
    """Parenthesize chained boolean ``or`` in vllm triton kernels.

    Cambricon 4.4.3 ships triton 3.2.0+mlu1.7.2, whose frontend rejects
    unparenthesized chained boolean operators (``A or B or C``) inside
    @triton.jit bodies with UnsupportedLanguageConstruct. The exception escapes
    the autotuner and kills the EngineCore (HTTP 500), unlike iluvatar where the
    same construct fails earlier at DependencyFinder resolution. neuware4.7.2
    (triton 3.4.0+mlu2.1.1) parses chained boolean operators natively, so the
    patch is gated on triton < 3.4.

    Rewrites the affected vllm source files in-place (idempotent, marked).
    Must run in the main process before Worker subprocesses spawn, so the
    patched files are on disk when Workers import them.

    TODO: Remove once the minimum supported cambricon triton version is >= 3.4.
    """
    import importlib.util
    import pathlib
    import re
    import shutil
    import sys

    try:
        import triton as _triton
        _tv = tuple(int(x) for x in _triton.__version__.split(".")[:2])
        if _tv >= (3, 4):
            return
    except Exception as e:
        logger.warning(
            "patch_triton_chained_or_for_cambricon: cannot determine triton "
            "version, applying patch defensively: %s", e
        )

    # Same failure class as metax (triton 3.0.0, recorded in
    # packaging/vllm/docs/vllm-0.24.0/backends/metax.md §4.3) and iluvatar
    # (vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py) — the 0.24.0
    # wheel's three unparenthesized three-way ``or`` chains in jit bodies.
    _MODULES = (
        # kernel_unified_attention fast path gate — actual crash site on the
        # first request (triton_attention_helpers.py imports it at kernel
        # compile time from triton_unified_attention.py).
        "vllm.v1.attention.ops.triton_attention_helpers",
        # batch_invariant gate — latent, reached when VLLM_BATCH_INVARIANT
        # variants compile.
        "vllm.model_executor.layers.batch_invariant",
        # sampler penalty gate — latent, reached on the first penalty request.
        "vllm.v1.worker.gpu.sample.penalties",
    )
    _MARKER = "# _cambricon_chained_or_patched"

    # Replace: A or B or C  →  (A or B) or C
    _OPERAND = r'(?:not\s+)?(?:\([^)]*\)|\w+)'
    pattern = re.compile(
        r'(' + _OPERAND + r')\s+or\s+(' + _OPERAND + r')\s+or\s+(' + _OPERAND + r')'
    )

    def _rewrite(m: re.Match) -> str:
        return f"({m.group(1)} or {m.group(2)}) or {m.group(3)}"

    for module_name in _MODULES:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            logger.warning(
                "patch_triton_chained_or_for_cambricon: %s not found, skipping.",
                module_name,
            )
            continue

        fpath = pathlib.Path(spec.origin)
        try:
            src = fpath.read_text()
        except Exception as e:
            logger.warning(
                "patch_triton_chained_or_for_cambricon: cannot read %s: %s",
                fpath, e,
            )
            continue

        if _MARKER in src:
            continue  # already patched

        new_src, count = re.subn(pattern, _rewrite, src)
        if count == 0:
            continue  # nothing to patch

        new_src += f"\n{_MARKER}\n"
        try:
            fpath.write_text(new_src)
        except Exception as e:
            logger.warning(
                "patch_triton_chained_or_for_cambricon: cannot write %s: %s",
                fpath, e,
            )
            continue

        # Clear pycache so Python and triton both see the patched source.
        pycache = fpath.parent / "__pycache__"
        if pycache.exists():
            try:
                shutil.rmtree(pycache)
            except Exception:
                pass  # non-fatal

        # Evict from sys.modules so this process reimports the patched source.
        sys.modules.pop(module_name, None)

        logger.info(
            "patch_triton_chained_or_for_cambricon: rewrote %d chained-or "
            "expression(s) in %s", count, fpath,
        )
