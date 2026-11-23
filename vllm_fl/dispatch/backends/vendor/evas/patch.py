# Copyright (c) 2026 BAAI. All rights reserved.
"""Evas (LPU) compatibility patches.

Aggregates LPU patches that bypass vLLM v1's Triton kernels (which fail on
LPU at compile time: pointwise/reduction kernels hit evas-triton-opt/evofc
compile blockers) and work around LPU backend limitations (2D-GEMM
correctness, event/stream semantics, block-table slot-mapping). Mirrors the
apply_<vendor>_patches() convention in vendor/{ascend,sunrise}/patch.py.

Individual patches are declared in ``patches/registry.py``. Use
``VLLM_FL_LPU_DISABLE_PATCHES=patch1,patch2`` to disable specific patches at
runtime.
"""

from __future__ import annotations

import itertools
import os

from vllm.logger import init_logger

logger = init_logger(__name__)

_patches_applied = False

# Names of specs whose apply function reported a missing target on the last
# install pass. Kept for tests and for callers that want to surface the list.
_declined_patches: list[str] = []


def _leading_int(part: str) -> int:
    """Leading digits of a version part ("25rc1" -> 25); 0 when there are none."""
    digits = "".join(itertools.takewhile(str.isdigit, part))
    return int(digits) if digits else 0


def _installed_vllm() -> tuple[int, int]:
    """Major/minor of the installed core; a parse failure never skips a spec."""
    try:
        import vllm

        parts = (vllm.__version__.split(".") + ["0", "0"])[:2]
        return (_leading_int(parts[0]), _leading_int(parts[1]))
    except Exception:  # noqa: BLE001 -- a version oddity must not skip patches
        # Fail open: an unreadable version must not silently drop patches whose
        # floor we cannot compare against.
        return (99, 99)


def _install_specs(specs) -> list[str]:
    """Run ``specs`` in order; return the names that declined to install.

    A spec's ``apply_fn`` returns False when it looked for its target symbol
    and found it missing. That case must not be logged as "Applied": the
    apply call itself always succeeds, so without this distinction a renamed
    method reads exactly like a working patch (see the mamba_state warning).
    ``VLLM_FL_STRICT_UPSTREAM_REBIND=1`` turns the decline into a raise, which
    is what a patch author should use when checking a rebind by hand.

    A spec whose ``min_vllm`` floor is above the installed core is skipped by
    design (its anchors do not exist yet), so it is not a decline.
    """
    strict = os.getenv("VLLM_FL_STRICT_UPSTREAM_REBIND", "0") == "1"
    installed = _installed_vllm()
    declined: list[str] = []
    for spec in specs:
        if spec.min_vllm is not None and installed < tuple(spec.min_vllm):
            logger.info(
                "Skipped evas patch (needs vllm >= %s): %s",
                ".".join(map(str, spec.min_vllm)),
                spec.name,
            )
            continue
        if spec.gate is not None and not spec.gate():
            # Log gate-skips: a closed gate is otherwise invisible, so which
            # patch is waiting on e.g. the base DSV4 package cannot be told.
            logger.info("Skipped evas patch (gate closed): %s", spec.name)
            continue
        status = spec.apply_fn()
        if status is False:
            declined.append(spec.name)
            logger.warning(
                "Declined evas patch (target missing): %s -- the rebind was "
                "not installed; see the patch's own warning for the symbol",
                spec.name,
            )
            if strict:
                raise RuntimeError(
                    f"evas patch {spec.name!r} declined to install: target "
                    "symbol missing (VLLM_FL_STRICT_UPSTREAM_REBIND=1)"
                )
            continue
        logger.info("Applied evas patch: %s", spec.name)
    return declined


def apply_evas_patches() -> None:
    """Apply all LPU compatibility patches, driven by the declarative registry.

    Idempotent. No-op off LPU. Respects ``VLLM_FL_LPU_DISABLE_PATCHES`` for
    per-patch disable control and ``VLLM_FL_STRICT_UPSTREAM_REBIND`` for
    turning a declined install into an error.
    """
    global _patches_applied, _declined_patches
    if _patches_applied:
        return

    from .config import get_evas_config
    from .patches.registry import _build_registry

    cfg = get_evas_config()
    specs = _build_registry(cfg.disable_patches)

    _declined_patches = _install_specs(specs)
    if _declined_patches:
        logger.warning(
            "Evas patches declined (target missing): %s",
            ", ".join(_declined_patches),
        )

    _patches_applied = True
