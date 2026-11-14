# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Supply the AMD architecture predicates FlagTree's ``target_info`` omits.

vLLM 0.32 vendors ``vllm/third_party/triton_kernels``, and both its
``target_info.py`` and ``tensor_details/layout.py`` import
``is_hip_gfx1250`` from ``triton.language.target_info``. Upstream Triton 3.8
defines it; FlagTree ``0.7.0+amd3.6`` exports only ``is_cuda``, ``is_hip``,
``is_hip_cdna3`` and ``is_hip_cdna4``. Once FlagTree supplies Triton, importing
those modules fails with::

    cannot import name 'is_hip_gfx1250' from 'triton.language.target_info'

which takes down any path that touches the bundled Triton kernels — the MoE
layers in particular, so a dense model can serve while an MoE model cannot.

Each predicate is derived from the live target rather than hardcoded, and only
names FlagTree is missing are added, so this becomes a no-op as FlagTree
catches up.
"""

from __future__ import annotations

from functools import lru_cache

from vllm.logger import init_logger

logger = init_logger(__name__)

# Architecture families queried by vLLM's bundled Triton kernels but absent
# from FlagTree's target_info, mapped to the gfx prefixes that satisfy them.
_ARCH_PREDICATES = {
    "is_hip_gfx1250": ("gfx1250", "gfx1251"),
    "is_hip_cdna": ("gfx908", "gfx90a", "gfx942", "gfx950"),
    "is_hip_cdna1": ("gfx908",),
    "is_hip_cdna2": ("gfx90a",),
    "is_hip_rdna": ("gfx11", "gfx12"),
    "is_hip_gfx11": ("gfx11",),
    "is_hip_gfx12": ("gfx12",),
}


def _current_arch(target_info) -> str:
    """Return the live gfx target name, or "" when it cannot be determined."""
    try:
        target = target_info.current_target()
    except Exception:
        return ""
    arch = getattr(target, "arch", None) if target is not None else None
    return arch if isinstance(arch, str) else ""


def _make_predicate(target_info, name: str, prefixes: tuple[str, ...]):
    """Build a ``triton.language`` constexpr predicate for *prefixes*."""

    def predicate() -> bool:
        if not target_info.is_hip():
            return False
        return _current_arch(target_info).startswith(prefixes)

    predicate.__name__ = name
    # The real predicates are constexpr functions so they can be folded inside
    # @triton.jit bodies; mirror that when FlagTree offers the decorator.
    decorator = getattr(target_info, "constexpr_function", None)
    if callable(decorator):
        try:
            return decorator(predicate)
        except Exception:
            return predicate
    return predicate


@lru_cache(maxsize=1)
def apply_flagtree_target_info() -> bool:
    """Add the missing AMD predicates to ``triton.language.target_info``.

    Cached, so repeat calls are free, and a no-op when FlagTree already
    defines them. Caching the negative result is safe because Triton's
    importability does not change within a process.

    Returns:
        True when the module is importable (whether or not anything was
        added), False when Triton is unavailable.
    """
    try:
        import triton.language.target_info as target_info
    except ImportError as e:
        logger.debug("triton.language.target_info unavailable: %s", e)
        return False

    added = []
    for name, prefixes in _ARCH_PREDICATES.items():
        if getattr(target_info, name, None) is None:
            setattr(target_info, name, _make_predicate(target_info, name, prefixes))
            added.append(name)

    if added:
        logger.info(
            "AMD: added %d architecture predicate(s) missing from this Triton's "
            "target_info (%s); vLLM's bundled triton_kernels import them.",
            len(added),
            ", ".join(added),
        )
    return True
