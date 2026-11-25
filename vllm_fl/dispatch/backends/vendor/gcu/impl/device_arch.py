# Copyright (c) 2026 BAAI. All rights reserved.

"""GCU architecture detection (gcu300 / gcu400 / ...).

The same plugin binary serves two Enflame platforms:

* **L600** -> ``gcu400`` (driver 4.1.6, more registers, int64 supported);
* **S60**  -> ``gcu300`` (40 GiB cards, no int64 element-wise ops, tighter
  register file).

Some workarounds are only required on one of them (gcu300 needs the opaque
gated-RMSNorm op and int32 MoE index math).  Detect the architecture once and
keep the platform-specific branches next to the workaround they guard instead
of scattering version/arch string checks.

Detection order (first hit wins):
1. ``triton_gcu`` driver (authoritative; the same query FlagGems uses to pick
   ``_enflame/gcu300`` vs ``_enflame/gcu400``);
2. ``torch.gcu.get_device_properties(0).major`` (L600: 4, S60: 3);
3. FlagGems sets ``ARCH`` to the numeric version when its backend initializes.
"""

from __future__ import annotations

import functools
import os
import re

_ARCH_RE = re.compile(r"gcu\s*(\d+)", re.IGNORECASE)


def _normalize(arch: str | None) -> str | None:
    if not arch:
        return None
    match = _ARCH_RE.search(arch)
    if match:
        return f"gcu{match.group(1)}"
    match = re.fullmatch(r"\s*(\d+)\s*", arch)
    if match:
        return f"gcu{match.group(1)}"
    return None


@functools.lru_cache(maxsize=1)
def get_gcu_arch() -> str | None:
    """Return ``"gcu300"`` / ``"gcu400"`` for the active device, or None.

    The result is cached: the architecture cannot change within a process.
    """
    try:
        from triton_gcu.triton.driver import _GCUDriver

        arch = _normalize(str(_GCUDriver().get_arch()))
        if arch:
            return arch
    except Exception:  # noqa: BLE001 - vendor API may differ between stacks
        pass

    try:
        import torch

        props = torch.gcu.get_device_properties(0)
        return f"gcu{int(props.major)}00"
    except Exception:  # noqa: BLE001 - not a GCU process / torch_gcu absent
        pass

    return _normalize(os.environ.get("ARCH"))


def is_gcu300() -> bool:
    """gcu300 family (S60); the driver may report e.g. ``gcu300`` only."""
    arch = get_gcu_arch()
    return bool(arch and arch.startswith("gcu3"))

