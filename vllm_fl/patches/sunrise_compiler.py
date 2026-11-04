# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""Resolve the historical Sunrise ``tang`` math namespace on FlagTree PTPU.

FlagGems-vllm v0.2 names its Sunrise libdevice ``extra.tang``. The matched
FlagTree runtime publishes the same device math ABI under ``extra.ptpu``.
Use that real vendor module, including its signed integer-power externs.
"""

import importlib
import sys

_LEGACY_PACKAGE = "triton.language.extra.tang"
_LEGACY_LIBDEVICE = _LEGACY_PACKAGE + ".libdevice"


def install_sunrise_libdevice_compat():
    """Alias a missing historical namespace to executable PTPU device math."""
    try:
        return importlib.import_module(_LEGACY_LIBDEVICE)
    except ModuleNotFoundError as exc:
        if exc.name not in {_LEGACY_PACKAGE, _LEGACY_LIBDEVICE}:
            raise

    vendor_package = importlib.import_module("triton.language.extra.ptpu")
    vendor_math = importlib.import_module("triton.language.extra.ptpu.libdevice")
    for name in ("pow", "erf", "exp", "tanh"):
        if not callable(getattr(vendor_math, name, None)):
            raise RuntimeError(f"Sunrise PTPU libdevice has no executable {name}")

    # Do not modify triton.language.math or impersonate a CUDA libdevice.
    # Both historical import forms resolve to the real PTPU module objects.
    legacy_package = sys.modules.setdefault(_LEGACY_PACKAGE, vendor_package)
    sys.modules[_LEGACY_LIBDEVICE] = vendor_math
    legacy_package.libdevice = vendor_math
    extra = importlib.import_module("triton.language.extra")
    extra.tang = legacy_package
    return vendor_math
