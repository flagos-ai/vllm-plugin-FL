# Copyright (c) 2026 BAAI. All rights reserved.

"""T-Head-only initialization before vLLM fallback schemas are registered."""

import logging
import os


logger = logging.getLogger(__name__)
_WARNED_MISSING_BUNDLES: set[str] = set()


def initialize_native_extensions() -> bool:
    """Load the PPU bundle without consulting the unfinished current_platform.

    The native loader owns SO paths, load order, idempotency and vLLM module
    compatibility. Loading must happen before portable TORCH_LIBRARY schemas;
    deferring it to operator backend selection would duplicate definitions.

    A missing optional bundle leaves the existing FlagGems/reference fallback
    path active. Other loading failures, including ABI or dependency errors,
    still propagate because continuing from a damaged bundle is unsafe.
    """
    if "PPU_SDK" not in os.environ:
        return False

    from .impl.native_extensions import (
        NativeExtensionBundleMissingError,
        load_all_native_extensions,
    )

    try:
        load_all_native_extensions()
    except NativeExtensionBundleMissingError as exc:
        message = str(exc)
        if message not in _WARNED_MISSING_BUNDLES:
            logger.warning(
                "T-Head native extensions are unavailable; continuing with "
                "configured FlagGems/reference fallbacks. Native vendor "
                "implementations will be disabled: %s",
                message,
            )
            _WARNED_MISSING_BUNDLES.add(message)
        return False
    return True
