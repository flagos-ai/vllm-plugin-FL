"""Load the Ascend custom operators shipped by the FL plugin itself."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_ENABLED: bool | None = None

# CANN packages A2 and A3 kernels under family names.  Keep the concrete
# runtime SOC names here so a missing A3 package can never fall back to A2.
_SOC_FAMILY_ALIASES = {
    "ascend910b": {
        "910b",
        "910b1",
        "910b2",
        "910b2c",
        "910b3",
        "910b4",
        "910b4-1",
        "ascend910b",
        "ascend910b1",
        "ascend910b2",
        "ascend910b2c",
        "ascend910b3",
        "ascend910b4",
        "ascend910b4-1",
    },
    "ascend910_93": {
        "910c",
        "ascend910_93",
        "ascend910_9362",
        "ascend910_9372",
        "ascend910_9381",
        "ascend910_9382",
        "ascend910_9391",
        "ascend910_9392",
    },
}

_SOC_PREBUILT_DIR = {
    "ascend910b": "ascend910b1",
    "ascend910_93": "ascend910_93",
}


def _prepend_env_path(name: str, path: Path) -> None:
    if not path.exists():
        return
    entries = [entry for entry in os.environ.get(name, "").split(":") if entry]
    path_str = str(path)
    if path_str not in entries:
        entries.insert(0, path_str)
        os.environ[name] = ":".join(entries)


def _soc_family(value: str) -> str:
    normalized = value.strip().lower()
    for family, aliases in _SOC_FAMILY_ALIASES.items():
        if normalized in aliases:
            return family
    return normalized


def _soc_candidates() -> list[str]:
    value = os.environ.get("SOC_VERSION", "ascend910b1").strip().lower()
    family = _soc_family(value)
    candidates = [value]
    prebuilt_dir = _SOC_PREBUILT_DIR.get(family)
    if prebuilt_dir:
        candidates.append(prebuilt_dir)
    candidates.append(family)
    return list(dict.fromkeys(candidates))


def _prebuilt_roots(package_dir: Path) -> list[Path]:
    root = package_dir / "dispatch" / "backends" / "vendor" / "ascend" / "prebuilt"
    return [root / soc for soc in _soc_candidates()]


def _find_compatible_extension(root: Path) -> Path | None:
    extensions = sorted(root.glob("lib/_C_ascend*.so"))
    cache_tag = sys.implementation.cache_tag
    if cache_tag:
        for extension in extensions:
            if cache_tag in extension.name:
                return extension
    return next((path for path in extensions if path.name == "_C_ascend.so"), None)


def _opp_supports_soc_family(opp: Path, family: str) -> bool:
    tbe = opp / "op_impl" / "ai_core" / "tbe"
    return (tbe / "kernel" / family).is_dir() and (tbe / "config" / family).is_dir()


def _load_local() -> bool:
    package_dir = Path(__file__).resolve().parent
    family = _soc_family(os.environ.get("SOC_VERSION", "ascend910b1"))
    for root in _prebuilt_roots(package_dir):
        extension = _find_compatible_extension(root)
        opp_candidates = (
            root / "opp" / "custom_transformer",
            root / "opp" / "vendors" / "custom_transformer",
        )
        opp = next((path for path in opp_candidates if path.is_dir()), None)
        if not (
            extension
            and opp is not None
            and _opp_supports_soc_family(opp, family)
        ):
            continue
        _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", opp)
        op_api_lib = opp / "op_api" / "lib"
        _prepend_env_path("LD_LIBRARY_PATH", op_api_lib)
        try:
            # The FL binding registers only the Qwen3.6-required schemas and
            # dispatch implementations.  The actual kernels come from OPP.
            custom_opapi = op_api_lib / "libcust_opapi.so"
            if custom_opapi.is_file():
                torch.ops.load_library(str(custom_opapi))
            torch.ops.load_library(str(extension))
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning("FL vendored Ascend libraries unavailable: %s", exc)
            return False
        logger.info("Loaded FL vendored Ascend libraries for %s", root.name)
        return True
    logger.debug("No FL vendored Ascend libraries found for SOC_VERSION=%s", os.environ.get("SOC_VERSION"))
    return False


def enable_custom_op() -> bool:
    """Load FL's local Ascend extension."""
    global _ENABLED
    if _ENABLED is not None:
        return _ENABLED
    _ENABLED = _load_local()
    if _ENABLED:
        logger.info("Ascend custom ops enabled")
    return _ENABLED


def bootstrap_custom_op_env() -> None:
    """Expose the custom OPP path packaged by FL."""
    package_dir = Path(__file__).resolve().parent
    for root in _prebuilt_roots(package_dir):
        for local_opp in (
            root / "opp" / "custom_transformer",
            root / "opp" / "vendors" / "custom_transformer",
        ):
            if local_opp.exists():
                _prepend_env_path("ASCEND_CUSTOM_OPP_PATH", local_opp)
                return
