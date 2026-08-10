# Copyright (c) 2025 BAAI. All rights reserved.

"""
Gold-value comparison for functional tests.

Supports two comparison modes:
- **Numeric**: Uses ``numpy.allclose`` with platform-specific tolerance
  (rtol/atol from ``PlatformConfig``).
- **Text**: Exact string match or substring containment.

Gold files are stored as JSON alongside test directories::

    tests/e2e_tests/inference/gold/qwen3_4b_tp2_cuda.json
    tests/e2e_tests/serving/gold/qwen2_5_0.5b_ascend.json

Gold file schema::

    {
        "platform": "cuda",
        "device": "a100",
        "entries": [
            {
                "name": "test_models[bfloat16-5-True]",
                "type": "numeric",        // "numeric" | "text"
                "value": [0.123, 0.456],   // list[float] for numeric
                "text": "Hello world"      // str for text
            }
        ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .platform_config import Tolerance

GOLD_DIR_NAME = "gold"
# Legacy directory names (from FlagScale migration)
_LEGACY_GOLD_DIRS = ["gold_values", "results_gold"]


@dataclass
class GoldEntry:
    """A single gold-value entry for one test case output."""

    name: str
    type: str = "numeric"  # "numeric" or "text"
    value: list[float] | None = None
    text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.value is not None:
            d["value"] = self.value
        if self.text is not None:
            d["text"] = self.text
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoldEntry:
        return cls(
            name=d["name"],
            type=d.get("type", "numeric"),
            value=d.get("value"),
            text=d.get("text"),
        )


@dataclass
class GoldFile:
    """Collection of gold entries for a test suite."""

    platform: str
    device: str
    entries: list[GoldEntry] = field(default_factory=list)

    def get_entry(self, name: str) -> GoldEntry | None:
        """Look up a gold entry by test name."""
        for e in self.entries:
            if e.name == name:
                return e
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "device": self.device,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoldFile:
        return cls(
            platform=d["platform"],
            device=d["device"],
            entries=[GoldEntry.from_dict(e) for e in d.get("entries", [])],
        )


def _gold_path(
    test_dir: str | Path,
    case_name: str,
    platform: str,
) -> Path:
    """Compute the path to a gold file.

    Example: ``tests/e2e_tests/inference/gold/qwen3_4b_tp2_cuda.json``
    """
    base = Path(test_dir) / GOLD_DIR_NAME
    # Sanitize case name for filesystem
    safe_name = case_name.replace("/", "_").replace(" ", "_")
    return base / f"{safe_name}_{platform}.json"


def _find_gold_file(test_dir: str | Path, case_name: str, platform: str) -> Path | None:
    """Search for a gold file across multiple directory conventions.

    Search order:
    1. ``<test_dir>/gold/<case>_<platform>.json``  (new convention)
    2. ``<test_dir>/gold_values/<case>.json``       (FlagScale legacy)
    3. ``<test_dir>/gold_values/<case>``            (plain text legacy)
    4. ``<test_dir>/results_gold/<case>.json``      (inference/rl legacy)
    5. ``<test_dir>/results_gold/<case>``            (plain text legacy)
    """
    primary = _gold_path(test_dir, case_name, platform)
    if primary.exists():
        return primary

    base = Path(test_dir)
    safe_name = case_name.replace("/", "_").replace(" ", "_")
    for legacy_dir in _LEGACY_GOLD_DIRS:
        for suffix in [f"{safe_name}.json", safe_name]:
            candidate = base / legacy_dir / suffix
            if candidate.exists():
                return candidate

    return None


def load_gold(
    test_dir: str | Path,
    case_name: str,
    platform: str,
    device: str | None = None,
) -> GoldFile | None:
    """Load gold values from disk, or return ``None`` if not found.

    Searches multiple directory conventions (see ``_find_gold_file``).
    If an override file exists (``gold_values/<case>.overrides.yaml``),
    platform-specific value overrides are applied.
    """
    path = _find_gold_file(test_dir, case_name, platform)
    if path is None:
        return None

    # Load raw gold data
    with open(path) as f:
        if path.suffix == ".json":
            raw = json.load(f)
        else:
            # Plain text — read lines as a single text entry
            raw = {"entries": [{"name": case_name, "type": "text", "text": f.read()}]}

    # New-format gold file has "entries" key
    if "entries" in raw:
        gold = GoldFile.from_dict(
            {
                "platform": raw.get("platform", platform),
                "device": raw.get("device", device or ""),
                "entries": raw["entries"],
            }
        )
    else:
        # Legacy format — treat entire dict as a single gold entry
        gold = GoldFile(platform=platform, device=device or "")
        gold.entries.append(
            GoldEntry(
                name=case_name,
                type="numeric",
                value=raw if isinstance(raw, list) else None,
            )
        )
        gold._raw = raw  # Preserve for override application

    # Apply per-platform overrides if present
    _apply_overrides(gold, test_dir, case_name, platform, device)

    return gold


def _apply_overrides(
    gold: GoldFile,
    test_dir: str | Path,
    case_name: str,
    platform: str,
    device: str | None,
) -> None:
    """Apply per-platform overrides from ``<case>.overrides.yaml``."""
    safe_name = case_name.replace("/", "_").replace(" ", "_")
    for search_dir in [GOLD_DIR_NAME, *_LEGACY_GOLD_DIRS]:
        override_path = Path(test_dir) / search_dir / f"{safe_name}.overrides.yaml"
        if override_path.exists():
            break
    else:
        return

    with open(override_path) as f:
        override_data = yaml.safe_load(f) or {}

    overrides = override_data.get("overrides", {})

    # Try most specific key first: "platform.device", then "platform"
    keys = [f"{platform}.{device}", platform] if device else [platform]
    matched = None
    for key in keys:
        if key in overrides:
            matched = overrides[key]
            break
    if matched is None:
        return

    # Apply per-metric value overrides to entries
    raw_dict = getattr(gold, "_raw", None)
    if isinstance(raw_dict, dict):
        for metric_key, metric_data in matched.items():
            if metric_key == "tolerance":
                continue
            if isinstance(metric_data, dict) and "values" in metric_data:
                raw_dict[metric_key] = metric_data


def save_gold(
    test_dir: str | Path,
    case_name: str,
    gold: GoldFile,
) -> Path:
    """Save gold values to disk. Creates the gold directory if needed."""
    path = _gold_path(test_dir, case_name, gold.platform)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(gold.to_dict(), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


@dataclass
class CompareResult:
    """Result of a single gold-value comparison."""

    name: str
    passed: bool
    message: str = ""
    expected: Any = None
    actual: Any = None


def compare_numeric(
    name: str,
    actual: list[float],
    expected: list[float],
    tolerance: Tolerance,
) -> CompareResult:
    """Compare two numeric arrays using ``numpy.allclose``."""
    if tolerance.exact:
        passed = actual == expected
        msg = "" if passed else "Values differ (exact match required)"
    else:
        a = np.array(actual, dtype=np.float64)
        e = np.array(expected, dtype=np.float64)
        passed = bool(np.allclose(a, e, rtol=tolerance.rtol, atol=tolerance.atol))
        if not passed:
            max_diff = float(np.max(np.abs(a - e)))
            msg = (
                f"Max absolute diff: {max_diff:.6e} "
                f"(rtol={tolerance.rtol}, atol={tolerance.atol})"
            )
        else:
            msg = ""
    return CompareResult(
        name=name,
        passed=passed,
        message=msg,
        expected=expected,
        actual=actual,
    )


def compare_text(
    name: str,
    actual: str,
    expected: str,
    *,
    substring: bool = False,
) -> CompareResult:
    """Compare two strings (exact or substring match)."""
    if substring:
        passed = expected in actual
        msg = "" if passed else "Expected substring not found in output"
    else:
        passed = actual == expected
        msg = "" if passed else "Text mismatch"
    return CompareResult(
        name=name,
        passed=passed,
        message=msg,
        expected=expected,
        actual=actual,
    )


def resolve_tolerance(
    test_dir: str | Path,
    case_name: str,
    platform: str,
    device: str | None = None,
    default: Tolerance | None = None,
) -> Tolerance:
    """Resolve tolerance without loading gold values.

    Resolution order (first match wins):
    1. Test-case override file (``<case>.overrides.yaml``, per-platform key)
    2. The provided *default* (typically from ``PlatformConfig.get_tolerance()``)
    3. ``Tolerance()`` library defaults
    """
    tol = default or Tolerance()

    safe_name = case_name.replace("/", "_").replace(" ", "_")
    for search_dir in [GOLD_DIR_NAME, *_LEGACY_GOLD_DIRS]:
        override_path = Path(test_dir) / search_dir / f"{safe_name}.overrides.yaml"
        if override_path.exists():
            break
    else:
        return tol

    with open(override_path) as f:
        override_data = yaml.safe_load(f) or {}

    overrides = override_data.get("overrides", {})
    keys = [f"{platform}.{device}", platform] if device else [platform]
    for key in keys:
        if key in overrides:
            tol_dict = overrides[key].get("tolerance", {})
            if tol_dict:
                return Tolerance.from_dict(
                    {
                        "rtol": tol_dict.get("rtol", tol.rtol),
                        "atol": tol_dict.get("atol", tol.atol),
                        "exact": tol_dict.get("exact", tol.exact),
                    }
                )
            break

    return tol


def compare_entry(
    entry: GoldEntry,
    actual_value: list[float] | None = None,
    actual_text: str | None = None,
    tolerance: Tolerance | None = None,
) -> CompareResult:
    """Compare a single gold entry against actual results."""
    if entry.type == "numeric":
        if actual_value is None or entry.value is None:
            return CompareResult(
                name=entry.name,
                passed=False,
                message="Missing numeric value for comparison",
            )
        return compare_numeric(
            entry.name,
            actual_value,
            entry.value,
            tolerance or Tolerance(),
        )
    elif entry.type == "text":
        if actual_text is None or entry.text is None:
            return CompareResult(
                name=entry.name,
                passed=False,
                message="Missing text value for comparison",
            )
        return compare_text(entry.name, actual_text, entry.text)
    else:
        return CompareResult(
            name=entry.name,
            passed=False,
            message=f"Unknown gold entry type: {entry.type}",
        )
