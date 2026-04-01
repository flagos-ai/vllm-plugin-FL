# Copyright (c) 2025 BAAI. All rights reserved.

"""
Platform configuration loader.

Reads ``tests/platforms/<platform>.yaml`` and provides a typed interface
for tolerance, device info, test filtering, and environment defaults.

Also supports **device-alias resolution**: pass ``"a100"`` and it resolves
to the ``cuda`` platform automatically by scanning device_types in all
platform YAML files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Root of the repository (two levels up from this file)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLATFORMS_DIR = _REPO_ROOT / "tests" / "platforms"


# ---------------------------------------------------------------------------
# Platform auto-discovery (merged from utils/lib/platform.py)
# ---------------------------------------------------------------------------


def get_platforms_dir() -> Path:
    """Return the platforms configuration directory."""
    return _PLATFORMS_DIR


def build_platform_file_map(
    platforms_dir: Path | None = None,
) -> dict[str, str]:
    """Auto-discover platforms from YAML files and their device_types.

    Scans the platforms directory for YAML files, registering each file
    by its stem name and also registering all device_types as aliases.

    Returns:
        Dict mapping platform/device names to YAML filenames.
        E.g., ``{"cuda": "cuda.yaml", "a100": "cuda.yaml", "a800": "cuda.yaml"}``
    """
    if platforms_dir is None:
        platforms_dir = _PLATFORMS_DIR
    platforms_dir = Path(platforms_dir)

    file_map: dict[str, str] = {}
    for yaml_file in sorted(platforms_dir.glob("*.yaml")):
        if yaml_file.stem == "template":
            continue
        name = yaml_file.stem
        file_map[name] = yaml_file.name

        # Also register device types as aliases
        try:
            config = yaml.safe_load(yaml_file.read_text()) or {}
            dt = config.get("device_types", [])
            device_names = dt if isinstance(dt, list) else list(dt.keys())
            for device in device_names:
                if device not in file_map:
                    file_map[str(device)] = yaml_file.name
        except yaml.YAMLError:
            continue

    return file_map


def _resolve_platform_file(
    platform: str,
    platforms_dir: Path | None = None,
) -> tuple[Path, str | None]:
    """Resolve a platform name or device alias to a YAML file path.

    Returns:
        Tuple of (yaml_path, device_hint) where device_hint is set if the
        input was a device alias rather than a platform name.
    """
    if platforms_dir is None:
        platforms_dir = _PLATFORMS_DIR

    # Direct match
    direct = platforms_dir / f"{platform}.yaml"
    if direct.exists():
        return direct, None

    # Try alias resolution via file map
    file_map = build_platform_file_map(platforms_dir)
    if platform in file_map:
        yaml_path = platforms_dir / file_map[platform]
        # platform was a device alias — hint it as the device
        device_hint = platform if platform != yaml_path.stem else None
        return yaml_path, device_hint

    available = sorted(
        f.stem for f in platforms_dir.glob("*.yaml") if f.stem != "template"
    )
    raise FileNotFoundError(
        f"Platform config not found for '{platform}'. "
        f"Available platforms: {', '.join(available)}. "
        f"Use template.yaml to create a new platform configuration."
    )


@dataclass
class Tolerance:
    """Numerical comparison tolerance for a single dtype/category."""

    rtol: float = 1e-5
    atol: float = 1e-8
    exact: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Tolerance:
        return cls(
            rtol=float(d.get("rtol", 1e-5)),
            atol=float(d.get("atol", 1e-8)),
            exact=bool(d.get("exact", False)),
        )


@dataclass
class DeviceType:
    """Hardware device descriptor."""

    name: str
    memory_gb: int = 0
    compute_capability: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class TestFilter:
    """Include/exclude patterns for unit tests."""

    include: str | list[str] = "*"
    exclude: list[str] = field(default_factory=list)


@dataclass
class FunctionalTests:
    """Functional test cases grouped by task and model."""

    # task -> model -> list of case names
    # e.g. {"inference": {"qwen3": ["4b_tp2"]}, "serve": {"qwen2_5": ["0.5b"]}}
    tests: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    def get_cases(
        self,
        task: str | None = None,
        model: str | None = None,
        test_list: str | None = None,
    ) -> list[dict[str, str]]:
        """Return a flat list of {task, model, case} dicts, optionally filtered.

        Args:
            task: Filter by task name.
            model: Filter by model name.
            test_list: Comma-separated list of case names to include.
        """
        allowed = {t.strip() for t in test_list.split(",")} if test_list else None
        cases: list[dict[str, str]] = []
        for t, models in self.tests.items():
            if task and t != task:
                continue
            for m, case_list in models.items():
                if model and m != model:
                    continue
                for c in case_list:
                    if allowed and c not in allowed:
                        continue
                    cases.append({"task": t, "model": m, "case": c})
        return cases


@dataclass
class PlatformConfig:
    """Full platform configuration loaded from YAML."""

    platform: str
    vendor: str
    device_types: dict[str, DeviceType]
    tolerance: dict[str, dict[str, Tolerance]]
    device_overrides: dict[str, Any]
    unsupported_features: list[str]
    env_defaults: dict[str, str]
    # Per-device test configurations
    device_tests: dict[str, dict[str, Any]]

    # --- Active device (set after load) ---
    device: str = ""

    @classmethod
    def load(
        cls,
        platform: str,
        device: str | None = None,
        platforms_dir: Path | None = None,
    ) -> PlatformConfig:
        """Load platform config from ``tests/platforms/<platform>.yaml``.

        Supports device-alias resolution: passing ``"a100"`` will resolve
        to the ``cuda`` platform config and set the active device to ``a100``.

        If *device* is ``None``, the first device type is used (or the alias
        if *platform* was given as a device alias).
        """
        config_path, device_hint = _resolve_platform_file(platform, platforms_dir)
        if device is None and device_hint is not None:
            device = device_hint

        with open(config_path) as f:
            raw = yaml.safe_load(f)

        # Parse device types (top-level structured section)
        # Supports both dict format {name: {info}} and list format [name, ...]
        device_types: dict[str, DeviceType] = {}
        raw_dt = raw.get("device_types") or {}
        if isinstance(raw_dt, dict):
            for name, info in raw_dt.items():
                if isinstance(info, dict):
                    device_types[str(name)] = DeviceType(
                        name=str(name),
                        memory_gb=info.get("memory_gb", 0),
                        compute_capability=info.get("compute_capability", ""),
                        tags=info.get("tags", []),
                    )
        elif isinstance(raw_dt, list):
            # Simple list of device names — look for detailed config as top-level keys
            for name in raw_dt:
                name = str(name)
                device_types[name] = DeviceType(name=name)

        # Parse tolerance
        tolerance: dict[str, dict[str, Tolerance]] = {}
        for category, dtypes in (raw.get("tolerance") or {}).items():
            tolerance[category] = {}
            for dtype_name, tol_dict in dtypes.items():
                tolerance[category][dtype_name] = Tolerance.from_dict(tol_dict)

        # Parse per-device test configs (top-level keys that match device names)
        device_tests: dict[str, dict[str, Any]] = {}
        for dt_name in device_types:
            if dt_name in raw and isinstance(raw[dt_name], dict):
                device_tests[dt_name] = raw[dt_name]

        # Resolve active device
        active_device = device or next(iter(device_types), "")

        config = cls(
            platform=raw.get("platform", platform),
            vendor=raw.get("vendor", ""),
            device_types=device_types,
            tolerance=tolerance,
            device_overrides=raw.get("device_overrides") or {},
            unsupported_features=raw.get("unsupported_features") or [],
            env_defaults=raw.get("env_defaults") or {},
            device_tests=device_tests,
            device=active_device,
        )
        return config

    def get_tolerance(
        self,
        category: str = "inference",
        dtype: str = "default",
    ) -> Tolerance:
        """Look up tolerance for a category/dtype, falling back to defaults.

        Resolution order:
        1. device_overrides[device].tolerance[category][dtype]
        2. tolerance[category][dtype]
        3. tolerance[category]["default"]
        4. Tolerance() (library defaults)
        """
        # Check device overrides first
        dev_override = self.device_overrides.get(self.device, {})
        if isinstance(dev_override, dict):
            dev_tol = dev_override.get("tolerance", {}).get(category, {})
            if dtype in dev_tol:
                return Tolerance.from_dict(dev_tol[dtype])
            if "default" in dev_tol:
                return Tolerance.from_dict(dev_tol["default"])

        # Platform-level tolerance
        cat_tol = self.tolerance.get(category, {})
        if dtype in cat_tol:
            return cat_tol[dtype]
        if "default" in cat_tol:
            return cat_tol["default"]

        return Tolerance()

    def get_e2e_tests(self) -> FunctionalTests:
        """Return e2e test configuration (inference/serving) for the active device."""
        dt = self.device_tests.get(self.device, {})
        e2e_raw = dt.get("tests", {}).get("e2e", {})
        return FunctionalTests(tests=e2e_raw)

    def get_functional_filter(self) -> TestFilter:
        """Return functional test include/exclude for the active device."""
        dt = self.device_tests.get(self.device, {})
        func_raw = dt.get("tests", {}).get("functional", {})
        return TestFilter(
            include=func_raw.get("include", "*"),
            exclude=func_raw.get("exclude", []),
        )

    def get_unit_filter(self) -> TestFilter:
        """Return unit test include/exclude for the active device."""
        dt = self.device_tests.get(self.device, {})
        unit_raw = dt.get("tests", {}).get("unit", {})
        return TestFilter(
            include=unit_raw.get("include", "*"),
            exclude=unit_raw.get("exclude", []),
        )

    def should_skip_model(self, model_name: str) -> bool:
        """Return True if model_name matches any unsupported feature."""
        return any(feat in model_name for feat in self.unsupported_features)

    def apply_env_defaults(self) -> None:
        """Set environment variables from env_defaults (if not already set)."""
        for key, value in self.env_defaults.items():
            os.environ.setdefault(key, str(value))
