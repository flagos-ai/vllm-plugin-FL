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
"""Compatibility glue for standard compressed-tensors WNA16 checkpoints.

The checkpoint contract remains owned by compressed-tensors. This module only
adapts vLLM's runtime implementation to the FL out-of-tree platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_LINEAR_WNA16_MODULES = (
    "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
    "compressed_tensors_wNa16",
    # Keep working if upstream normalizes the historical mixed-case filename.
    "vllm.model_executor.layers.quantization.compressed_tensors.schemes."
    "compressed_tensors_wna16",
)
_MOE_WNA16_MODULES = (
    "vllm.model_executor.layers.quantization.compressed_tensors."
    "compressed_tensors_moe.compressed_tensors_moe_wna16",
)


@dataclass(frozen=True)
class WNA16Scheme:
    num_bits: int
    group_size: int | None
    symmetric: bool
    strategy: str
    has_activation_quantization: bool

    @classmethod
    def from_group(cls, group: dict[str, Any]) -> WNA16Scheme:
        weights = group.get("weights") or {}
        return cls(
            num_bits=int(weights.get("num_bits", 0)),
            group_size=weights.get("group_size"),
            symmetric=bool(weights.get("symmetric", False)),
            strategy=str(weights.get("strategy", "")),
            has_activation_quantization=group.get("input_activations") is not None,
        )

    def validate(self) -> None:
        if self.num_bits not in {4, 8}:
            raise ValueError(
                f"WNA16 supports 4-bit or 8-bit weights, got {self.num_bits}"
            )
        if self.strategy not in {"group", "channel"}:
            raise ValueError(
                f"WNA16 requires group or channel strategy, got {self.strategy!r}"
            )
        if self.strategy == "group" and (
            not isinstance(self.group_size, int) or self.group_size <= 0
        ):
            raise ValueError("Group-wise WNA16 requires a positive group_size")
        if not self.symmetric:
            raise ValueError("FL WNA16 currently requires symmetric weights")
        if self.has_activation_quantization:
            raise ValueError("WNA16 is weight-only; input_activations must be omitted")


def validate_compressed_tensors_wna16_config(
    config: dict[str, Any],
) -> list[WNA16Scheme]:
    """Validate the standard subset consumed by the FL WNA16 runtime."""
    if config.get("quant_method") != "compressed-tensors":
        raise ValueError("quant_method must be 'compressed-tensors'")
    if config.get("format") != "pack-quantized":
        raise ValueError("WNA16 requires compressed-tensors format 'pack-quantized'")
    groups = config.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("compressed-tensors config_groups must be a non-empty mapping")
    schemes: list[WNA16Scheme] = []
    for name, group in groups.items():
        if not isinstance(group, dict) or not group.get("targets"):
            raise ValueError(f"config group {name!r} must declare targets")
        scheme = WNA16Scheme.from_group(group)
        scheme.validate()
        schemes.append(scheme)
    return schemes


@dataclass(frozen=True)
class CompatibilityReport:
    vllm_version: str
    linear_wna16: bool
    moe_wna16: bool
    details: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return self.linear_wna16 and self.moe_wna16


def _class_is_available(
    module_names: tuple[str, ...], class_name: str
) -> tuple[
    bool,
    str | None,
]:
    """Find an upstream class without pinning its method implementation.

    The adapters intentionally rely on vLLM's public scheme classes rather
    than a frozen list of methods. Methods may move to a base class or be
    refactored between vLLM releases while the integration point remains
    compatible.
    """
    failures: list[str] = []
    for module_name in module_names:
        try:
            candidate = getattr(import_module(module_name), class_name)
        except (ImportError, AttributeError, OSError, RuntimeError) as exc:
            failures.append(f"{module_name}: {exc}")
            continue
        if isinstance(candidate, type):
            return True, None
        failures.append(f"{module_name}: {class_name} is not a class")
    return False, "; ".join(failures)


def inspect_vllm_compressed_tensors_api() -> CompatibilityReport:
    """Probe the narrow upstream API surface used by this plugin."""
    try:
        vllm_version = version("vllm")
    except PackageNotFoundError:
        vllm_version = "unknown"

    details: list[str] = []
    linear_wna16, linear_error = _class_is_available(
        _LINEAR_WNA16_MODULES,
        "CompressedTensorsWNA16",
    )
    if linear_error:
        details.append(f"linear WNA16 unavailable: {linear_error}")

    moe_wna16, moe_error = _class_is_available(
        _MOE_WNA16_MODULES,
        "CompressedTensorsWNA16MoEMethod",
    )
    if moe_error:
        details.append(f"MoE WNA16 unavailable: {moe_error}")

    return CompatibilityReport(
        vllm_version=vllm_version,
        linear_wna16=linear_wna16,
        moe_wna16=moe_wna16,
        details=tuple(details),
    )


def register_compressed_tensors_oot() -> CompatibilityReport:
    """Configure upstream WNA16 runtime selection for the FL platform.

    No-op unless the plugin-local WNA16 MoE operator is actually built.
    This keeps the upstream vLLM path (Marlin on CUDA, generic elsewhere)
    fully untouched until the FL kernel is available.
    """
    report = inspect_vllm_compressed_tensors_api()

    from vllm_fl.quantization.wna16.kernels import is_wna16_moe_available

    if not is_wna16_moe_available():
        return report

    # Linear registration is independent and handled by
    # register_fl_wna16_linear_kernel. Do not disable the MoE adapter merely
    # because a vLLM release moved or removed its linear WNA16 scheme.
    if not report.moe_wna16:
        logger.warning(
            "compressed-tensors WNA16 MoE is unavailable for vLLM %s: %s",
            report.vllm_version,
            "; ".join(report.details),
        )
        return report

    from vllm_fl.utils import is_oot_enabled

    if is_oot_enabled():
        try:
            from vllm_fl.quantization.marlin import configure_wna16_moe_backend
            from vllm_fl.quantization.wna16.moe import (
                install_fl_wna16_moe_method,
            )

            if not install_fl_wna16_moe_method():
                logger.warning(
                    "FL WNA16 MoE operator disappeared during registration; "
                    "leaving vLLM's upstream backend selection unchanged"
                )
                return report
            backend = configure_wna16_moe_backend()
            logger.info(
                "compressed-tensors WNA16 MoE backend for FL: %s",
                backend,
            )
        except (ImportError, AttributeError, OSError, RuntimeError) as exc:
            logger.warning(
                "Could not configure FL compressed-tensors WNA16 MoE; "
                "vLLM's upstream backend selection remains unchanged: %s",
                exc,
            )
    return report
