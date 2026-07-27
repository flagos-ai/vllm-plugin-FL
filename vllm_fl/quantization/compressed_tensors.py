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


@dataclass(frozen=True)
class W8A8DynamicTokenScheme:
    weight_num_bits: int
    weight_type: str
    weight_strategy: str
    weight_symmetric: bool
    weight_dynamic: bool
    weight_group_size: int | None
    input_num_bits: int
    input_type: str
    input_strategy: str
    input_symmetric: bool
    input_dynamic: bool

    @classmethod
    def from_group(cls, group: dict[str, Any]) -> W8A8DynamicTokenScheme:
        weights = group.get("weights") or {}
        inputs = group.get("input_activations") or {}
        return cls(
            weight_num_bits=int(weights.get("num_bits", 0)),
            weight_type=str(weights.get("type", "")),
            weight_strategy=str(weights.get("strategy", "")),
            weight_symmetric=bool(weights.get("symmetric", False)),
            weight_dynamic=bool(weights.get("dynamic", False)),
            weight_group_size=weights.get("group_size"),
            input_num_bits=int(inputs.get("num_bits", 0)),
            input_type=str(inputs.get("type", "")),
            input_strategy=str(inputs.get("strategy", "")),
            input_symmetric=bool(inputs.get("symmetric", False)),
            input_dynamic=bool(inputs.get("dynamic", False)),
        )

    def validate(self) -> None:
        if self.weight_num_bits != 8 or self.input_num_bits != 8:
            raise ValueError("W8A8 requires 8-bit weights and activations")
        if self.weight_type != "int" or self.input_type != "int":
            raise ValueError("W8A8 requires integer weights and activations")
        if self.weight_strategy != "channel":
            raise ValueError("W8A8 requires per-channel weights")
        if self.weight_dynamic:
            raise ValueError("W8A8 checkpoint weights must be statically quantized")
        if self.weight_group_size is not None:
            raise ValueError("Per-channel W8A8 weights must not set group_size")
        if self.input_strategy != "token":
            raise ValueError("W8A8 requires per-token input activations")
        if not self.weight_symmetric or not self.input_symmetric:
            raise ValueError("FL W8A8 currently requires symmetric quantization")
        if not self.input_dynamic:
            raise ValueError("W8A8 per-token input quantization must be dynamic")


def validate_compressed_tensors_w8a8_config(
    config: dict[str, Any],
) -> list[W8A8DynamicTokenScheme]:
    """Validate the canonical compressed-tensors dynamic-token W8A8 subset."""
    if config.get("quant_method") != "compressed-tensors":
        raise ValueError("quant_method must be 'compressed-tensors'")
    if config.get("format") != "int-quantized":
        raise ValueError("W8A8 requires compressed-tensors format 'int-quantized'")
    groups = config.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("compressed-tensors config_groups must be a non-empty mapping")

    schemes: list[W8A8DynamicTokenScheme] = []
    for name, group in groups.items():
        if not isinstance(group, dict) or not group.get("targets"):
            raise ValueError(f"config group {name!r} must declare targets")
        scheme = W8A8DynamicTokenScheme.from_group(group)
        scheme.validate()
        schemes.append(scheme)
    return schemes


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
    """Configure compressed-tensors runtime selection for the FL platform.

    W8A8 uses FlagGems and is registered independently. WNA16 remains a no-op
    unless the plugin-local MoE operator is built, keeping vLLM's native
    Marlin/generic selection untouched otherwise.
    """
    report = inspect_vllm_compressed_tensors_api()

    from vllm_fl.utils import is_oot_enabled

    if is_oot_enabled():
        try:
            from vllm_fl.quantization.w8a8.moe import (
                install_fl_w8a8_moe_selector,
            )

            install_fl_w8a8_moe_selector()
        except (ImportError, AttributeError, OSError, RuntimeError) as exc:
            logger.warning(
                "Could not configure FL compressed-tensors W8A8 MoE: %s",
                exc,
            )

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
