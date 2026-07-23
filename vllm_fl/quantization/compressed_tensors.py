# Copyright 2026 FlagOS Contributors
"""Compatibility glue for standard compressed-tensors WNA16 checkpoints.

The checkpoint contract remains owned by compressed-tensors. This module only
adapts vLLM's runtime implementation to the FL out-of-tree platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


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


def inspect_vllm_compressed_tensors_api() -> CompatibilityReport:
    """Probe the narrow upstream API surface used by this plugin."""
    try:
        vllm_version = version("vllm")
    except PackageNotFoundError:
        vllm_version = "unknown"

    details: list[str] = []
    linear_wna16 = False
    moe_wna16 = False
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa: E501
            CompressedTensorsWNA16,
        )

        linear_wna16 = all(
            hasattr(CompressedTensorsWNA16, name)
            for name in (
                "create_weights",
                "process_weights_after_loading",
                "apply_weights",
            )
        )
        if not linear_wna16:
            details.append("CompressedTensorsWNA16 API is incomplete")
    except (ImportError, AttributeError, OSError, RuntimeError) as exc:
        details.append(f"linear WNA16 unavailable: {exc}")

    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
            CompressedTensorsWNA16MoEMethod,
        )

        moe_wna16 = all(
            hasattr(CompressedTensorsWNA16MoEMethod, name)
            for name in (
                "create_weights",
                "process_weights_after_loading",
                "get_fused_moe_quant_config",
            )
        )
        if not moe_wna16:
            details.append("CompressedTensorsWNA16MoEMethod API is incomplete")
    except (ImportError, AttributeError, OSError, RuntimeError) as exc:
        details.append(f"MoE WNA16 unavailable: {exc}")

    return CompatibilityReport(
        vllm_version=vllm_version,
        linear_wna16=linear_wna16,
        moe_wna16=moe_wna16,
        details=tuple(details),
    )


def register_compressed_tensors_oot() -> CompatibilityReport:
    """Configure upstream WNA16 runtime selection for the FL platform."""
    report = inspect_vllm_compressed_tensors_api()
    if not report.supported:
        logger.warning(
            "compressed-tensors WNA16 compatibility is incomplete for vLLM %s: %s",
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

            install_fl_wna16_moe_method()
            backend = configure_wna16_moe_backend()
            logger.info(
                "compressed-tensors WNA16 MoE backend for FL: %s",
                backend,
            )
        except (ImportError, AttributeError, OSError, RuntimeError) as exc:
            logger.warning(
                "Could not configure FL compressed-tensors WNA16 MoE: %s",
                exc,
            )
    return report
