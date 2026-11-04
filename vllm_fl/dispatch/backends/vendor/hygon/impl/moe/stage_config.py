# Copyright (c) 2026 BAAI. All rights reserved.

"""Resolve Hygon MoE configurations for the two GEMM stages."""

from collections.abc import Mapping
from typing import Any

_REQUIRED_CONFIG_KEYS = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
)


def _validate_stage_config(stage: str, config: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in _REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(
            f"MoE {stage} config is missing required keys: {', '.join(missing)}"
        )
    return config if isinstance(config, dict) else dict(config)


def resolve_moe_stage_configs(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a legacy flat config or separate GEMM1/GEMM2 configs."""
    has_gemm1 = "gemm1" in config
    has_gemm2 = "gemm2" in config

    if not has_gemm1 and not has_gemm2:
        flat_config = _validate_stage_config("shared", config)
        return flat_config, flat_config

    if not has_gemm1 or not has_gemm2:
        missing_stage = "gemm1" if not has_gemm1 else "gemm2"
        raise ValueError(
            "Stage-specific MoE config must define both gemm1 and gemm2; "
            f"missing {missing_stage}"
        )

    shared = config.get("shared", {})
    gemm1 = config["gemm1"]
    gemm2 = config["gemm2"]
    if not isinstance(shared, Mapping):
        raise TypeError("MoE shared config must be a mapping")
    if not isinstance(gemm1, Mapping):
        raise TypeError("MoE gemm1 config must be a mapping")
    if not isinstance(gemm2, Mapping):
        raise TypeError("MoE gemm2 config must be a mapping")

    gemm1_config = {**shared, **gemm1}
    gemm2_config = {**shared, **gemm2}
    return (
        _validate_stage_config("gemm1", gemm1_config),
        _validate_stage_config("gemm2", gemm2_config),
    )


def requires_separate_expert_assignment(
    gemm1_config: Mapping[str, Any],
    gemm2_config: Mapping[str, Any],
) -> bool:
    """Return whether GEMM2 needs metadata aligned to a different M tile."""
    return gemm1_config["BLOCK_SIZE_M"] != gemm2_config["BLOCK_SIZE_M"]
