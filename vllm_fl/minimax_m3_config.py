# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Single, immutable deployment policy. Standard-library only; no framework imports."""

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

ROOT = Path(__file__).resolve().parent
PROFILE_NAME = "minimax_m3_metax"
_DISPATCH_KEYS = (
    "prefer",
    "strict",
    "op_backends",
    "flagos_blacklist",
    "oot_blacklist",
)


def configuration_path():
    value = os.environ.get("VLLM_FL_CONFIG", "").strip()
    if not value:
        raise RuntimeError("MiniMax M3 requires an explicit VLLM_FL_CONFIG")
    return Path(value)


def is_enabled():
    value = os.environ.get("VLLM_FL_CONFIG", "").strip()
    if not value:
        return False
    try:
        data = json.loads(Path(value).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get("model_profile") == PROFILE_NAME


def dispatch_config():
    data = json.loads(configuration_path().read_text())
    if data.get("model_profile") != PROFILE_NAME:
        raise ValueError("The selected configuration is not a MiniMax M3 profile")
    return {k: data[k] for k in _DISPATCH_KEYS}


@dataclass(frozen=True)
class FrameworkPolicy:
    tensor_parallel_size: int
    flagos_norm: bool
    flagos_swiglu: bool
    flagos_msa: bool
    exact_decode_fallback: bool
    vision_chunk_patches: int
    graph_collective_boundary: bool
    max_context: int
    long_prefill_query_lower: int
    long_prefill_query_upper: int
    long_index_min_context: int


@dataclass(frozen=True)
class KernelPolicy:
    safe_topk: bool
    safe_head_tile: bool
    safe_stages: bool
    moe_alignment_tle: bool


@dataclass(frozen=True)
class Profile:
    release: str
    framework: FrameworkPolicy
    kernels: KernelPolicy
    service: Mapping[str, Any]
    dispatch: Mapping[str, Any]
    linear: Mapping[str, Any]
    moe: Mapping[str, Any]
    attention: Mapping[str, Any]
    infrastructure_env: Mapping[str, str]
    fingerprint: str


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    return value


def thaw(value):
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    return value


def typed(cls, values):
    expected = {f.name: f.type for f in fields(cls)}
    if set(values) != set(expected):
        raise ValueError(f"{cls.__name__} fields differ: {set(values) ^ set(expected)}")
    for name, t in expected.items():
        if type(values[name]) is not t:
            raise TypeError(f"{cls.__name__}.{name} requires {t}")
    return cls(**values)


@lru_cache(maxsize=1)
def load():
    payload = configuration_path().read_bytes()
    raw = json.loads(payload)
    if raw.get("model_profile") != PROFILE_NAME:
        raise ValueError("Expected the MiniMax M3 MetaX profile")
    data = dict(raw["minimax_m3"])
    data["dispatch"] = dispatch_config()
    expected = {
        "release",
        "framework",
        "kernels",
        "service",
        "dispatch",
        "linear",
        "moe",
        "attention",
        "infrastructure_env",
    }
    if set(data) != expected:
        raise ValueError(f"Unknown/missing profile sections: {set(data) ^ expected}")
    if data["dispatch"]["flagos_blacklist"] != ["masked_fill"] or data["dispatch"][
        "oot_blacklist"
    ] != ["masked_fill"]:
        raise ValueError("Preserve the validated ATen/OOT blacklists")
    framework = typed(FrameworkPolicy, data["framework"])
    kernels = typed(KernelPolicy, data["kernels"])
    if framework.tensor_parallel_size != 16:
        raise ValueError("Only accepted TP16 is configured")
    if not all(
        (
            framework.flagos_norm,
            framework.flagos_swiglu,
            framework.flagos_msa,
            framework.exact_decode_fallback,
            framework.graph_collective_boundary,
        )
    ):
        raise ValueError("An unvalidated baseline path was requested")
    if (
        not all((kernels.safe_topk, kernels.safe_head_tile, kernels.safe_stages))
        or kernels.moe_alignment_tle
    ):
        raise ValueError(
            "Preserve the accepted safe MSA / four-stage alignment algorithms"
        )
    for group in ("legacy", "small"):
        keys = []
        for row in data["linear"][group]:
            for k in ("M", "N", "K"):
                if type(row[k]) is not int or row[k] <= 0:
                    raise TypeError("Linear dimensions must be positive ints")
            if row["dtype"] not in ("bfloat16", "float32"):
                raise ValueError("Unsupported configured linear dtype")
            keys.append((row["M"], row["N"], row["K"], row["dtype"]))
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate linear shape")
    if len(data["linear"]["legacy"]) != 35 or len(data["linear"]["small"]) != 28:
        raise ValueError("Accepted table cardinality changed")
    if any(type(v) is not str for v in data["infrastructure_env"].values()):
        raise TypeError("Infrastructure env values must be strings")
    if any(k.startswith("M3_") for k in data["infrastructure_env"]):
        raise ValueError("Project optimization env must not be exported")
    return Profile(
        data["release"],
        framework,
        kernels,
        **{
            k: freeze(data[k])
            for k in (
                "service",
                "dispatch",
                "linear",
                "moe",
                "attention",
                "infrastructure_env",
            )
        },
        fingerprint=hashlib.sha256(payload).hexdigest(),
    )
