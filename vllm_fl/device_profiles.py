"""Capability-gated defaults for the validated long-context GQA paths."""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.vllm_fl.device_profiles")

NATIVE_SPARSE_PROFILE = "native-bf16-sparse"
HYBRID_INT8_PROFILE = "hybrid-int8-kv"

_DEFAULTED: dict[str, str] = {}
_ACTIVE: dict[int, dict] = {}

PROFILES = {
    NATIVE_SPARSE_PROFILE: {
        "switches": {
            "VLLM_FL_PREFILL_GATEUP_INT8": True,
            "VLLM_FL_SINGLE_WEIGHT_GATEUP": True,
            "VLLM_FL_NATIVE_SPARSE_DECODE": True,
            "VLLM_FL_NATIVE_SPARSE_PREFILL": True,
        },
        "prefill_gateup_weight_layout": "rowmajor",
        "mm_tuning_buckets": True,
        "mm_large_tiles": True,
        "native_sparse_decode": True,
        "native_sparse_prefill": True,
        "radix_histogram_intrinsic": True,
        "topp_select": True,
    },
    HYBRID_INT8_PROFILE: {
        "switches": {
            "VLLM_FL_GQA_HYBRID": True,
            "VLLM_FL_GQA_HYBRID_PREFILL_LPT": True,
            "VLLM_FL_GQA_HYBRID_PREFILL_REBLOCK": True,
            "VLLM_FL_GQA_HYBRID_PREFILL_WIDE": True,
            "VLLM_FL_GQA_HYBRID_PREFILL_FIXED_ANCHOR": True,
            "VLLM_FL_KV_INT8": True,
            "VLLM_FL_PREFILL_GATEUP_INT8": True,
        },
        "mixed_attention": {
            "int8": {
                "lpt": True,
                "scratch_tokens": 65536,
                "table": {
                    "m:d2:c1:q4": {"mode": "separate", "tile": "reblock"},
                    "m:d3:c1:q4": {"mode": "separate", "tile": "reblock"},
                    "m:d3:c0:q4": {"mode": "separate", "tile": "reblock"},
                    "m:d0:c0:q1": {"mode": "separate", "tile": "p64x32"},
                    "p:c0:q3": {"mode": "fused", "tile": "p64x32"},
                    "m:d0:c0:q4": {
                        "mode": "separate",
                        "tile": "reblock",
                        "fused_decode_max": 2,
                    },
                    "m:d1:c0:q4": {"mode": "separate", "tile": "reblock"},
                    "m:d2:c0:q4": {"mode": "separate", "tile": "reblock"},
                    "m:d0:c1:q4": {
                        "mode": "separate",
                        "tile": "reblock",
                        "fused_decode_max": 2,
                    },
                    "m:d1:c1:q4": {"mode": "separate", "tile": "reblock"},
                    "m:d3:c0:q1": {"mode": "separate", "tile": "p64x32"},
                    "m:d3:c0:q3": {"mode": "separate", "tile": "p64x32"},
                },
            },
        },
        "sampler_row_parallel": 4,
        "int8_decode": {
            "variant": 8,
            "tile": 32,
            "num_warps": 1,
            "num_stages": 1,
            "table": (
                (1, 32),
                (2, 32),
                (4, 16),
                (8, 16),
                (16, 8),
                (32, 4),
                (64, 16),
                (128, 8),
            ),
        },
        "int8_sparse": True,
    },
}

_VENDOR_PROFILE = {
    "metax": NATIVE_SPARSE_PROFILE,
    "iluvatar": HYBRID_INT8_PROFILE,
}


def _profile_name(index: int) -> str | None:
    from flag_gems.runtime import device

    return _VENDOR_PROFILE.get(device.vendor_name)


def _matches_supported_model(vllm_config) -> bool:
    """Limit automatic defaults to the validated single-device GQA geometry."""
    if vllm_config is None:
        return False
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    cache = vllm_config.cache_config
    try:
        geometry = (
            model.get_num_attention_heads(parallel),
            model.get_num_kv_heads(parallel),
            model.get_head_size(),
        )
    except Exception:
        return False
    return bool(
        geometry == (16, 2, 128)
        and model.dtype == torch.bfloat16
        and parallel.tensor_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and parallel.decode_context_parallel_size == 1
        and not parallel.use_ubatching
        and cache.cache_dtype == "auto"
        and (cache.block_size or 16) == 16
    )


def select(index: int, vllm_config) -> dict | None:
    """Return a profile only when hardware and model capabilities both match."""
    if not _matches_supported_model(vllm_config):
        return None
    name = _profile_name(index)
    return PROFILES.get(name) if name is not None else None


def lookup(index: int) -> dict | None:
    """Return the profile activated for this worker, if any."""
    return _ACTIVE.get(int(index))


def prefill_gateup_weight_layout(index: int) -> str:
    """Use the established packed layout unless an active profile opts in."""
    return (lookup(index) or {}).get("prefill_gateup_weight_layout", "packed")


def apply_defaults(index: int, vllm_config=None) -> None:
    """Activate a matching profile and preserve every explicit user override."""
    index = int(index)
    if vllm_config is not None:
        profile = select(index, vllm_config)
        if profile is None:
            _ACTIVE.pop(index, None)
            return
        _ACTIVE[index] = profile
    else:
        profile = lookup(index)
    if profile is None:
        return

    applied = []
    for name, default in profile["switches"].items():
        if os.environ.get(name, "auto").strip().lower() not in ("", "auto"):
            continue
        if name == "VLLM_FL_KV_INT8":
            hybrid = os.environ.get("VLLM_FL_GQA_HYBRID", "auto").strip().lower()
            if hybrid not in ("", "auto", "1", "true", "yes"):
                default = False
        os.environ[name] = "1" if default else "0"
        _DEFAULTED[name] = os.environ[name]
        applied.append(f"{name}={os.environ[name]}")
    if applied:
        logger.info("Capability-gated defaults: %s", ", ".join(applied))


def was_defaulted(name: str) -> bool:
    """Distinguish worker defaults from explicit user choices."""
    return name in _DEFAULTED and _DEFAULTED[name] == os.environ.get(name)
