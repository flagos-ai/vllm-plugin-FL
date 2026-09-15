# Copyright (c) 2026 BAAI. All rights reserved.

"""Lazy loader and narrow wrappers for optional T-Head PPU extensions.

The externally provisioned binaries are loaded during T-Head platform
initialization, before portable fallback schemas can claim the same operators.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
from threading import Lock
from types import ModuleType

import torch


_DEFAULT_LIB_DIR = Path(__file__).resolve().parent.parent / "lib"
_LIB_DIR_ENV = "VLLM_FL_THEAD_NATIVE_LIB_DIR"
_FILES = {
    "cache": "_C_stable_libtorch.abi3.so",
    "core": "_C.abi3.so",
    "moe": "_moe_C.abi3.so",
}
_LOADED: set[str] = set()
_LOAD_LOCK = Lock()


def _native_library_directory() -> Path:
    """Return the configured native-library directory for this process."""
    configured = os.environ.get(_LIB_DIR_ENV)
    if configured is None:
        return _DEFAULT_LIB_DIR
    directory = Path(configured)
    if not directory.is_absolute():
        raise ValueError(f"{_LIB_DIR_ENV} must be an absolute path: {configured!r}")
    return directory


class NativeExtensionBundleMissingError(RuntimeError):
    """Raised when the optional T-Head native bundle is not installed."""

    def __init__(self, missing_paths: list[Path]) -> None:
        self.missing_paths = tuple(missing_paths)
        super().__init__(
            "Incomplete T-Head native extension bundle: "
            f"{[str(path) for path in self.missing_paths]}"
        )


def load_native_extension(component: str) -> None:
    """Load one pinned PPU extension exactly once in the current process."""
    if component not in _FILES:
        raise ValueError(f"Unknown T-Head native extension component: {component}")
    if component in _LOADED:
        return
    with _LOAD_LOCK:
        if component in _LOADED:
            return
        path = _native_library_directory() / _FILES[component]
        if not path.is_file():
            raise NativeExtensionBundleMissingError([path])
        torch.ops.load_library(str(path))
        _LOADED.add(component)


def load_all_native_extensions() -> None:
    """Load the complete, mutually compatible bundle before fallback schemas."""
    library_directory = _native_library_directory()
    missing = [
        library_directory / name
        for name in _FILES.values()
        if not (library_directory / name).is_file()
    ]
    if missing:
        raise NativeExtensionBundleMissingError(missing)
    # This is the import order used by the pinned reference vLLM wheel.
    for component in ("cache", "core", "moe"):
        load_native_extension(component)
    # The shared objects export Torch registrations rather than Python symbols.
    # Publish empty module shims, matching their real extension-module surface,
    # so vLLM capability probes do not mistake the already-loaded bundle for a
    # missing extension and install portable fallback kernels over it.
    for module_name in (
        "vllm._C_stable_libtorch",
        "vllm._C",
        "vllm._moe_C",
    ):
        sys.modules.setdefault(module_name, ModuleType(module_name))


def native_extension_is_available(component: str, op_name: str) -> bool:
    """Return whether the binary loads and registers the requested CUDA kernel."""
    try:
        load_native_extension(component)
        table = torch._C._dispatch_dump_table(op_name)
        return any(line.startswith("CUDA:") for line in table.splitlines())
    except Exception:
        return False


def concat_mla_q(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    out: torch.Tensor,
) -> None:
    load_native_extension("cache")
    torch.ops._C_cache_ops.concat_mla_q(ql_nope, q_pe, out)


def mla_kv_cache_update(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
) -> None:
    load_native_extension("cache")
    slots = slot_mapping.flatten()
    torch.ops._C_cache_ops.concat_and_cache_mla(
        kv_c_normed[: slots.shape[0]],
        k_pe.squeeze(1)[: slots.shape[0]],
        kv_cache,
        slots,
        kv_cache_dtype,
        k_scale,
    )


def bf16_indexer_cache_write(
    keys: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    load_native_extension("cache")
    slots = slot_mapping.flatten()
    torch.ops._C_cache_ops.concat_and_cache_mla(
        keys[: slots.shape[0]],
        keys[: slots.shape[0], :0],
        cache,
        slots,
        "auto",
        keys[:1, :1],
    )


def dynamic_per_token_quant_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    load_native_extension("core")
    if x.ndim != 2 or not x.is_floating_point():
        raise ValueError("T-Head dynamic INT8 quantization expects floating [M,K]")
    x = x.contiguous()
    output = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    torch.ops._C.dynamic_scaled_int8_quant(output, x, scale, None)
    return output, scale


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the PPU-native token alignment with the vLLM allocation contract."""
    load_native_extension("moe")
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = (
            (max_num_tokens_padded + block_size - 1) // block_size * block_size
        )
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.empty(
        (math.ceil(max_num_tokens_padded / block_size),),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_tokens_post_pad = torch.empty(
        (1,), dtype=torch.int32, device=topk_ids.device
    )
    native_expert_map = expert_map if ignore_invalid_experts else None
    torch.ops._moe_C.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        native_expert_map,
    )
    if expert_map is not None and not ignore_invalid_experts:
        expert_ids = expert_map[expert_ids]
    return sorted_ids, expert_ids, num_tokens_post_pad


def moe_sum(inp: torch.Tensor, out: torch.Tensor) -> None:
    load_native_extension("moe")
    torch.ops._moe_C.moe_sum(inp, out)


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    load_native_extension("moe")
    torch.ops._moe_C.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        None,
    )
    return topk_weights, topk_indices


def grouped_topk(
    scores: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    load_native_extension("moe")
    return torch.ops._moe_C.grouped_topk(
        scores,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )


__all__ = [
    "bf16_indexer_cache_write",
    "concat_mla_q",
    "dynamic_per_token_quant_int8",
    "load_all_native_extensions",
    "load_native_extension",
    "mla_kv_cache_update",
    "moe_align_block_size",
    "moe_sum",
    "native_extension_is_available",
    "grouped_topk",
    "topk_softmax",
]
