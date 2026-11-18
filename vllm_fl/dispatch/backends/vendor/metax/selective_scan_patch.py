"""Compatibility patch for MetaX legacy selective_scan_fwd schema."""

from __future__ import annotations

import threading
from typing import Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_patch_lock = threading.Lock()
_patch_applied = False
_block_size_warned = False


def _get_selective_scan_schema() -> Optional[str]:
    try:
        op = torch.ops._C.selective_scan_fwd
    except AttributeError:
        return None

    schemas = getattr(op, "_schemas", None)
    if schemas:
        schema = schemas.get("")
        if schema is not None:
            return str(schema)

    get_schema = getattr(torch._C, "_get_schema", None)
    if get_schema is None:
        return None

    try:
        return str(get_schema("_C::selective_scan_fwd", ""))
    except Exception:
        return None


def _is_legacy_schema(schema: str) -> bool:
    # vLLM 0.13.0 calls the 18-arg variant with block metadata. Legacy MetaX
    # kernels only register the older 14-arg schema.
    return "block_size" not in schema and "initial_state_idx" not in schema


def _legacy_selective_scan_fwd_adapter(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D_: torch.Tensor | None,
    z_: torch.Tensor | None,
    delta_bias_: torch.Tensor | None,
    delta_softplus: bool,
    query_start_loc: torch.Tensor | None,
    cache_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    ssm_states: torch.Tensor,
    pad_slot_id: int,
    block_size: int = 1024,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
):
    global _block_size_warned

    if (
        block_idx_first_scheduled_token is not None
        or block_idx_last_scheduled_token is not None
        or initial_state_idx is not None
    ):
        raise RuntimeError(
            "MetaX selective_scan_fwd compatibility patch only supports the "
            "legacy 14-arg kernel without prefix-caching block metadata."
        )

    if block_size != 1024 and not _block_size_warned:
        logger.warning(
            "Using MetaX selective_scan_fwd compatibility patch with legacy "
            "14-arg kernel; block_size=%s will be ignored.",
            block_size,
        )
        _block_size_warned = True

    torch.ops._C.selective_scan_fwd(
        u,
        delta,
        A,
        B,
        C,
        D_,
        z_,
        delta_bias_,
        delta_softplus,
        query_start_loc,
        cache_indices,
        has_initial_state,
        ssm_states,
        pad_slot_id,
    )


def apply_metax_selective_scan_patch() -> bool:
    global _patch_applied

    with _patch_lock:
        if _patch_applied:
            return True

        schema = _get_selective_scan_schema()
        if schema is None or not _is_legacy_schema(schema):
            return False

        import vllm._custom_ops as vllm_custom_ops

        vllm_custom_ops.selective_scan_fwd = _legacy_selective_scan_fwd_adapter
        _patch_applied = True
        logger.warning(
            "Applied MetaX selective_scan_fwd compatibility patch for legacy "
            "14-arg schema: %s",
            schema,
        )
        return True
