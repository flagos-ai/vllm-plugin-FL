# Copyright (c) 2025 BAAI. All rights reserved.
#
# Register torch.ops._C op schemas so that vllm compilation passes can
# reference them for pattern matching even when the native vllm._C extension
# is not compiled for this platform.

import logging

import torch

logger = logging.getLogger(__name__)


# Fallback implementations for query ops
_QUERY_OP_IMPLS = [
    ("cutlass_scaled_mm_supports_fp8", lambda cap: cap >= 89),
    ("cutlass_scaled_mm_supports_block_fp8", lambda cap: cap >= 100),
    ("cutlass_group_gemm_supported", lambda cap: cap >= 90),
    ("cutlass_scaled_mm_supports_fp4", lambda cap: cap >= 100),
    ("weak_ref_tensor", lambda t: t),
    ("get_cuda_view_from_cpu_tensor", lambda t: t),
]


def _concat_and_cache_mla_impl(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    """Pure-torch fallback for _C_cache_ops::concat_and_cache_mla.

    For each token, concatenates kv_c and k_pe and scatters the result
    into kv_cache at positions indicated by slot_mapping.
    """
    num_tokens = slot_mapping.size(0)
    block_size = kv_cache.size(1)

    # slot_mapping may be shorter than kv_c when CUDA-graph padding is used
    kv_c = kv_c[:num_tokens]
    k_pe = k_pe[:num_tokens]

    # Filter out padding slots (slot == -1)
    valid_mask = slot_mapping >= 0
    valid_slots = slot_mapping[valid_mask]
    valid_kv_c = kv_c[valid_mask]
    valid_k_pe = k_pe[valid_mask]

    block_idx = valid_slots // block_size
    block_offset = valid_slots % block_size

    kv_lora_rank = valid_kv_c.size(1)
    # Concatenate along the feature dimension and write into cache
    combined = torch.cat([valid_kv_c, valid_k_pe], dim=-1)
    kv_cache[block_idx, block_offset, :combined.size(-1)] = combined.to(
        kv_cache.dtype
    )


def _gather_and_maybe_dequant_cache_impl(
    src_cache: torch.Tensor,     # [NUM_BLOCKS, BLOCK_SIZE, entry_size]
    dst: torch.Tensor,           # [num_tokens, entry_size]
    block_table: torch.Tensor,   # [BATCH, max_blocks_per_seq]
    cu_seq_lens: torch.Tensor,   # [BATCH+1]
    token_to_seq: torch.Tensor,  # [num_tokens]
    num_tokens: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: "torch.Tensor | None" = None,
) -> None:
    """Pure-torch fallback for _C_cache_ops::gather_and_maybe_dequant_cache.

    Gathers KV cache entries from paged blocks into a flat destination tensor.
    """
    block_size = src_cache.size(1)

    token_ids = torch.arange(num_tokens, device=src_cache.device)
    batch_ids = token_to_seq[:num_tokens].long()
    batch_starts = cu_seq_lens[batch_ids]
    batch_offsets = (token_ids - batch_starts).int()

    if seq_starts is not None:
        batch_offsets = batch_offsets + seq_starts[batch_ids]

    block_table_ids = batch_offsets // block_size
    slot_ids = batch_offsets % block_size

    block_ids = block_table[batch_ids, block_table_ids.long()]
    entries = src_cache[block_ids.long(), slot_ids.long()]
    entry_size = min(entries.size(-1), dst.size(-1))
    dst[:num_tokens, :entry_size] = entries[:, :entry_size].to(dst.dtype)


def _apply_repetition_penalties_impl(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    """Pure-torch fallback for _C::apply_repetition_penalties_."""
    rp = repetition_penalties.unsqueeze(dim=1).repeat(1, logits.size(1))
    penalties = torch.where(prompt_mask | output_mask, rp, 1.0)
    scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
    logits.mul_(scaling)


# Ops that need a CUDA dispatch because vLLM calls them directly
# (not routed through FL's call_op) and only has _C kernel + torch fallback
# gated behind is_cuda checks.
_CUDA_FALLBACK_IMPLS = [
    ("apply_repetition_penalties_", _apply_repetition_penalties_impl),
]


def register_op_schemas():
    """Register _C op schemas if not already present."""
    if getattr(register_op_schemas, "_lib", None) is not None:
        return

    try:
        import vllm._C  # noqa: F401
        return
    except (ImportError, OSError):
        pass

    # Pre-load mcoplib._C (MetaX) so its TORCH_LIBRARY registrations land
    # before our FRAGMENT definitions.  The hasattr check below will then
    # skip any ops already registered by mcoplib, avoiding c10::Error.
    import importlib.util
    if importlib.util.find_spec("mcoplib") is not None:
        try:
            import mcoplib._C  # noqa: F401
        except ImportError:
            logger.warning("Failed to import mcoplib._C")

    from vllm_fl.ops._C_ops_schemas import SCHEMAS as schemas

    if not schemas:
        logger.warning("No op schemas found; torch.compile may not work.")
        return

    lib = torch.library.Library("_C", "FRAGMENT")

    for schema in schemas:
        full_name = schema.split("(")[0]
        op_name = full_name.split(".")[0]
        overload = full_name.split(".")[1] if "." in full_name else "default"
        if hasattr(torch.ops._C, op_name) and hasattr(
            getattr(torch.ops._C, op_name), overload
        ):
            continue
        try:
            lib.define(schema)
        except Exception as e:
            logger.debug("Failed to register _C op schema '%s': %s", full_name, e)

    for name, fn in _QUERY_OP_IMPLS:
        try:
            lib.impl(name, fn, "CompositeImplicitAutograd")
        except Exception:
            pass

    for name, fn in _CUDA_FALLBACK_IMPLS:
        try:
            lib.impl(name, fn, "CUDA")
        except Exception:
            pass

    # Register _C_cache_ops fallbacks for platforms without vllm._C
    _register_cache_ops_fallbacks()

    register_op_schemas._lib = lib


# Schema and fallback for _C_cache_ops namespace (normally compiled in vllm._C)
_CACHE_OPS_SCHEMAS = [
    'concat_and_cache_mla(Tensor kv_c, Tensor k_pe, Tensor(a!) kv_cache, Tensor slot_mapping, str kv_cache_dtype, Tensor scale) -> ()',
    'gather_and_maybe_dequant_cache(Tensor src_cache, Tensor(a!) dst, Tensor block_table, Tensor cu_seq_lens, Tensor token_to_seq, int num_tokens, str kv_cache_dtype, Tensor scale, Tensor? seq_starts) -> ()',
]

_CACHE_OPS_IMPLS = [
    ("concat_and_cache_mla", _concat_and_cache_mla_impl),
    ("gather_and_maybe_dequant_cache", _gather_and_maybe_dequant_cache_impl),
]


def _register_cache_ops_fallbacks():
    """Register _C_cache_ops schemas and Python fallbacks."""
    # Check if ALL required ops are already present
    _required_ops = [name for name, _ in _CACHE_OPS_IMPLS]
    if hasattr(torch.ops, "_C_cache_ops") and all(
        hasattr(torch.ops._C_cache_ops, op) for op in _required_ops
    ):
        return

    try:
        cache_lib = torch.library.Library("_C_cache_ops", "FRAGMENT")
    except Exception as e:
        logger.debug("Failed to create _C_cache_ops library: %s", e)
        return

    for schema in _CACHE_OPS_SCHEMAS:
        op_name = schema.split("(")[0]
        if hasattr(torch.ops, "_C_cache_ops") and hasattr(
            torch.ops._C_cache_ops, op_name
        ):
            continue
        try:
            cache_lib.define(schema)
        except Exception as e:
            logger.debug(
                "Failed to define _C_cache_ops schema '%s': %s", op_name, e
            )

    for name, fn in _CACHE_OPS_IMPLS:
        try:
            cache_lib.impl(name, fn, "CUDA")
        except Exception as e:
            logger.debug(
                "Failed to register _C_cache_ops impl '%s': %s", name, e
            )

    register_op_schemas._cache_lib = cache_lib
