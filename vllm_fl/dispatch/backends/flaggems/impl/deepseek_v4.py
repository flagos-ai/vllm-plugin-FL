# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4 contract mappings to public FlagGems-vllm library APIs."""

import importlib
import inspect

from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

_PUBLIC_APIS = {
    "deepseek_v4_compressor": "compressor",
    "deepseek_v4_sparse_attention": "sparse_attn_sharedkv",
    "deepseek_v4_indexer": "quant_lightning_indexer",
    "paged_cache_write": "paged_scatter_triton",
}
_RESOLVED_APIS = {}


def initialize_deepseek_v4_api(op_name):
    """Validate and load a scoped API, including its mutable graph boundary."""
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "ascend":
        raise RuntimeError("DeepSeek V4 public library APIs require Ascend")
    if op_name in _RESOLVED_APIS:
        return _RESOLVED_APIS[op_name]
    # Public import initializes compressor_paged and its fake implementation.
    library = importlib.import_module("flaggems_vllm")
    public_name = _PUBLIC_APIS[op_name]
    implementation = getattr(library, public_name, None)
    if not callable(implementation):
        raise RuntimeError(f"FlagGems-vllm requires callable public API {public_name}")
    args, kwargs = _CAPABILITY_CALLS[op_name]
    try:
        signature = inspect.signature(implementation)
        # Legacy sparse wrappers accept arbitrary kwargs but lack this path.
        required = _EXPLICIT_PARAMETERS[op_name]
        if any(name not in signature.parameters for name in required):
            raise TypeError("Public API lacks explicit device metadata parameters")
        signature.bind(*args, **kwargs)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Incompatible FlagGems-vllm public API {public_name}"
        ) from exc
    _RESOLVED_APIS[op_name] = implementation
    return implementation


def is_available(op_name):
    try:
        initialize_deepseek_v4_api(op_name)
    except (ImportError, RuntimeError):
        return False
    return True


def deepseek_v4_compressor(
    hidden_states,
    wkv,
    wgate,
    ape,
    state_cache,
    positions,
    slot_mapping,
    state_block_table,
    output_request_ids,
    output_positions,
    rms_weight,
    cos_sin_cache,
    compress_ratio=4,
    rms_eps=1e-6,
):
    return _RESOLVED_APIS["deepseek_v4_compressor"](
        hidden_states,
        wkv,
        wgate,
        state_cache,
        ape,
        rms_weight,
        state_block_table=state_block_table,
        cmp_ratio=compress_ratio,
        coff=2 if compress_ratio == 4 else 1,
        rope_head_dim=cos_sin_cache.shape[1],
        norm_eps=rms_eps,
        positions=positions,
        slot_mapping=slot_mapping,
        output_request_ids=output_request_ids,
        output_positions=output_positions,
        cos_sin_cache=cos_sin_cache,
    )


def deepseek_v4_sparse_attention(
    query,
    swa_cache,
    compact_cache,
    swa_block_table,
    compact_block_table,
    request_ids,
    positions,
    swa_lengths,
    compact_lengths,
    compact_indices,
    window_size,
    compress_ratio=4,
    scale=None,
    attn_sink=None,
    value_dim=None,
):
    output, _ = _RESOLVED_APIS["deepseek_v4_sparse_attention"](
        query,
        ori_kv=swa_cache,
        cmp_kv=compact_cache,
        ori_block_table=swa_block_table,
        cmp_block_table=compact_block_table,
        request_ids=request_ids,
        positions=positions,
        ori_lengths=swa_lengths,
        cmp_lengths=compact_lengths,
        cmp_sparse_indices=compact_indices,
        ori_win_left=window_size - 1,
        cmp_ratio=0 if compact_cache is None else compress_ratio,
        softmax_scale=scale,
        sinks=attn_sink,
        value_dim=value_dim,
    )
    return output


def deepseek_v4_indexer(
    query,
    key_cache,
    weights,
    block_table,
    request_ids,
    positions,
    key_lengths,
    topk,
    compress_ratio=4,
    query_scale=None,
    key_scale_cache=None,
):
    return _RESOLVED_APIS["deepseek_v4_indexer"](
        query,
        key_cache,
        weights,
        block_table,
        request_ids,
        positions,
        key_lengths,
        topk,
        compress_ratio=compress_ratio,
        query_scale=query_scale,
        key_scale_cache=key_scale_cache,
    )


def paged_cache_write(values, slots, cache):
    """Write unique token slots through the public in-place cache API."""
    if (
        cache.ndim != 3
        or values.ndim != 2
        or values.shape[1] != cache.shape[-1]
        or slots.shape != values.shape[:1]
    ):
        raise ValueError(
            "Expected matching [tokens, dim], [tokens], [pages, block, dim]"
        )
    if not _RESOLVED_APIS["paged_cache_write"](
        cache, slots, values, cache.shape[1]
    ):
        raise NotImplementedError("Unsupported paged cache write contract")


# Bind complete public calls without allocating tensors or executing a kernel.
_CAPABILITY_CALLS = {
    "paged_cache_write": ((None,) * 4, {}),
    "deepseek_v4_compressor": (
        (None,) * 6,
        dict(
            state_block_table=None,
            cmp_ratio=4,
            coff=2,
            rope_head_dim=64,
            norm_eps=1e-6,
            positions=None,
            slot_mapping=None,
            output_request_ids=None,
            output_positions=None,
            cos_sin_cache=None,
        ),
    ),
    "deepseek_v4_sparse_attention": (
        (None,),
        dict(
            ori_kv=None,
            cmp_kv=None,
            ori_block_table=None,
            cmp_block_table=None,
            request_ids=None,
            positions=None,
            ori_lengths=None,
            cmp_lengths=None,
            cmp_sparse_indices=None,
            ori_win_left=127,
            cmp_ratio=4,
            softmax_scale=None,
            sinks=None,
            value_dim=None,
        ),
    ),
    "deepseek_v4_indexer": (
        (None,) * 8,
        dict(compress_ratio=4, query_scale=None, key_scale_cache=None),
    ),
}


_EXPLICIT_PARAMETERS = {
    "paged_cache_write": ("cache", "slots", "values", "block_size"),
    "deepseek_v4_compressor": (
        "positions",
        "slot_mapping",
        "output_request_ids",
        "output_positions",
        "cos_sin_cache",
        "state_cache",
        "ape",
    ),
    "deepseek_v4_sparse_attention": (
        "request_ids",
        "positions",
        "ori_lengths",
        "cmp_lengths",
        "value_dim",
    ),
    "deepseek_v4_indexer": (
        "query",
        "key_cache",
        "weights",
        "block_table",
        "request_ids",
        "positions",
        "key_lengths",
        "topk",
        "compress_ratio",
        "query_scale",
        "key_scale_cache",
    ),
}


def register_deepseek_v4_operators(registry):
    """Register required library calls independently of optional ATen takeover."""

    def bind(name):
        implementation = globals()[name]

        def call(*args, **kwargs):
            return implementation(*args, **kwargs)

        call.__name__ = name
        call._is_available = lambda: is_available(name)
        return call

    registry.register_many(
        [
            OpImpl(
                name,
                "default.flaggems_vllm",
                BackendImplKind.DEFAULT,
                bind(name),
                priority=BackendPriority.DEFAULT,
                allow_execution_fallback=False,
            )
            for name in _PUBLIC_APIS
        ]
    )
