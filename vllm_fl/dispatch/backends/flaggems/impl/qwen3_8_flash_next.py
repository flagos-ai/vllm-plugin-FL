# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 FGV providers and model-specific QSA selection composition.

Numerical kernels remain in FlagGems-vllm. The block-to-token expansion is a
temporary ATen composition in the plugin: its request/causal/tail contract is
model-owned until a reusable library ABI exists. It has no device-to-host reads,
and ordinary ATen dispatch remains under the configured FlagGems policy.
The composition allocates shape-dependent temporary tensors; it is not an
allocation-free provider. Fixed-shape graph capture/replay and the selected
FlagGems ATen implementations still require device validation.
"""

from __future__ import annotations

from functools import partial, wraps

import torch

from vllm_fl.dispatch.capabilities import get_flaggems_vllm_operator

QWEN38_OPERATORS = (
    "qsa_mqa_paged",
    "expand_qsa_block_indices",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
    "qsa_compress_groups_with_ratio",
    "ple_short_conv_",
    "gdn_packed_decode",
    "qwen38_small_projection",
    "qwen38_ple_gate",
    "qwen38_gdn_mixed_qkv",
)

_FGV_DEPENDENCIES = {
    "qsa_mqa_paged": ("qsa_mqa_paged",),
    "expand_qsa_block_indices": (),
    "qsa_select_paged_tokens": ("qsa_mqa_paged", "top_k_per_row_prefill"),
    "qsa_sparse_paged_attention": ("qsa_sparse_paged_attention",),
    "qsa_store_cache_rows": ("qsa_store_cache_rows",),
    "qsa_compress_groups_with_ratio": ("qsa_compress_groups_with_ratio",),
    "ple_short_conv_": ("ple_short_conv_",),
    "gdn_packed_decode": ("fused_recurrent_gated_delta_rule_packed_decode",),
    "gdn_prefill": ("chunk_gated_delta_rule_fwd",),
}


def _fgv(name):
    fn = get_flaggems_vllm_operator(name)
    if not callable(fn):
        raise RuntimeError(f"Required FlagGems-vllm operator {name} is unavailable")
    return fn


def is_available(name):
    if name in {
        "qwen38_small_projection", "qwen38_ple_gate", "qwen38_gdn_mixed_qkv"
    }:
        return True
    import vllm_fl.envs as fl_envs

    return bool(fl_envs.VLLM_FL_USE_FLAGGEMS_VLLM) and all(
        callable(get_flaggems_vllm_operator(dep)) for dep in _FGV_DEPENDENCIES[name]
    )


def qwen38_gdn_mixed_qkv(mixed_qkv):
    """Preserve the upstream QKV view unless a model policy selects otherwise."""
    return mixed_qkv


def qwen38_ple_gate(gate):
    """Original model gate expression, retaining each input-dtype boundary."""
    return torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())


def qwen38_small_projection(x, weight):
    """Retain the original F.linear contract outside model-specific policy."""
    return torch.nn.functional.linear(x, weight)


def qsa_mqa_paged(
    q,
    k_cache,
    page_table,
    token_to_req,
    query_positions,
    sequence_lengths,
    compress_ratio,
    num_columns=None,
    score_scale=None,
):
    # Metadata-derived width is stable during capture/replay. Never ask a vendor
    # wrapper to derive its width from sequence_lengths.max().item().
    if num_columns is None:
        num_columns = page_table.shape[1] * k_cache.shape[1]
    return _fgv("qsa_mqa_paged")(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
        num_columns,
        score_scale,
    )


def qsa_store_cache_rows(cache, slot_mapping, rows):
    return _fgv("qsa_store_cache_rows")(cache, slot_mapping, rows)


def qsa_compress_groups_with_ratio(
    raw_cache,
    raw_block_table,
    token_to_req,
    logical_positions,
    compressed_slots,
    compress_ratio,
    rope_cache=None,
):
    return _fgv("qsa_compress_groups_with_ratio")(
        raw_cache,
        raw_block_table,
        token_to_req,
        logical_positions,
        compressed_slots,
        compress_ratio,
        rope_cache,
    )


def qsa_sparse_paged_attention(
    q,
    k_cache,
    v_cache,
    logical_indices,
    block_table,
    token_to_req,
    softmax_scale=None,
    out=None,
    gate=None,
    split_workspace=None,
):
    return _fgv("qsa_sparse_paged_attention")(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        softmax_scale,
        out,
        gate,
        split_workspace,
    )


def _check_selection_geometry(compress_ratio, token_topk):
    if compress_ratio <= 0 or token_topk <= 0 or token_topk % compress_ratio:
        raise ValueError("token_topk must be positive and divisible by compress_ratio")
    return token_topk // compress_ratio, token_topk + compress_ratio - 1


def _selection_output(rows, width, device, out):
    if out is None:
        return torch.empty((rows, width), dtype=torch.int32, device=device)
    if out.shape != (rows, width) or out.dtype != torch.int32 or out.device != device:
        raise ValueError("QSA output must be same-device int32 [rows, output_width]")
    return out


def expand_qsa_block_indices(
    block_indices,
    query_positions,
    sequence_lengths,
    token_to_req,
    compress_ratio,
    token_topk,
    out=None,
):
    """Expand selected complete blocks and append the current incomplete tail.

    Invalid requests, invalid selected blocks and noncausal tokens produce -1.
    Shapes and scalar geometry alone control Python branches; request contents
    stay on device. The optional output is caller-owned for graph integration.
    """
    block_topk, width = _check_selection_geometry(compress_ratio, token_topk)
    if block_indices.ndim != 2 or block_indices.shape[1] < block_topk:
        raise ValueError("QSA block indices must provide block_topk columns")
    rows = block_indices.shape[0]
    if query_positions.shape != (rows,) or token_to_req.shape != (rows,):
        raise ValueError("QSA request metadata must match block-index rows")
    if sequence_lengths.ndim != 1:
        raise ValueError("QSA sequence lengths must be a vector")
    tensors = (block_indices, query_positions, sequence_lengths, token_to_req)
    if any(t.dtype not in (torch.int32, torch.int64) for t in tensors):
        raise TypeError("QSA indices and lengths must be int32 or int64")
    if any(t.device != block_indices.device for t in tensors):
        raise ValueError("QSA index metadata must be on one device")
    out = _selection_output(rows, width, block_indices.device, out)
    if rows == 0:
        return out

    requests = token_to_req.to(torch.int64)
    positions = query_positions.to(torch.int64)
    valid_request = (requests >= 0) & (requests < sequence_lengths.numel())
    if sequence_lengths.numel() == 0:
        lengths = torch.zeros_like(positions)
    else:
        safe_requests = requests.clamp(0, sequence_lengths.numel() - 1)
        lengths = sequence_lengths[safe_requests].to(torch.int64)
        lengths = torch.where(valid_request, lengths, torch.zeros_like(lengths))

    complete = torch.minimum(
        torch.div((positions + 1).clamp_min(0), compress_ratio, rounding_mode="floor"),
        torch.div(lengths.clamp_min(0), compress_ratio, rounding_mode="floor"),
    ).clamp_max(block_topk)
    # FG advanced indexing may produce column-major selected rows. A Python
    # scalar in its noncontiguous mul path is materialized via torch.tensor
    # on CUDA, which is not capture-safe. Build the integer constant on device
    # and use tensor/tensor multiplication for every ratio product here.
    ratio = torch.full((), compress_ratio, dtype=torch.int64, device=block_indices.device)
    expanded_count = complete * ratio
    tail_start = (
        torch.div((positions + 1).clamp_min(0), compress_ratio, rounding_mode="floor")
        * ratio
    )
    tail_count = (positions + 1 - tail_start).clamp_min(0)

    columns = torch.arange(width, dtype=torch.int64, device=block_indices.device)
    block_columns = torch.div(columns, compress_ratio, rounding_mode="floor")
    # Clamp only the unused tail reads; their values are replaced below.
    selected = block_indices[:, block_columns.clamp_max(block_topk - 1)].to(torch.int64)
    block_tokens = selected * ratio + columns.remainder(compress_ratio)
    tail_offsets = columns.unsqueeze(0) - expanded_count.unsqueeze(1)
    from_block = columns.unsqueeze(0) < expanded_count.unsqueeze(1)
    tail_tokens = tail_start.unsqueeze(1) + tail_offsets
    candidates = torch.where(from_block, block_tokens, tail_tokens)
    tail_valid = (
        (tail_offsets >= 0)
        & (tail_offsets < tail_count.unsqueeze(1))
        & (tail_offsets < compress_ratio - 1)
    )
    valid = (
        valid_request.unsqueeze(1)
        & (from_block | tail_valid)
        & (candidates >= 0)
        & (candidates < lengths.unsqueeze(1))
        & (candidates <= positions.unsqueeze(1))
    )
    result = torch.where(valid, candidates, torch.full_like(candidates, -1))
    out.copy_(result.to(torch.int32))
    return out


def qsa_select_paged_tokens(
    q,
    k_cache,
    page_table,
    token_to_req,
    query_positions,
    sequence_lengths,
    token_topk,
    compress_ratio,
    out=None,
    num_columns=None,
):
    """Compose existing FGV scoring/top-k with Qwen's block/tail expansion.

    Warm up the selected FGV vendor/TLE path before graph capture. Its first
    invocation may initialize library dispatch state; policy must then remain
    fixed for the worker/graph lifetime. This is composition, not a retry path.
    """
    block_topk, width = _check_selection_geometry(compress_ratio, token_topk)
    rows = q.shape[0]
    out = _selection_output(rows, width, q.device, out)
    if rows == 0:
        return out
    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else num_columns
    if not isinstance(columns, int) or not 0 <= columns <= capacity:
        raise ValueError("QSA score width must fit the allocated cache capacity")
    scores, visible = qsa_mqa_paged(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
        num_columns=columns,
    )
    indices = torch.full((rows, block_topk), -1, dtype=torch.int32, device=q.device)
    if columns:
        starts = torch.zeros_like(visible, dtype=torch.int32)
        ends = visible.to(torch.int32).clamp(0, columns)
        # With starts=0 the FGV prefill ABI returns request-relative block
        # indices. Its per-row valid range also covers decode/speculative rows.
        _fgv("top_k_per_row_prefill")(
            scores,
            starts,
            ends,
            indices,
            rows,
            scores.stride(0),
            scores.stride(1),
            block_topk,
        )
    return expand_qsa_block_indices(
        indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio,
        token_topk,
        out,
    )


def ple_short_conv_(
    x,
    output,
    conv_state,
    conv_weight,
    state_indices,
    *,
    dilation,
    query_start_loc=None,
    has_initial_state=None,
    accepted_tokens=None,
    max_query_len=1,
    null_block_id=-1,
):
    return _fgv("ple_short_conv_")(
        x,
        output,
        conv_state,
        conv_weight,
        state_indices,
        dilation=dilation,
        query_start_loc=query_start_loc,
        has_initial_state=has_initial_state,
        accepted_tokens=accepted_tokens,
        max_query_len=max_query_len,
        null_block_id=null_block_id,
    )


def gdn_packed_decode(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    initial_state,
    out,
    ssm_state_indices,
    use_qk_l2norm_in_kernel=False,
):
    """One token per sequence only; speculative calls retain native vLLM."""
    if mixed_qkv.ndim != 2 or ssm_state_indices.ndim != 1:
        raise ValueError("Packed GDN is only valid for one-token non-spec decode")
    # The pinned FGV kernel rounds sigmoid(beta) through b.dtype. FP32 b
    # preserves the precision of vLLM FL's corrected packed kernel and the
    # native prefill/spec paths without changing library code or gate order.
    b = b.to(torch.float32)
    return _fgv("fused_recurrent_gated_delta_rule_packed_decode")(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        out,
        ssm_state_indices,
        use_qk_l2norm_in_kernel,
    )


def register_qwen38_operators(registry, *, allow):
    from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

    def bind(name):
        fn = globals()[name]

        @wraps(fn)
        def call(*args, **kwargs):
            # No execution-time retry: several operations mutate cache/state.
            return fn(*args, **kwargs)

        call._is_available = partial(is_available, name)
        return call

    registry.register_many(
        [
            OpImpl(
                name,
                "default.flagos",
                BackendImplKind.DEFAULT,
                bind(name),
                priority=BackendPriority.DEFAULT,
            )
            for name in QWEN38_OPERATORS
            if allow(name)
        ]
    )
