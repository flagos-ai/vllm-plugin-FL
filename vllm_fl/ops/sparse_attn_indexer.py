# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os
from math import prod

import torch
import torch.distributed as dist

from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.worker.workspace import current_workspace_manager

from vllm_fl.dispatch import CachedOp
from vllm_fl.ops.deepseek_v4 import (
    int8_mqa_logits,
    int8_paged_mqa_logits,
)

# Register handles during model construction. The worker resolves and freezes
# them after loading so Dynamo sees only the selected implementation.
_indexer_k_quant_and_cache = CachedOp("indexer_k_quant_and_cache")
_cp_gather_indexer_k_quant_cache = CachedOp("cp_gather_indexer_k_quant_cache")
_top_k_per_row_prefill = CachedOp("top_k_per_row_prefill")
_pack_seq_triton = CachedOp("pack_seq_triton")
_top_k_per_row_decode = CachedOp("top_k_per_row_decode")
_unpack_seq_triton = CachedOp("unpack_seq_triton")

logger = init_logger(__name__)


# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32

# These switches are process-level configuration. Reading them once avoids
# repeating Python environment lookups in every indexer layer invocation.
_INDEXER_CP_REQUESTED_SIZE = int(os.getenv("VLLM_FL_INDEXER_CP_SIZE", "0"))
_INDEXER_CP_PROCESS_GROUP = None
_INDEXER_CP_GROUP_RANK = 0
_INDEXER_CP_GROUP_SIZE = 1
_INDEXER_CP_GROUP_RANKS: list[int] = []


def _indexer_cp_info(use_int8_cache: bool) -> tuple[int, int]:
    """Return rank/size for indexer-only CP over the existing TP group."""
    global _INDEXER_CP_PROCESS_GROUP
    global _INDEXER_CP_GROUP_RANK
    global _INDEXER_CP_GROUP_SIZE
    global _INDEXER_CP_GROUP_RANKS

    requested_size = _INDEXER_CP_REQUESTED_SIZE
    if not use_int8_cache or requested_size in (0, 1):
        return 0, 1
    if requested_size < 0:
        raise ValueError("VLLM_FL_INDEXER_CP_SIZE must be non-negative")
    tp_group = get_tp_group()
    if tp_group.world_size % requested_size != 0:
        raise ValueError(
            "VLLM_FL_INDEXER_CP_SIZE must be 0/1 (disabled) or a divisor "
            "of TP size greater than 1"
        )
    if requested_size == tp_group.world_size:
        _INDEXER_CP_PROCESS_GROUP = tp_group.device_group
        _INDEXER_CP_GROUP_RANK = tp_group.rank_in_group
        _INDEXER_CP_GROUP_SIZE = tp_group.world_size
        _INDEXER_CP_GROUP_RANKS = tp_group.ranks
        return _INDEXER_CP_GROUP_RANK, _INDEXER_CP_GROUP_SIZE

    if _INDEXER_CP_PROCESS_GROUP is None or requested_size != _INDEXER_CP_GROUP_SIZE:
        selected_group = None
        selected_rank = 0
        selected_ranks: list[int] = []
        backend = dist.get_backend(tp_group.device_group)
        for start in range(0, tp_group.world_size, requested_size):
            ranks = tp_group.ranks[start : start + requested_size]
            process_group = dist.new_group(ranks=ranks, backend=backend)
            if tp_group.rank in ranks:
                selected_group = process_group
                selected_rank = ranks.index(tp_group.rank)
                selected_ranks = ranks
        assert selected_group is not None
        _INDEXER_CP_PROCESS_GROUP = selected_group
        _INDEXER_CP_GROUP_RANK = selected_rank
        _INDEXER_CP_GROUP_SIZE = requested_size
        _INDEXER_CP_GROUP_RANKS = selected_ranks
    return _INDEXER_CP_GROUP_RANK, _INDEXER_CP_GROUP_SIZE


def _indexer_cp_group(cp_size: int):
    tp_group = get_tp_group()
    if cp_size == tp_group.world_size:
        return tp_group.device_group, tp_group.ranks
    assert _INDEXER_CP_PROCESS_GROUP is not None
    return _INDEXER_CP_PROCESS_GROUP, _INDEXER_CP_GROUP_RANKS


def _cp_row_ids(
    num_groups: int, cp_rank: int, cp_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return equally shaped round-robin row ids for an uneven batch."""
    local_groups = (num_groups + cp_size - 1) // cp_size
    group_ids = (
        torch.arange(local_groups, dtype=torch.int64, device=device) * cp_size + cp_rank
    )
    valid = group_ids < num_groups
    return group_ids.clamp_max(num_groups - 1), valid


def _all_gather_cp_rows(
    local_rows: torch.Tensor,
    num_groups: int,
    rows_per_group: int,
    cp_size: int,
    out_rows: torch.Tensor,
) -> torch.Tensor:
    """Gather row-CP results and restore the original group-major order."""
    local_groups = local_rows.shape[0] // rows_per_group
    cp_process_group, _ = _indexer_cp_group(cp_size)
    if num_groups % cp_size == 0:
        result = out_rows[: num_groups * rows_per_group]
        dist.all_gather_into_tensor(
            result,
            local_rows.contiguous(),
            group=cp_process_group,
        )
        return result

    gathered = torch.empty(
        (cp_size * local_rows.shape[0], *local_rows.shape[1:]),
        dtype=local_rows.dtype,
        device=local_rows.device,
    )
    dist.all_gather_into_tensor(
        gathered,
        local_rows.contiguous(),
        group=cp_process_group,
    )
    reordered = (
        gathered.view(
            cp_size,
            local_groups,
            rows_per_group,
            *local_rows.shape[1:],
        )
        .permute(1, 0, 2, *range(3, local_rows.dim() + 2))
        .reshape(
            cp_size * local_groups * rows_per_group,
            *local_rows.shape[1:],
        )[: num_groups * rows_per_group]
    )
    out_rows[: num_groups * rows_per_group].copy_(reordered)
    return out_rows[: num_groups * rows_per_group]


def _global_topk_from_cp_shard(
    local_logits: torch.Tensor,
    topk_tokens: int,
    cp_rank: int,
    cp_size: int,
    shard_unit: int,
    local_lens: torch.Tensor | None = None,
    global_lens: torch.Tensor | None = None,
    next_n: int = 1,
) -> torch.Tensor:
    """Return the exact global top-k from round-robin key-tile shards.

    An item outside a shard's local K cannot be in the global K.  Each rank
    therefore exchanges only K score/index pairs, independent of context
    length, instead of gathering the complete logits tensor.
    """
    if local_lens is None:
        local_values, local_indices = torch.topk(
            local_logits, topk_tokens, dim=-1, sorted=False
        )
        global_indices = (
            local_indices // shard_unit * cp_size + cp_rank
        ) * shard_unit + local_indices % shard_unit
    else:
        # Decode logits reserve max_model_len / CP_SIZE columns for graph
        # replay, but only a small prefix is valid for short contexts. Reuse
        # vLLM's length-aware CUDA radix top-k so 1K decode scans ~128 values
        # per TP8 rank instead of the full 16K allocation.
        local_indices_i32 = torch.full(
            (local_logits.shape[0], topk_tokens),
            -1,
            dtype=torch.int32,
            device=local_logits.device,
        )
        _top_k_per_row_decode(
            local_logits,
            next_n,
            local_lens,
            local_indices_i32,
            local_logits.shape[0],
            local_logits.stride(0),
            local_logits.stride(1),
            topk_tokens,
        )
        local_width = local_logits.shape[1]
        valid_indices = (local_indices_i32 >= 0) & (local_indices_i32 < local_width)
        local_indices = local_indices_i32.clamp(0, local_width - 1).to(torch.int64)
        local_values = torch.gather(local_logits, 1, local_indices)
        local_values = torch.where(valid_indices, local_values, -float("inf"))
        global_indices = (
            local_indices // shard_unit * cp_size + cp_rank
        ) * shard_unit + local_indices % shard_unit
    if global_lens is not None:
        valid_global_indices = global_indices < global_lens.reshape(-1, 1)
        local_values = torch.where(valid_global_indices, local_values, -float("inf"))
    candidates = torch.stack(
        (local_values, global_indices.to(local_values.dtype)), dim=-1
    )
    tp_group = get_tp_group()
    cp_process_group, cp_group_ranks = _indexer_cp_group(cp_size)
    # A full all-gather makes every rank receive and merge CP_SIZE*K
    # candidates.  Only the final indices are consumed by every rank, so
    # merge once on rank 0 and broadcast the K int32 results.  For TP8/K512
    # this reduces each non-root rank's receive volume from 32 KiB/row to
    # 2 KiB/row while preserving exact top-k semantics.
    if cp_size == tp_group.world_size:
        gathered = tp_group.gather(candidates, dst=0, dim=1)
    else:
        gather_list = (
            [torch.empty_like(candidates) for _ in range(cp_size)]
            if cp_rank == 0
            else None
        )
        dist.gather(
            candidates,
            gather_list=gather_list,
            dst=cp_group_ranks[0],
            group=cp_process_group,
        )
        gathered = torch.cat(gather_list, dim=1) if gather_list is not None else None
    if cp_rank == 0:
        assert gathered is not None
        global_values, candidate_indices = torch.topk(
            gathered[..., 0], topk_tokens, dim=-1, sorted=False
        )
        global_indices = torch.gather(gathered[..., 1], 1, candidate_indices).to(
            torch.int32
        )
        global_indices = torch.where(torch.isfinite(global_values), global_indices, -1)
    else:
        global_indices = torch.empty(
            local_values.shape[0],
            topk_tokens,
            dtype=torch.int32,
            device=local_values.device,
        )
    if cp_size == tp_group.world_size:
        return tp_group.broadcast(global_indices, src=0)
    dist.broadcast(
        global_indices,
        src=cp_group_ranks[0],
        group=cp_process_group,
    )
    return global_indices


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
    use_int8_cache: bool = False,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    if use_int8_cache:
        return (
            ((total_seq_lens, head_dim), torch.int8),
            ((total_seq_lens, 4), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


_WORKSPACE_ALIGNMENT = 256


def _workspace_nbytes(
    specs: list[tuple[tuple[int, ...], torch.dtype]],
) -> int:
    total = 0
    for shape, dtype in specs:
        actual = prod(shape) * dtype.itemsize
        total += (
            (actual + _WORKSPACE_ALIGNMENT - 1)
            // _WORKSPACE_ALIGNMENT
            * _WORKSPACE_ALIGNMENT
        )
    return total


def _workspace_views(
    workspace: torch.Tensor,
    specs: list[tuple[tuple[int, ...], torch.dtype]],
) -> list[torch.Tensor]:
    views = []
    offset = 0
    for shape, dtype in specs:
        actual = prod(shape) * dtype.itemsize
        views.append(workspace[offset : offset + actual].view(dtype).reshape(shape))
        offset += (
            (actual + _WORKSPACE_ALIGNMENT - 1)
            // _WORKSPACE_ALIGNMENT
            * _WORKSPACE_ALIGNMENT
        )
    assert offset <= workspace.numel()
    return views


def _get_workspace_buffers(
    workspace: torch.Tensor,
    specs: list[tuple[tuple[int, ...], torch.dtype]],
) -> list[torch.Tensor]:
    # The packed attention arena is sized for the actual prefill lifecycle.
    # Small synthetic/warmup shapes may not reserve an indexer region; retain
    # the regular global-workspace path in that case.
    if workspace.numel() >= _workspace_nbytes(specs):
        return _workspace_views(workspace, specs)
    return current_workspace_manager().get_simultaneous(*specs)


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


def sparse_attn_indexer_fl(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    if workspace is None:
        workspace = hidden_states.new_empty((0,), dtype=torch.uint8)
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)
    use_int8_cache = q_quant.dtype == torch.int8
    indexer_cp_rank, indexer_cp_size = _indexer_cp_info(use_int8_cache)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens,
            head_dim,
            fp8_dtype,
            use_fp4_cache,
            use_int8_cache,
        )
        _get_workspace_buffers(workspace, [values_spec, scales_spec])

        # Memory profiling runs a real max-token prefill for DSV4.  A second
        # synthetic logits allocation here would instead happen during the
        # post-KV-cache compile warmup and double-count an already-profiled
        # peak.

        return sparse_attn_indexer_fl_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
            workspace,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        _indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens,
            head_dim,
            fp8_dtype,
            use_fp4_cache,
            use_int8_cache,
        )
        workspace_specs = [values_spec, scales_spec]
        if indexer_cp_size > 1:
            # Row-CP handles every chunk with at least CP_SIZE query rows.
            # This workspace is only needed by the smaller key-CP tail.
            max_key_cp_rows = min(topk_indices_buffer.shape[0], indexer_cp_size - 1)
            local_logits_width = max(
                512,
                (((total_seq_lens + 63) // 64 + indexer_cp_size - 1) // indexer_cp_size)
                * 64,
            )
            workspace_specs.append(
                ((max_key_cp_rows, local_logits_width), torch.float32)
            )
        workspace_buffers = _get_workspace_buffers(workspace, workspace_specs)
        k_quant_full, k_scale_full = workspace_buffers[:2]
        if indexer_cp_size > 1:
            local_logits_full = workspace_buffers[2]
        for chunk in prefill_metadata.chunks:
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                _cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)

            prefill_row_cp = False
            if use_int8_cache:
                prefill_weights = weights[chunk.token_start : chunk.token_end]
                prefill_row_cp = (
                    indexer_cp_size > 1 and q_slice_cast.shape[0] >= indexer_cp_size
                )
                if prefill_row_cp:
                    if q_slice_cast.shape[0] % indexer_cp_size == 0:
                        local_rows = q_slice_cast.shape[0] // indexer_cp_size
                        row_start = indexer_cp_rank * local_rows
                        row_end = row_start + local_rows
                        local_q = q_slice_cast[row_start:row_end]
                        local_weights = prefill_weights[row_start:row_end]
                        local_cu_ks = chunk.cu_seqlen_ks[row_start:row_end]
                        local_cu_ke = chunk.cu_seqlen_ke[row_start:row_end]
                    else:
                        row_ids, valid_rows = _cp_row_ids(
                            q_slice_cast.shape[0],
                            indexer_cp_rank,
                            indexer_cp_size,
                            q_slice_cast.device,
                        )
                        local_q = q_slice_cast.index_select(0, row_ids)
                        local_weights = prefill_weights.index_select(0, row_ids)
                        local_cu_ks = chunk.cu_seqlen_ks.index_select(0, row_ids)
                        local_cu_ke = chunk.cu_seqlen_ke.index_select(0, row_ids)
                        local_cu_ks = torch.where(valid_rows, local_cu_ks, 0)
                        local_cu_ke = torch.where(valid_rows, local_cu_ke, 0)
                    logits = int8_mqa_logits(
                        local_q,
                        k_quant_cast,
                        k_scale_cast,
                        local_weights,
                        local_cu_ks,
                        local_cu_ke,
                    )
                else:
                    logits = int8_mqa_logits(
                        q_slice_cast,
                        k_quant_cast,
                        k_scale_cast,
                        prefill_weights,
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                        cp_rank=indexer_cp_rank,
                        cp_size=indexer_cp_size,
                        logits_out=(
                            local_logits_full[: q_slice_cast.shape[0]]
                            if indexer_cp_size > 1
                            else None
                        ),
                    )
            else:
                logits = fp8_fp4_mqa_logits(
                    (q_slice_cast, q_scale_slice),
                    (k_quant_cast, k_scale_cast),
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    clean_logits=False,
                )
            num_rows = logits.shape[0] if prefill_row_cp else q_slice_cast.shape[0]

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            if use_int8_cache and prefill_row_cp:
                local_topk_indices = torch.empty(
                    (logits.shape[0], topk_tokens),
                    dtype=torch.int32,
                    device=logits.device,
                )
                _top_k_per_row_prefill(
                    logits,
                    local_cu_ks,
                    local_cu_ke,
                    local_topk_indices,
                    logits.shape[0],
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
                _all_gather_cp_rows(
                    local_topk_indices,
                    q_slice_cast.shape[0],
                    rows_per_group=1,
                    cp_size=indexer_cp_size,
                    out_rows=topk_indices,
                )
            elif indexer_cp_size > 1:
                topk_indices.copy_(
                    _global_topk_from_cp_shard(
                        logits,
                        topk_tokens,
                        indexer_cp_rank,
                        indexer_cp_size,
                        shard_unit=64,
                    )
                )
            else:
                _top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        if not use_int8_cache:
            kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = _pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = _pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = _pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )

        decode_row_cp = False
        if use_int8_cache:
            decode_row_cp = indexer_cp_size > 1 and batch_size >= indexer_cp_size
            if decode_row_cp:
                if batch_size % indexer_cp_size == 0:
                    local_batches = batch_size // indexer_cp_size
                    batch_start = indexer_cp_rank * local_batches
                    batch_end = batch_start + local_batches
                    local_q = padded_q_quant_decode_tokens[batch_start:batch_end]
                    local_seq_lens = seq_lens[batch_start:batch_end]
                    local_weights = (
                        weights[:num_padded_tokens]
                        .reshape(batch_size, next_n, -1)[batch_start:batch_end]
                        .reshape(-1, weights.shape[-1])
                    )
                    local_block_table = decode_metadata.block_table[
                        batch_start:batch_end
                    ]
                else:
                    batch_ids, valid_batches = _cp_row_ids(
                        batch_size,
                        indexer_cp_rank,
                        indexer_cp_size,
                        padded_q_quant_decode_tokens.device,
                    )
                    local_q = padded_q_quant_decode_tokens.index_select(0, batch_ids)
                    local_seq_lens = seq_lens.index_select(0, batch_ids)
                    local_seq_lens = torch.where(
                        valid_batches[:, None], local_seq_lens, 0
                    )
                    local_weights = (
                        weights[:num_padded_tokens]
                        .reshape(batch_size, next_n, -1)
                        .index_select(0, batch_ids)
                        .reshape(-1, weights.shape[-1])
                    )
                    local_block_table = decode_metadata.block_table.index_select(
                        0, batch_ids
                    )
                logits = int8_paged_mqa_logits(
                    local_q,
                    kv_cache,
                    local_weights,
                    local_seq_lens,
                    local_block_table,
                    max_model_len=max_model_len,
                    pages_per_cta=2,
                )
            # local_lens is only consumed by the distributed top-k path.  In
            # CP=0 the native global top-k already uses seq_lens, so computing
            # it here merely launches several pointwise CUDA kernels per layer.
            decode_local_lens = None
            if indexer_cp_size > 1 and not decode_row_cp:
                decode_full_pages = torch.div(seq_lens, 64, rounding_mode="floor")
                decode_tail = seq_lens % 64
                decode_local_full_pages = torch.clamp(
                    torch.div(
                        decode_full_pages + indexer_cp_size - 1 - indexer_cp_rank,
                        indexer_cp_size,
                        rounding_mode="floor",
                    ),
                    min=0,
                )
                decode_local_lens = decode_local_full_pages * 64 + torch.where(
                    decode_full_pages % indexer_cp_size == indexer_cp_rank,
                    decode_tail,
                    0,
                )
            if decode_row_cp:
                pass
            elif indexer_cp_size > 1:
                logits = int8_paged_mqa_logits(
                    padded_q_quant_decode_tokens,
                    kv_cache,
                    weights[:num_padded_tokens],
                    seq_lens,
                    decode_metadata.block_table,
                    max_model_len=max_model_len,
                    cp_rank=indexer_cp_rank,
                    cp_size=indexer_cp_size,
                )
            else:
                logits = int8_paged_mqa_logits(
                    padded_q_quant_decode_tokens,
                    kv_cache,
                    weights[:num_padded_tokens],
                    seq_lens,
                    decode_metadata.block_table,
                    max_model_len=max_model_len,
                )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        if use_int8_cache and decode_row_cp:
            local_topk_indices = torch.empty(
                (logits.shape[0], topk_tokens),
                dtype=torch.int32,
                device=logits.device,
            )
            _top_k_per_row_decode(
                logits,
                next_n,
                local_seq_lens,
                local_topk_indices,
                logits.shape[0],
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )
            _all_gather_cp_rows(
                local_topk_indices,
                batch_size,
                rows_per_group=next_n,
                cp_size=indexer_cp_size,
                out_rows=topk_indices,
            )
        elif indexer_cp_size > 1:
            assert decode_local_lens is not None
            topk_indices.copy_(
                _global_topk_from_cp_shard(
                    logits,
                    topk_tokens,
                    indexer_cp_rank,
                    indexer_cp_size,
                    shard_unit=64,
                    local_lens=decode_local_lens,
                    global_lens=seq_lens,
                    next_n=next_n,
                )
            )
        else:
            _top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = _unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fl_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    del workspace
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer_fl",
    op_func=sparse_attn_indexer_fl,
    mutates_args=["topk_indices_buffer", "workspace"],
    fake_impl=sparse_attn_indexer_fl_fake,
    dispatch_key=current_platform.dispatch_key,
)


class SparseAttnIndexerFL(SparseAttnIndexer):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    # def __init__(
    #     self,
    #     k_cache,
    #     quant_block_size: int,
    #     scale_fmt: str,
    #     topk_tokens: int,
    #     head_dim: int,
    #     max_model_len: int,
    #     max_total_seq_len: int,
    #     topk_indices_buffer: torch.Tensor,
    #     skip_k_cache_insert: bool = False,
    #     use_fp4_cache: bool = False,
    # ):
    #     super().__init__()
    #     self.k_cache = k_cache
    #     self.quant_block_size = quant_block_size
    #     self.scale_fmt = scale_fmt
    #     self.topk_tokens = topk_tokens
    #     self.head_dim = head_dim
    #     self.max_model_len = max_model_len
    #     self.max_total_seq_len = max_total_seq_len
    #     self.topk_indices_buffer = topk_indices_buffer
    #     self.skip_k_cache_insert = skip_k_cache_insert
    #     self.use_fp4_cache = use_fp4_cache
    #     if current_platform.is_cuda() and not has_deep_gemm():
    #         raise RuntimeError(
    #             "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
    #         )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        workspace: torch.Tensor,
    ):
        return self.forward_oot(hidden_states, q_quant, k, weights, workspace)

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        workspace: torch.Tensor,
    ):
        """Keep NVIDIA CUDA on the FL INT8-aware indexer path.

        vLLM 0.24 selects its in-tree CUDA platform even when the FL plugin is
        loaded.  Without this override ``CustomOp.forward`` resolves the
        inherited upstream ``SparseAttnIndexer.forward_cuda``, which sends an
        INT8 query to DeepGEMM's FP8-only MQA kernel.  The FL entrypoint owns
        the mixed-cache dispatch and routes its helper kernels through
        OpManager.
        """
        return self.forward_oot(hidden_states, q_quant, k, weights, workspace)

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        workspace: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer_fl(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
            workspace,
        )
