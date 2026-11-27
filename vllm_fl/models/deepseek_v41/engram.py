# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 BAAI. All rights reserved.
"""INT8 Engram model layers, adapted from vLLM e77daef89e18.

Only parameter loading, framework-owned staging and TP/Linear orchestration
live here. Numerical hashes/lookups/gates resolve through the existing Plugin
registry to public FGV APIs. This initial path supports HBM tables and TP;
sequence-parallel Engram and V1 slot-history caches are not implemented.
"""

import torch
from torch import nn

from vllm.distributed import (
    get_engram_dp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v41.common.engram import DPSharedEngramStorage
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from .engram_metadata import build_compressed_token_map
from vllm_fl.dispatch import resolve_op


def _load_engram_shard(param, loaded_weight):
    e8m0 = getattr(param, "engram_scale_encoding", None) == "F8_E8M0"
    expected_dtype = torch.float8_e8m0fnu if e8m0 else param.dtype
    if loaded_weight.dtype != expected_dtype:
        raise TypeError("Engram checkpoint/table dtype mismatch")
    if tuple(loaded_weight.shape) != param.engram_full_shape:
        raise ValueError("Engram checkpoint/table shape mismatch")
    shard = loaded_weight.narrow(0, param.engram_vocab_start, param.shape[0])
    if shard.shape != param.shape:
        raise ValueError("Engram checkpoint shard shape mismatch")
    if e8m0:
        convert = resolve_op("e8m0_scale_to_bf16")
        for first in range(0, shard.shape[0], 1 << 18):
            stop = min(first + (1 << 18), shard.shape[0])
            param.data[first:stop].copy_(convert(shard[first:stop]))
    else:
        param.data.copy_(shard)


class ParallelEngramEmbedding(nn.Module):
    """Store this rank's complete INT8 buckets and checkpoint BF16 group32 scales."""

    def __init__(
        self, layout, layer_hash_index, engram_config=None, scale_encoding=None
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        offload = engram_config is not None and engram_config.cpu_offload
        shared = offload and engram_config.dp_shared_memory
        self.dp_group = get_engram_dp_group() if offload and not shared else None
        self.dp_size = self.dp_group.world_size if self.dp_group is not None else 1
        self.cpu_offload = offload
        shard_rank = self.tp_rank * self.dp_size + (
            self.dp_group.rank_in_group if self.dp_group else 0
        )
        shard = layout.shard(layer_hash_index, self.tp_size * self.dp_size, shard_rank)
        self.n_hash_cols = layout.n_hash_cols
        self.head_dim = layout.head_dim
        self.shard = shard
        total_rows = layout.num_embeddings[layer_hash_index]
        self.host_storage = None
        if shared:
            group = get_engram_dp_group()
            if group is None or group.world_size <= 1:
                raise ValueError(
                    "INT8 Engram shared memory requires co-located DP replicas"
                )
            self.host_storage = Int8EngramStorage(shard.rows, self.head_dim, group)
            weight = self.host_storage.weight
            scales = self.host_storage.weight_scale_inv
        elif offload:
            weight = torch.empty(
                (shard.rows, self.head_dim),
                dtype=torch.int8,
                device="cpu",
                pin_memory=True,
            )
            scales = torch.empty(
                (shard.rows, self.head_dim // 32),
                dtype=torch.bfloat16,
                device="cpu",
                pin_memory=True,
            )
        else:
            weight = torch.empty((shard.rows, self.head_dim), dtype=torch.int8)
            scales = torch.empty(
                (shard.rows, self.head_dim // 32), dtype=torch.bfloat16
            )
        self.weight = nn.Parameter(
            weight,
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            scales,
            requires_grad=False,
        )
        self.weight_scale.engram_scale_encoding = scale_encoding
        self._views = None
        for parameter, full_shape in [
            (self.weight, (total_rows, self.head_dim)),
            (self.weight_scale, (total_rows, self.head_dim // 32)),
        ]:
            set_weight_attrs(
                parameter,
                {
                    "weight_loader": self._load_weight,
                    "engram_vocab_start": shard.vocab_start,
                    "engram_full_shape": full_shape,
                },
            )

    def _load_weight(self, parameter, loaded_weight):
        storage = self.host_storage
        if storage is None or storage.group.rank_in_group == 0:
            _load_engram_shard(parameter, loaded_weight)
        if storage is not None:
            torch.distributed.barrier(group=storage.group.cpu_group)

    def storage_views(self):
        if self.host_storage is not None:
            return self.host_storage.get_views(self.weight, self.weight_scale)
        if self.cpu_offload:
            if self._views is None:
                self._views = (
                    get_accelerator_view_from_cpu_tensor(self.weight),
                    get_accelerator_view_from_cpu_tensor(self.weight_scale),
                )
            return self._views
        return self.weight, self.weight_scale


class Int8EngramStorage(DPSharedEngramStorage):
    """Native shared mapping/lifetime with checkpoint INT8 and BF16 group32 views."""

    def __init__(self, rows, dim, group):
        self.group = group
        weight_bytes = rows * dim
        storage = self._allocate(weight_bytes + rows * (dim // 32) * 2)
        self.weight = storage[:weight_bytes].view(torch.int8).view(rows, dim)
        self.weight_scale_inv = (
            storage[weight_bytes:].view(torch.bfloat16).view(rows, dim // 32)
        )
        self._views = None


class EngramHasher(nn.Module):
    """Fixed compressed token map/layout; V2 supplies raw history per step."""

    def __init__(self, config, tokenizer, layout):
        super().__init__()
        mapping, compressed_size = build_compressed_token_map(tokenizer)
        if (
            len(mapping) != config.vocab_size
            or compressed_size != layout.compressed_vocab_size
        ):
            raise ValueError(
                "Engram tokenizer compressed vocabulary does not match config"
            )
        self.pad_id = mapping[layout.pad_token_id]
        self.lookback_depth = layout.max_ngram_size - 1
        for name, value, dtype in [
            ("token_map", mapping, torch.int32),
            ("multipliers", layout.multipliers, torch.int64),
            ("primes", layout.primes, torch.int64),
            ("offsets", layout.offsets, torch.int64),
        ]:
            self.register_buffer(
                name, torch.tensor(value, dtype=dtype), persistent=False
            )

    def forward(self, raw_history_ids):
        implementation = resolve_op("engram_ngram_hash")
        return implementation(
            raw_history_ids,
            self.token_map,
            self.multipliers,
            self.primes,
            self.offsets,
            self.pad_id,
        )


class Engram(nn.Module):
    """Gather this layer's INT8 table, project W8A8 KV, then gate residuals."""

    def __init__(
        self,
        config,
        quant_config,
        layout,
        layer_hash_index,
        max_tokens,
        use_sequence_parallel=False,
        prefix="",
        engram_config=None,
    ):
        super().__init__()
        if use_sequence_parallel:
            raise NotImplementedError(
                "V4.1 Engram sequence parallel is not implemented"
            )
        if config.hc_mult != 4 or max_tokens <= 0:
            raise ValueError("Engram requires HC=4 and positive token capacity")
        self.layer_hash_index = layer_hash_index
        self.dim = config.hidden_size
        self.eps = config.rms_norm_eps
        from vllm_fl.quantization.modelslim import ModelSlimConfig

        modelslim = quant_config if isinstance(quant_config, ModelSlimConfig) else None
        storage = None
        if modelslim is not None:
            key = f"layers.{layout.layer_ids[layer_hash_index]}.engram.embed"
            storage = modelslim.embedding_storage[key]
        self.embed_tokens = ParallelEngramEmbedding(
            layout,
            layer_hash_index,
            engram_config,
            None if storage is None else storage["scale_dtype"],
        )
        self.register_buffer("rotation_block", None, persistent=False)
        if modelslim is not None:
            self.rotation_block = torch.empty((32, 32), dtype=torch.float32)
            self.rotation_block.copy_(modelslim.rotation_block)
        rotation = (
            getattr(config, "online_rotation", {})
            .get("modules", {})
            .get(prefix.removeprefix("language_model.") + ".global_rotation")
        )
        self.global_rotation = None
        if rotation is not None:
            if (
                rotation.get("type") != "composite_hadamard"
                or rotation.get("hidden_size") != self.dim
            ):
                raise NotImplementedError("unsupported checkpoint Engram rotation")
            self.global_rotation = EngramRotation(rotation)
        self.wkv = ReplicatedLinear(
            layout.n_hash_cols * layout.head_dim,
            5 * self.dim,
            bias=False,
            quant_config=quant_config,
            params_dtype=torch.bfloat16,
            return_bias=False,
            prefix=f"{prefix}.wkv",
        )
        self.q_weight = nn.Parameter(
            torch.empty((4, self.dim), dtype=torch.bfloat16), requires_grad=False
        )
        self.k_weight = nn.Parameter(
            torch.empty((4, self.dim), dtype=torch.bfloat16), requires_grad=False
        )
        self.register_buffer(
            "staged_rows",
            torch.zeros(
                (
                    max_tokens * self.embed_tokens.dp_size,
                    (
                        layout.n_hash_cols
                        + self.embed_tokens.tp_size * self.embed_tokens.dp_size
                        - 1
                    )
                    // (self.embed_tokens.tp_size * self.embed_tokens.dp_size),
                    layout.head_dim,
                ),
                dtype=torch.bfloat16,
            ),
            persistent=False,
        )
        self.hash_shard_counts = tuple(
            layout.shard(layer_hash_index, self.embed_tokens.tp_size, rank).num_heads
            for rank in range(self.embed_tokens.tp_size)
        )
        self.register_buffer(
            "assembled_rows",
            (
                torch.empty(
                    (max_tokens, layout.n_hash_cols, layout.head_dim),
                    dtype=torch.bfloat16,
                )
                if layout.n_hash_cols % self.embed_tokens.tp_size
                else None
            ),
            persistent=False,
        )

    def forward(self, hidden_states, hash_ids, token_mask=None):
        if hidden_states.ndim != 3 or hidden_states.shape[1:] != (4, self.dim):
            raise ValueError("Engram hidden shape must be [tokens,4,hidden]")
        n = hidden_states.shape[0]
        if (
            hash_ids.ndim != 2
            or hash_ids.shape[1] != self.embed_tokens.n_hash_cols
            or n > self.staged_rows.shape[0]
        ):
            raise ValueError(
                "Engram hashes or token capacity do not match hidden states"
            )
        # Resolve both numerical APIs before lookup or a staging write starts.
        lookup = resolve_op("engram_embedding_int8")
        gate = resolve_op("engram_gated_add")
        shard = self.embed_tokens.shard
        weight, scales = self.embed_tokens.storage_views()
        local = lookup(
            weight,
            scales,
            hash_ids,
            shard.head_start,
            shard.num_heads,
            shard.vocab_start,
        )
        lookup_tokens = hash_ids.shape[0]
        rows = self.staged_rows[:lookup_tokens]
        rows[:, : shard.num_heads].copy_(local)
        if self.embed_tokens.dp_size > 1:
            group = self.embed_tokens.dp_group
            slot = lookup_tokens // group.world_size
            gathered = group.all_gather(rows, dim=0)
            rank_major = gathered.view(
                group.world_size,
                lookup_tokens,
                shard.num_heads,
                self.embed_tokens.head_dim,
            )
            start = group.rank_in_group * slot
            rows = (
                rank_major[:, start : start + n]
                .transpose(0, 1)
                .contiguous()
                .view(n, group.world_size * shard.num_heads, self.embed_tokens.head_dim)
            )
        if n == 0:
            rows = rows.new_empty(
                (0, self.embed_tokens.n_hash_cols, self.embed_tokens.head_dim)
            )
        elif self.embed_tokens.tp_size > 1:
            rows = tensor_model_parallel_all_gather(rows, dim=1)
            if self.assembled_rows is not None:
                # Uneven whole-head TP shards (24 heads across TP16). Only
                # transport padding is removed; hash/embedding order is kept.
                restored = self.assembled_rows[:n]
                width = self.staged_rows.shape[1]
                offset = 0
                for rank, count in enumerate(self.hash_shard_counts):
                    restored[:, offset : offset + count].copy_(
                        rows[:, rank * width : rank * width + count]
                    )
                    offset += count
                rows = restored
        kv = self.wkv(rows.flatten(-2))
        return gate(
            hidden_states,
            kv,
            self.q_weight,
            self.k_weight,
            eps=self.eps,
            clamp_value=1e-6,
            token_mask=token_mask,
            gate_input=(
                None
                if self.global_rotation is None
                else self.global_rotation(hidden_states)
            ),
            rotation_block=self.rotation_block,
        )


class EngramRotation(nn.Module):
    """Checkpoint parameters and dispatch only; transform arithmetic is in FG."""

    def __init__(self, specification):
        super().__init__()
        self.sign = nn.Parameter(
            torch.empty(specification["hidden_size"], dtype=torch.int8),
            requires_grad=False,
        )
        self.hk = nn.Parameter(
            torch.empty((specification["K"], specification["K"]), dtype=torch.float32),
            requires_grad=False,
        )
        self.inverse = bool(specification.get("inverse", True))
        self.transform = resolve_op("composite_hadamard_transform")

    def forward(self, value):
        return self.transform(value, self.sign, self.hk, inverse=self.inverse)
