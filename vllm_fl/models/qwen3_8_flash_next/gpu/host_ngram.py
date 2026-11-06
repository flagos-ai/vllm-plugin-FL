# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from plugin-FL PR573 at 45b24bc7748720e0022a72e47a983398ddda36d3.
from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace

import torch
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.models.utils import AutoWeightsLoader

from ..common.ple import (
    _PLE_HASH_BUFFER_NAMES,
    copy_ple_embedding_shard_,
    initialize_ple_hash_buffers,
    validate_ple_embedding_shard,
    validate_ple_embedding_weights,
)
from vllm_fl.models.qwen3_8_flash_next.config import Qwen3_8FlashNextTextConfig


class HostVocabEmbedding(nn.Module):
    """A pageable CPU vocabulary shard; only gathered rows cross PCIe.

    This module deliberately ignores model-wide device conversion. Its weight
    remains a named parameter for the standard checkpoint completeness checks.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        *,
        tp_rank: int,
        tp_size: int,
        dtype: torch.dtype,
        allocate: bool = True,
        shared_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if not 0 <= tp_rank < tp_size or vocab_size % tp_size:
            raise ValueError("host vocabulary must divide evenly across valid TP ranks")
        self.org_vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        rows = vocab_size // tp_size
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=tp_rank * rows,
            org_vocab_end_index=(tp_rank + 1) * rows,
        )
        if shared_weight is None:
            shared_weight = torch.empty(
                rows if allocate else 0, embedding_dim, dtype=dtype, device="cpu"
            )
        elif tuple(shared_weight.shape) != (rows, embedding_dim):
            raise ValueError("Shared host table shape mismatch")
        self.weight = nn.Parameter(shared_weight, requires_grad=False)

    def _apply(self, fn, recurse=True):
        # A later model.to(device) must never turn host offload into full GPU
        # residency. The BF16 base path does not change this parameter's dtype.
        return self


def compute_ngram_ids_cpu(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    context: torch.Tensor,
    *,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    vocab_offsets: torch.Tensor,
    heads_per_ngram: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Hash a packed batch without allocating max_requests x max_tokens.

    Context contains only committed tokens. Signed int64 overflow and remainder
    match the checkpoint's reference hash; EOS blocks reads across segments.
    This function is intentionally CPU-only and must run outside a GPU graph.
    """
    tensors = (
        input_ids,
        query_start_loc,
        context,
        multipliers,
        vocab_sizes,
        vocab_offsets,
    )
    if any(t.device.type != "cpu" for t in tensors):
        raise ValueError("N-gram host hashing expects CPU tensors")
    ids = input_ids.reshape(-1).to(torch.int64)
    starts = query_start_loc.to(torch.int64).tolist()
    ngram_size = multipliers.numel()
    num_heads = (ngram_size - 1) * heads_per_ngram
    if (
        not starts
        or starts[0] != 0
        or starts[-1] != ids.numel()
        or any(b < a for a, b in zip(starts, starts[1:]))
    ):
        raise ValueError("invalid packed N-gram query offsets")
    if tuple(context.shape) != (len(starts) - 1, ngram_size - 1):
        raise ValueError("N-gram context shape does not match packed requests")
    if vocab_sizes.numel() != num_heads or vocab_offsets.numel() != num_heads:
        raise ValueError("N-gram hash head metadata has inconsistent sizes")
    out = torch.empty(ids.numel(), num_heads, dtype=torch.int64, device="cpu")
    if ids.numel() == 0:
        return out
    # Map packed tokens to their request, without padding requests to a common
    # length. right=True skips empty requests at repeated query offsets.
    positions = torch.arange(ids.numel(), dtype=torch.int64, device="cpu")
    offsets = query_start_loc.to(torch.int64).contiguous()
    requests = torch.bucketize(positions, offsets[1:], right=True)
    local_positions = positions - offsets[requests]
    history = context.to(torch.int64)
    valid_history = torch.ones_like(positions, dtype=torch.bool)
    mixed = ids * multipliers[0]
    for shift in range(1, ngram_size):
        from_query = ids[(positions - shift).clamp_min(0)]
        history_positions = (ngram_size - 1 + local_positions - shift).clamp_max(
            ngram_size - 2
        )
        from_context = history[requests, history_positions]
        previous = torch.where(local_positions >= shift, from_query, from_context)
        # Once EOS occurs, all older tokens are replaced by EOS as well.
        valid_history &= previous != eos_token_id
        previous = torch.where(valid_history, previous, eos_token_id)
        mixed = torch.bitwise_xor(mixed, previous * multipliers[shift])
        h = (shift - 1) * heads_per_ngram
        out[:, h : h + heads_per_ngram] = (
            torch.remainder(mixed.unsqueeze(-1), vocab_sizes[h : h + heads_per_ngram])
            + vocab_offsets[h : h + heads_per_ngram]
        )
    return out


class HostNGramEmbedding(nn.Module):
    """TP vocabulary-sharded host table with an explicit graph-external prepare.

    ``prepare_host_lookup`` performs CPU gather, bounded pinned H2D and TP
    reduction. ``forward`` only reads the fixed device buffer, including during
    graph replay. There is no implicit host work or missing-input fallback.
    """

    def __init__(
        self,
        config: Qwen3_8FlashNextTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        max_num_reqs: int,
        prefix: str,
        params_dtype: torch.dtype = torch.bfloat16,
        host_lookup_mode: str = "sharded",
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2 or self.heads_per_ngram <= 0:
            raise ValueError("invalid N-gram order or head count")
        if embedding_dim % self.ngram_heads:
            raise ValueError("embedding width must divide by the N-gram head count")
        if params_dtype != torch.bfloat16:
            raise ValueError("Qwen3.8 base host N-gram path requires BF16 weights")
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 128))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")
        self.max_total_tokens = max_total_tokens
        self.max_num_reqs = max_num_reqs
        self.tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        if host_lookup_mode not in {
            "sharded",
            "shared_buffer",
            "uva_broadcast",
            "uva_sharded",
            "uva_auto",
        }:
            raise ValueError("Invalid N-gram host lookup mode")
        self.host_lookup_mode = host_lookup_mode
        self.tp_rank = tp_rank
        self._shared_lookup = None
        self._uva_lookup = None
        uva_mode = host_lookup_mode in {"uva_broadcast", "uva_sharded", "uva_auto"}
        group = None
        if host_lookup_mode == "shared_buffer" or uva_mode:
            from vllm.distributed import get_tp_group
            from vllm.distributed.parallel_state import in_the_same_node_as

            from vllm_fl.models.qwen3_8_flash_next.host_lookup import SharedNgramBuffer

            group = get_tp_group()
            if group.world_size > 1 and not all(in_the_same_node_as(group.cpu_group)):
                raise NotImplementedError(
                    "Shared N-gram results require one TP host/IPC namespace"
                )
            if not uva_mode:
                self._shared_lookup = SharedNgramBuffer(
                    group, max_total_tokens, embedding_dim, params_dtype
                )
        self._host_buffer_names = _PLE_HASH_BUFFER_NAMES
        padded_vocab_size = initialize_ple_hash_buffers(
            self, config, ple_dense_layer_id, device="cpu"
        )
        device = torch.empty(0).device
        if uva_mode:
            from .uva_ngram import UvaNgramLookup

            self._uva_lookup = UvaNgramLookup(
                group,
                padded_vocab_size,
                self.head_dim,
                params_dtype,
                device,
                max_total_tokens,
                self.heads_per_ngram,
                self.ngram_heads,
                self.eos_token_id,
                host_lookup_mode,
            )
        full_table = self._shared_lookup is not None or uva_mode
        self.ngram_embedding = HostVocabEmbedding(
            padded_vocab_size,
            self.head_dim,
            tp_rank=0 if full_table else tp_rank,
            tp_size=1 if full_table else self.tp_size,
            dtype=params_dtype,
            allocate=self._shared_lookup is None or tp_rank == 0,
            shared_weight=self._uva_lookup.weight if uva_mode else None,
        )
        # The CPU fallback pins only its gathered staging buffer. UVA explicitly
        # registers the shared full table after checkpoint loading.
        self.register_buffer(
            "prepared_embeddings",
            torch.empty(
                max_total_tokens, embedding_dim, dtype=params_dtype, device=device
            ),
            persistent=False,
        )
        self._host_stage = (
            None
            if uva_mode
            else torch.empty(
                max_total_tokens,
                self.ngram_heads,
                self.head_dim,
                dtype=params_dtype,
                device="cpu",
                pin_memory=device.type != "cpu",
            )
        )
        self._copy_event = None
        self._prepared = False
        self._prepared_capacity = 0

    def _apply(self, fn, recurse=True):
        # Hash metadata belongs to the CPU producer, even after model.to().
        host = {name: self._buffers.pop(name) for name in self._host_buffer_names}
        try:
            return super()._apply(fn, recurse)
        finally:
            self._buffers.update(host)

    def _host_hash_inputs(self, input_ids, query_start_loc, ngram_context, capacity):
        starts = query_start_loc.detach().to(device="cpu", dtype=torch.int64)
        nreq, actual = starts.numel() - 1, int(starts[-1])
        if nreq > self.max_num_reqs or not 0 <= actual <= capacity:
            raise ValueError("PLE request/token count exceeds fixed buffers")
        ids = input_ids[:actual].detach().to(device="cpu", dtype=torch.int64)
        context = ngram_context[:nreq].detach().to(device="cpu", dtype=torch.int64)
        hashed = compute_ngram_ids_cpu(
            ids,
            starts,
            context,
            multipliers=self.layer_multipliers,
            vocab_sizes=self.ngram_heads_vocab_sizes,
            vocab_offsets=self.ngram_heads_offsets,
            heads_per_ngram=self.heads_per_ngram,
            eos_token_id=self.eos_token_id,
        )
        return hashed, actual

    def prepare_host_lookup(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        *,
        dummy: bool = False,
        num_reqs: int | None = None,
        num_tokens: int | None = None,
    ) -> None:
        if input_ids is None or query_start_loc is None or ngram_context is None:
            raise ValueError(
                "PLE host lookup requires IDs, query offsets and committed context"
            )
        capacity = input_ids.numel()
        if capacity > self.max_total_tokens:
            raise ValueError("PLE token capacity exceeds its fixed buffer")
        uva = getattr(self, "_uva_lookup", None)
        if uva is not None:
            if num_reqs is None or num_tokens is None:
                raise ValueError("UVA lookup requires CPU-known request/token counts")
            uva.prepare(
                input_ids,
                query_start_loc,
                ngram_context,
                self.prepared_embeddings,
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                dummy=dummy,
            )
            self._prepared = True
            self._prepared_capacity = capacity
            return
        if self._copy_event is not None:
            # The CPU staging allocation cannot be overwritten while its
            # previous asynchronous H2D copy is still reading it.
            self._copy_event.synchronize()
        self.prepared_embeddings.zero_()
        actual = 0
        shared = self._shared_lookup is not None
        if shared and not dummy:

            def produce(rows):
                hashed, actual = self._host_hash_inputs(
                    input_ids, query_start_loc, ngram_context, capacity
                )
                if actual:
                    torch.index_select(
                        self.ngram_embedding.weight,
                        0,
                        hashed.reshape(-1),
                        out=rows[:actual].view(-1, self.head_dim),
                    )
                return actual

            actual = self._shared_lookup.exchange(
                produce, self._host_stage.reshape(-1, self.embedding_dim), capacity
            )
            if actual:
                source = self._host_stage[:actual].reshape(actual, self.embedding_dim)
        elif not dummy:
            hashed, actual = self._host_hash_inputs(
                input_ids, query_start_loc, ngram_context, capacity
            )
            if actual:
                shard = self.ngram_embedding.shard_indices
                local = hashed.reshape(-1) - shard.org_vocab_start_index
                owned = (local >= 0) & (local < self.ngram_embedding.weight.shape[0])
                safe = local.masked_fill(~owned, 0)
                stage = self._host_stage[:actual].reshape(-1, self.head_dim)
                torch.index_select(self.ngram_embedding.weight, 0, safe, out=stage)
                stage.masked_fill_(~owned.unsqueeze(-1), 0)
                source = stage.reshape(actual, self.embedding_dim)
        if actual:
            target = self.prepared_embeddings[:actual]
            target.copy_(source, non_blocking=True)
            if target.device.type != "cpu":
                if self._copy_event is None:
                    self._copy_event = torch.get_device_module(target.device).Event()
                self._copy_event.record()
            if not shared and self.tp_size > 1:
                target.copy_(tensor_model_parallel_all_reduce(target))
        self._prepared = True
        self._prepared_capacity = capacity

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids is None or query_start_loc is None or ngram_context is None:
            raise ValueError("PLE input metadata must not be omitted")
        if not self._prepared:
            raise RuntimeError(
                "PLE host lookup was not prepared outside the model graph"
            )
        count = input_ids.numel()
        if count > self._prepared_capacity:
            raise ValueError("PLE forward exceeds its prepared token capacity")
        return self.prepared_embeddings[:count]

    def _finish_host_lookup_loading(self):
        state = self._ple_embedding_load_state
        expected = set(_PLE_HASH_BUFFER_NAMES)
        if (
            state["shards"] != set(range(self.split_ngram_parts))
            or not expected <= state["buffers"]
        ):
            raise ValueError("UVA requires every PLE checkpoint shard and hash buffer")
        self._uva_lookup.finish_loading(
            self.layer_multipliers,
            self.ngram_heads_vocab_sizes,
            self.ngram_heads_offsets,
        )

    @validate_ple_embedding_weights
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and shards within the PLE coverage transaction."""

        persistent_buffers = {
            name: getattr(self, name) for name in _PLE_HASH_BUFFER_NAMES
        }
        loaded: set[str] = set()
        state = getattr(self, "_ple_embedding_load_state", None)
        seen_shards = state["shards"] if state is not None else set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if buffer.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"
                    )
                buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
                loaded.add(name)
                if state is not None:
                    state["buffers"].add(name)
                continue
            if name.startswith(shard_prefix) and name.endswith(".weight"):
                shard_text = name[len(shard_prefix) : -len(".weight")]
                if not shard_text.isdigit():
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE embedding shard index {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}"
                    )
                if shard_index in seen_shards:
                    raise ValueError(f"Duplicate PLE checkpoint shard {shard_index}")
                seen_shards.add(shard_index)
                embedding = self.ngram_embedding
                checkpoint_start = validate_ple_embedding_shard(
                    embedding, loaded_weight, shard_index, self.split_ngram_parts
                )
                if (
                    self._shared_lookup is not None
                    or getattr(self, "_uva_lookup", None) is not None
                ) and self.tp_rank != 0:
                    # Consumers validate checkpoint shape/name completeness but
                    # keep no duplicate full table and touch no table payload.
                    loaded.add("ngram_embedding.weight")
                    continue
                copy_ple_embedding_shard_(
                    embedding.weight.data,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                    tp_start=embedding.shard_indices.org_vocab_start_index,
                    tp_end=embedding.shard_indices.org_vocab_end_index,
                )
                loaded.add("ngram_embedding.weight")
                continue
            regular_weights.append((name, loaded_weight))

        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        return loaded
