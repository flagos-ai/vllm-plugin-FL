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

from vllm_fl.models.qwen3_8_flash_next.config import Qwen3_8FlashNextTextConfig
from ..common.ple import copy_ple_embedding_shard_

from ..common.ple import _MASK64, _SPLITMIX_GAMMA, _PLE_LAYER_PRIME, _splitmix64, _nth_prime_after


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
        self.weight = nn.Parameter(
            torch.empty(rows if allocate else 0, embedding_dim, dtype=dtype, device="cpu"),
            requires_grad=False,
        )

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
    for req, (begin, end) in enumerate(zip(starts, starts[1:])):
        if begin == end:
            continue
        tokens = torch.cat((context[req].to(torch.int64), ids[begin:end]))
        positions = torch.arange(tokens.numel(), dtype=torch.int64)
        eos_pos = torch.where(tokens == eos_token_id, positions, -1)
        previous_eos = torch.cat(
            (eos_pos.new_full((1,), -1), torch.cummax(eos_pos, dim=0).values[:-1])
        )
        distance = positions - previous_eos - 1
        shifted = [tokens]
        for shift in range(1, ngram_size):
            source = positions - shift
            values = tokens[source.clamp_min(0)]
            shifted.append(
                torch.where(
                    (source >= 0) & (distance >= shift),
                    values,
                    values.new_full((), eos_token_id),
                )
            )
        mixed = shifted[0] * multipliers[0]
        for order in range(2, ngram_size + 1):
            mixed = torch.bitwise_xor(
                mixed, shifted[order - 1] * multipliers[order - 1]
            )
            h = (order - 2) * heads_per_ngram
            v = mixed[ngram_size - 1 :].unsqueeze(-1)
            out[begin:end, h : h + heads_per_ngram] = (
                torch.remainder(v, vocab_sizes[h : h + heads_per_ngram])
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
        if host_lookup_mode not in {"sharded", "shared_buffer"}:
            raise ValueError("N-gram lookup mode must be sharded or shared_buffer")
        self.host_lookup_mode = host_lookup_mode
        self.tp_rank = tp_rank
        self._shared_lookup = None
        if host_lookup_mode == "shared_buffer":
            from vllm.distributed import get_tp_group
            from vllm.distributed.parallel_state import in_the_same_node_as
            from vllm_fl.models.qwen3_8_flash_next.host_lookup import SharedNgramBuffer

            group = get_tp_group()
            if group.world_size > 1 and not all(in_the_same_node_as(group.cpu_group)):
                raise NotImplementedError("Shared N-gram results require one TP host/IPC namespace")
            self._shared_lookup = SharedNgramBuffer(
                group, max_total_tokens, embedding_dim, params_dtype,
            )
        bound = max(1, (((1 << 63) - 1) // self.unigram_vocab_size) // 2)
        seed = (
            int(getattr(config, "seed", 1234)) + _PLE_LAYER_PRIME * ple_dense_layer_id
        )
        multipliers = [
            2 * (_splitmix64(seed + _SPLITMIX_GAMMA * (i + 1)) % bound) + 1
            for i in range(self.ngram_size)
        ]
        sizes, offsets, total = [], [], 0
        for local_head in range(self.ngram_heads):
            global_head = ple_dense_layer_id * self.ngram_heads + local_head
            size = _nth_prime_after(
                int(config.ngram_vocab_size_base) - 1, global_head + 1
            )
            sizes.append(size)
            offsets.append(total)
            total += size
        self._host_buffer_names = (
            "layer_multipliers",
            "ngram_heads_vocab_sizes",
            "ngram_heads_offsets",
        )
        for name, values in zip(self._host_buffer_names, (multipliers, sizes, offsets)):
            self.register_buffer(
                name, torch.tensor(values, dtype=torch.int64, device="cpu")
            )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab_size = ((total + divisor - 1) // divisor) * divisor
        self.ngram_embedding = HostVocabEmbedding(
            padded_vocab_size,
            self.head_dim,
            tp_rank=0 if self._shared_lookup is not None else tp_rank,
            tp_size=1 if self._shared_lookup is not None else self.tp_size,
            dtype=params_dtype,
            allocate=self._shared_lookup is None or tp_rank == 0,
        )
        # Follow the model construction device, but never pin the huge table.
        device = torch.empty(0).device
        self.register_buffer(
            "prepared_embeddings",
            torch.empty(
                max_total_tokens, embedding_dim, dtype=params_dtype, device=device
            ),
            persistent=False,
        )
        self._host_stage = torch.empty(
            max_total_tokens,
            self.ngram_heads,
            self.head_dim,
            dtype=params_dtype,
            device="cpu",
            pin_memory=device.type != "cpu",
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

    def prepare_host_lookup(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        *,
        dummy: bool = False,
    ) -> None:
        if input_ids is None or query_start_loc is None or ngram_context is None:
            raise ValueError(
                "PLE host lookup requires IDs, query offsets and committed context"
            )
        capacity = input_ids.numel()
        if capacity > self.max_total_tokens:
            raise ValueError("PLE token capacity exceeds its fixed buffer")
        if self._copy_event is not None:
            # The CPU staging allocation cannot be overwritten while its
            # previous asynchronous H2D copy is still reading it.
            self._copy_event.synchronize()
        self.prepared_embeddings.zero_()
        if self._shared_lookup is not None and not dummy:
            def produce(rows):
                starts = query_start_loc.detach().to(device="cpu", dtype=torch.int64)
                nreq, actual = starts.numel() - 1, int(starts[-1])
                if nreq > self.max_num_reqs or not 0 <= actual <= capacity:
                    raise ValueError("PLE request/token count exceeds fixed buffers")
                ids = input_ids[:actual].detach().to(device="cpu", dtype=torch.int64)
                context = ngram_context[:nreq].detach().to(device="cpu", dtype=torch.int64)
                hashed = compute_ngram_ids_cpu(
                    ids, starts, context,
                    multipliers=self.layer_multipliers,
                    vocab_sizes=self.ngram_heads_vocab_sizes,
                    vocab_offsets=self.ngram_heads_offsets,
                    heads_per_ngram=self.heads_per_ngram,
                    eos_token_id=self.eos_token_id,
                )
                if actual:
                    torch.index_select(
                        self.ngram_embedding.weight, 0, hashed.reshape(-1),
                        out=rows[:actual].view(-1, self.head_dim),
                    )
                return actual

            actual = self._shared_lookup.exchange(
                produce, self._host_stage.reshape(-1, self.embedding_dim), capacity,
            )
            if actual:
                target = self.prepared_embeddings[:actual]
                target.copy_(self._host_stage[:actual].reshape(actual, self.embedding_dim), non_blocking=True)
                if target.device.type != "cpu":
                    if self._copy_event is None:
                        self._copy_event = torch.get_device_module(target.device.type).Event()
                    self._copy_event.record()
            self._prepared = True
            self._prepared_capacity = capacity
            return
        if not dummy:
            starts = query_start_loc.detach().to(device="cpu", dtype=torch.int64)
            nreq = starts.numel() - 1
            actual = int(starts[-1])
            if nreq > self.max_num_reqs or actual > capacity or actual < 0:
                raise ValueError("PLE request/token count exceeds its fixed buffers")
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
            if actual:
                shard = self.ngram_embedding.shard_indices
                local = hashed.reshape(-1) - shard.org_vocab_start_index
                owned = (local >= 0) & (local < self.ngram_embedding.weight.shape[0])
                safe = local.masked_fill(~owned, 0)
                stage = self._host_stage[:actual].reshape(-1, self.head_dim)
                torch.index_select(self.ngram_embedding.weight, 0, safe, out=stage)
                stage.masked_fill_(~owned.unsqueeze(-1), 0)
                target = self.prepared_embeddings[:actual]
                target.copy_(
                    stage.reshape(actual, self.embedding_dim), non_blocking=True
                )
                if target.device.type != "cpu":
                    if self._copy_event is None:
                        self._copy_event = torch.get_device_module(
                            target.device
                        ).Event()
                    self._copy_event.record()
                if self.tp_size > 1:
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""

        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set[str] = set()
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
                embedding = self.ngram_embedding
                shard_size = (
                    embedding.org_vocab_size + self.split_ngram_parts - 1
                ) // self.split_ngram_parts
                checkpoint_start = shard_index * shard_size
                expected_rows = max(
                    0,
                    min(shard_size, embedding.org_vocab_size - checkpoint_start),
                )
                expected_shape = (expected_rows, embedding.embedding_dim)
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard {shard_index}: "
                        f"expected {expected_shape}, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                if self._shared_lookup is not None and self.tp_rank != 0:
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
