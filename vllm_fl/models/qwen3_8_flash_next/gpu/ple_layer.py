# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-vendor Qwen3.8-Flash-Next position-learning enhancement layers."""

from collections.abc import Iterable, Sequence
from functools import wraps

import torch
from flaggems_vllm import ple_short_conv_
from torch import nn

from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from ..common.hyperconnection import GroupedGemmaRMSNorm
from ..common.ple import copy_ple_embedding_shard_
from ..common.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
)
from ..config import Qwen3_8FlashNextTextConfig
from .ops.ple_fusion import (
    PLE_FUSION_MIN_TOKENS,
    ple_gate_norm,
    ple_prefill_short_conv_,
)

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime_64(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


class Qwen3_8FlashNextPLEGroupedNorm(GroupedGemmaRMSNorm):
    @property
    def eps(self) -> float:
        return self.variance_epsilon


def validate_ple_embedding_weights(load_weights):
    """Validate row coverage at the outermost model-loading boundary.

    AutoWeightsLoader groups *adjacent* module prefixes, so one PLE table can
    receive multiple calls as checkpoint files are streamed. Nested loaders
    participate in the same transaction; a direct embedding call is also a
    complete transaction. Only bookkeeping is retained, never checkpoint tensors.
    """

    @wraps(load_weights)
    def checked_load(model, weights, *args, **kwargs):
        owners = [
            module
            for module in model.modules()
            if isinstance(module, Qwen3_8FlashNextNGramEmbedding)
            and getattr(module, "_ple_embedding_load_state", None) is None
        ]
        for module in owners:
            module._ple_embedding_load_state = {"shards": set(), "full": False}
        try:
            loaded = load_weights(model, weights, *args, **kwargs)
            for module in owners:
                seen = module._ple_embedding_load_state["shards"]
                missing = sorted(module._required_embedding_shards() - seen)
                if seen and missing:
                    raise ValueError(
                        f"Missing PLE embedding shards for this TP rank: {missing}"
                    )
            return loaded
        finally:
            for module in owners:
                del module._ple_embedding_load_state

    return checked_load


class Qwen3_8FlashNextNGramEmbedding(nn.Module):
    def __init__(
        self,
        config: Qwen3_8FlashNextTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        max_num_reqs: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_num_reqs = int(max_num_reqs)
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        max_multiplier = ((1 << 63) - 1) // self.unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        seed = int(getattr(config, "seed", 1234))
        base_seed = seed + _PLE_LAYER_PRIME * ple_dense_layer_id
        multipliers = []
        for index in range(self.ngram_size):
            value = base_seed + _SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        ngram_vocab_size_base = int(config.ngram_vocab_size_base)
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(self.ngram_heads):
            global_head = ple_dense_layer_id * self.ngram_heads + local_head
            size = _nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab_size = ((offset + divisor - 1) // divisor) * divisor
        self.ngram_embedding = VocabParallelEmbedding(
            padded_vocab_size,
            self.head_dim,
            padding_size=divisor,
            prefix=f"{prefix}.ngram_embedding",
        )
        self.register_buffer(
            "positions_buffer",
            torch.arange(max_total_tokens, dtype=torch.int64),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1).long()
        query_start_loc = query_start_loc.long()
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]
        if num_tokens > self.positions_buffer.numel():
            raise ValueError(
                f"PLE received {num_tokens} tokens, but its workspace supports "
                f"at most {self.positions_buffer.numel()}"
            )
        if num_reqs > self.max_num_reqs:
            raise ValueError(
                f"PLE received {num_reqs} requests, but its workspace supports "
                f"at most {self.max_num_reqs}"
            )
        if num_reqs == 0:
            if num_tokens:
                raise ValueError("PLE received token rows without a request boundary")
            return input_ids.new_empty((0, self.embedding_dim))
        if ngram_context.ndim != 2 or ngram_context.shape[0] < num_reqs:
            raise ValueError("PLE ngram_context must contain one row per request")

        if ngram_context.shape[1] != self.ngram_size - 1:
            raise ValueError("PLE ngram_context has the wrong history width")
        if num_tokens == 0:
            empty_ids = input_ids.new_empty((0, self.ngram_heads))
            return self.ngram_embedding(empty_ids).flatten(-2)

        positions = self.positions_buffer[:num_tokens]
        requests = torch.searchsorted(query_start_loc, positions, right=True) - 1
        valid_tokens = (requests >= 0) & (requests < num_reqs)
        safe_requests = requests.clamp(0, num_reqs - 1)
        columns = positions - query_start_loc[safe_requests]
        ngram_context = ngram_context[:num_reqs].to(
            device=input_ids.device, dtype=torch.long
        )

        # Gather only the n-gram window of each flattened token. Work and
        # intermediate storage depend on this step's token bucket, never on
        # num_reqs * max_num_batched_tokens. No host read of GPU lengths is
        # needed, so request boundaries may change during graph replay.
        eos = input_ids.new_full((), self.eos_token_id)
        shifted = [input_ids]
        in_segment = valid_tokens
        for shift in range(1, self.ngram_size):
            source_columns = columns - shift
            from_input = source_columns >= 0
            token = input_ids[(positions - shift).clamp_min(0)]
            history_column = (self.ngram_size - 1 + source_columns).clamp(
                0, self.ngram_size - 2
            )
            history = ngram_context[safe_requests, history_column]
            previous = torch.where(from_input, token, history)
            # An EOS at the current token does not erase its own history;
            # an EOS in the preceding window masks all older history.
            in_segment = in_segment & (previous != self.eos_token_id)
            shifted.append(torch.where(in_segment, previous, eos))

        id_blocks = []
        mixed = shifted[0] * self.layer_multipliers[0]
        for ngram in range(2, self.ngram_size + 1):
            index = ngram - 1
            mixed = torch.bitwise_xor(
                mixed, shifted[index] * self.layer_multipliers[index]
            )
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(torch.where(valid_tokens.unsqueeze(-1), ids, eos))
        ngram_ids = torch.cat(id_blocks, dim=-1)
        return self.ngram_embedding(ngram_ids).flatten(-2)

    def _required_embedding_shards(self) -> set[int]:
        embedding = self.ngram_embedding
        size = (
            embedding.org_vocab_size + self.split_ngram_parts - 1
        ) // self.split_ngram_parts
        start = embedding.shard_indices.org_vocab_start_index
        end = min(embedding.shard_indices.org_vocab_end_index, embedding.org_vocab_size)
        return (
            set(range(start // size, (end - 1) // size + 1)) if start < end else set()
        )

    @validate_ple_embedding_weights
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
        state = self._ple_embedding_load_state
        seen_shards = state["shards"]

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
            if name == "ngram_embedding.weight":
                if state["full"] or seen_shards:
                    raise ValueError("Duplicate or mixed full/sharded PLE embedding")
                state["full"] = True
            if name.startswith(shard_prefix) and name.endswith(".weight"):
                shard_text = name[len(shard_prefix) : -len(".weight")]
                if not shard_text.isdigit():
                    raise ValueError(f"Invalid PLE embedding shard name: {name}")
                shard_index = int(shard_text)
                if state["full"]:
                    raise ValueError("Mixed full/sharded PLE embedding")
                if shard_index in seen_shards:
                    raise ValueError(f"Duplicate PLE embedding shard {shard_index}")
                seen_shards.add(shard_index)
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
                copy_ple_embedding_shard_(
                    embedding.weight.data,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                    tp_start=embedding.shard_indices.org_vocab_start_index,
                    tp_end=embedding.shard_indices.org_vocab_end_index,
                )
                continue
            regular_weights.append((name, loaded_weight))

        # The outer vLLM check sees only parameter names. Certify the table
        # only once every locally required row has arrived across all fragments.
        if seen_shards and self._required_embedding_shards() <= seen_shards:
            loaded.add("ngram_embedding.weight")
        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        return loaded


class Qwen3_8FlashNextPLELayer(nn.Module, MambaBase):
    def __init__(
        self,
        config: Qwen3_8FlashNextTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.model_config: ModelConfig = model_config
        self.cache_config: CacheConfig = cache_config
        self.layer_idx = layer_idx
        self.ple_dense_layer_id = (
            int(ple_dense_layer_id)
            if ple_dense_layer_id is not None
            else int(layer_idx)
        )
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.activation = "silu"
        self.ple_embedding: nn.Module = Qwen3_8FlashNextNGramEmbedding(
            config,
            int(config.ple_embed_dim),
            self.ple_dense_layer_id,
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.scheduler_config.max_num_seqs,
            f"{prefix}.ple_embedding",
        )
        self.key_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hc_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.value_proj",
        )
        norm_args = (
            self.hc_hidden_size,
            config.rms_norm_eps,
            self.hidden_size,
            model_config.dtype,
        )
        self.norm_key = Qwen3_8FlashNextPLEGroupedNorm(*norm_args)
        self.norm_query = Qwen3_8FlashNextPLEGroupedNorm(*norm_args)
        self.norm_conv = Qwen3_8FlashNextPLEGroupedNorm(*norm_args)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=model_config.dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        self.conv1d.weight._no_reinit = True
        self.kv_cache = (torch.tensor([]),)
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    def get_attn_backend(self) -> type[PleShortConvAttentionBackend]:
        return PleShortConvAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> Sequence[tuple[int, ...]]:
        # vLLM 0.24 has no explicit ``num_spec`` parameter; folding it into
        # the effective kernel yields the same state length:
        # (kernel - 1) + num_spec.
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=self.hc_hidden_size,
            conv_kernel=self.conv_state_len + self.num_spec_tokens + 1,
        )

    def _short_conv_fallback(self, inputs: torch.Tensor) -> torch.Tensor:
        # Profiling has no cache metadata or persistent state updates.
        output = torch.empty_like(inputs)
        state = torch.empty(
            (0, inputs.shape[1], self.conv_state_len),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        ple_short_conv_(
            inputs,
            output,
            state,
            self.conv1d.weight.squeeze(1),
            None,
            dilation=self.short_conv_dilation,
        )
        return output

    def _short_conv_dilated_decode_batched(
        self,
        x_d,
        conv_state,
        conv_weights,
        state_indices_tensor_d,
        has_initial_states_d,
    ) -> torch.Tensor:
        output = torch.empty_like(x_d)
        ple_short_conv_(
            x_d,
            output,
            conv_state,
            conv_weights,
            state_indices_tensor_d,
            dilation=self.short_conv_dilation,
            has_initial_state=has_initial_states_d,
            null_block_id=NULL_BLOCK_ID,
        )
        return output

    def _short_conv_dilated_prefill_batched(
        self,
        x_p,
        metadata,
        conv_state,
        conv_weights,
        state_indices_tensor_p,
        num_prefills,
        num_decode_tokens,
        num_prefill_tokens,
        output=None,
    ) -> torch.Tensor | None:
        starts = metadata.non_spec_query_start_loc
        if starts is None or metadata.has_initial_states_p is None:
            raise ValueError(
                "PLE prefill requires query offsets and initial-state flags"
            )
        starts = starts[-num_prefills - 1 :]
        if num_decode_tokens:
            starts = starts - num_decode_tokens
        result = output if output is not None else torch.empty_like(x_p)
        if not x_p.numel():
            return None if output is not None else result
        # Keep the existing measured large-prefill launch policy.
        fused = ple_prefill_short_conv_(
            x_p,
            result,
            conv_state,
            conv_weights,
            starts,
            state_indices_tensor_p[:num_prefills],
            metadata.has_initial_states_p[:num_prefills],
            num_prefills=num_prefills,
            max_len=metadata.max_prefill_query_len,
            state_len=self.conv_state_len,
            kernel_width=self.conv_kernel_size,
            dilation=self.short_conv_dilation,
            null_block_id=NULL_BLOCK_ID,
            min_tokens=PLE_FUSION_MIN_TOKENS,
        )
        if not fused:
            ple_short_conv_(
                x_p,
                result,
                conv_state,
                conv_weights,
                state_indices_tensor_p[:num_prefills],
                dilation=self.short_conv_dilation,
                query_start_loc=starts,
                has_initial_state=metadata.has_initial_states_p[:num_prefills],
                null_block_id=NULL_BLOCK_ID,
            )
        return None if output is not None else result

    def _short_conv_dilated_spec_batched(
        self,
        x_spec,
        conv_state,
        conv_weights,
        spec_state_indices_tensor,
        spec_query_start_loc,
        num_accepted_tokens,
        spec_query_len,
    ) -> torch.Tensor:
        output = torch.empty_like(x_spec)
        requests = spec_state_indices_tensor.numel()
        ple_short_conv_(
            x_spec,
            output,
            conv_state,
            conv_weights,
            spec_state_indices_tensor,
            dilation=self.short_conv_dilation,
            query_start_loc=spec_query_start_loc[: requests + 1],
            accepted_tokens=num_accepted_tokens[:requests],
            max_query_len=spec_query_len,
            null_block_id=NULL_BLOCK_ID,
        )
        return output

    def _short_conv_dilated_dispatch(
        self,
        inputs: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        num_prefills = metadata.num_prefills
        num_decodes = metadata.num_decodes
        num_decode_tokens = metadata.num_decode_tokens
        num_prefill_tokens = metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        has_spec = metadata.spec_sequence_masks is not None
        x = inputs[: metadata.num_actual_tokens]

        # A pure-prefill step reaches the fused path before the generic
        # spec/non-spec split, so it also avoids the split and vstack copies.
        if has_prefill and not has_decode and not has_spec:
            state_indices_tensor = metadata.state_indices_tensor
            if state_indices_tensor is None:
                raise ValueError("PLE prefill requires state indices")
            return self._short_conv_dilated_prefill_batched(
                x_p=x,
                metadata=metadata,
                conv_state=conv_state,
                conv_weights=conv_weights,
                state_indices_tensor_p=state_indices_tensor[:num_prefills],
                num_prefills=num_prefills,
                num_decode_tokens=0,
                num_prefill_tokens=num_prefill_tokens,
                output=(
                    output[: metadata.num_actual_tokens] if output is not None else None
                ),
            )

        # Split spec / non-spec tokens.
        if has_spec:
            if has_prefill or has_decode:
                assert metadata.spec_token_indx is not None
                assert metadata.non_spec_token_indx is not None
                x_spec = x.index_select(0, metadata.spec_token_indx.long())
                x_non_spec = x.index_select(0, metadata.non_spec_token_indx.long())
            else:
                x_spec = x
                x_non_spec = None
        else:
            x_spec = None
            x_non_spec = x

        spec_output = None
        # 1. Run the multi-query speculative-decode part.
        if has_spec:
            assert metadata.spec_state_indices_tensor is not None
            assert metadata.spec_query_start_loc is not None
            assert metadata.num_accepted_tokens is not None
            spec_output = self._short_conv_dilated_spec_batched(
                x_spec=x_spec,
                conv_state=conv_state,
                conv_weights=conv_weights,
                spec_state_indices_tensor=metadata.spec_state_indices_tensor[
                    : metadata.num_spec_decodes
                ],
                spec_query_start_loc=metadata.spec_query_start_loc,
                num_accepted_tokens=metadata.num_accepted_tokens,
                spec_query_len=metadata.spec_query_len,
            )

        # 2. Run regular decode and prefill requests.
        conv_out_non_spec = None
        state_indices_tensor = metadata.state_indices_tensor
        if x_non_spec is not None:
            assert state_indices_tensor is not None
            if has_prefill:
                state_indices_tensor_d, state_indices_tensor_p = torch.split(
                    state_indices_tensor,
                    [num_decodes, num_prefills],
                    dim=0,
                )
                x_d, x_p = torch.split(
                    x_non_spec,
                    [num_decode_tokens, num_prefill_tokens],
                    dim=0,
                )
                non_spec_parts: list[torch.Tensor] = []
                if has_decode:
                    non_spec_parts.append(
                        self._short_conv_dilated_decode_batched(
                            x_d=x_d,
                            conv_state=conv_state,
                            conv_weights=conv_weights,
                            state_indices_tensor_d=state_indices_tensor_d,
                            has_initial_states_d=metadata.has_initial_states_d,
                        )
                    )
                non_spec_parts.append(
                    self._short_conv_dilated_prefill_batched(
                        x_p=x_p,
                        metadata=metadata,
                        conv_state=conv_state,
                        conv_weights=conv_weights,
                        state_indices_tensor_p=state_indices_tensor_p,
                        num_prefills=num_prefills,
                        num_decode_tokens=num_decode_tokens,
                        num_prefill_tokens=num_prefill_tokens,
                    )
                )
                conv_out_non_spec = torch.vstack(non_spec_parts)
            else:
                conv_out_non_spec = self._short_conv_dilated_decode_batched(
                    x_d=x_non_spec,
                    conv_state=conv_state,
                    conv_weights=conv_weights,
                    state_indices_tensor_d=state_indices_tensor[: x_non_spec.size(0)],
                    has_initial_states_d=metadata.has_initial_states_d,
                )

        # 3. Merge both parts back into the original token order.
        if has_spec and conv_out_non_spec is not None:
            assert metadata.spec_token_indx is not None
            assert metadata.non_spec_token_indx is not None
            assert spec_output is not None
            output = x.new_empty((metadata.num_actual_tokens, x.size(-1)))
            output.index_copy_(0, metadata.spec_token_indx, spec_output)
            output.index_copy_(0, metadata.non_spec_token_indx, conv_out_non_spec)
            return output
        elif has_spec:
            assert spec_output is not None
            return spec_output
        if conv_out_non_spec is None:
            return x
        return conv_out_non_spec

    def _short_conv(
        self, inputs: torch.Tensor, output: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return self._short_conv_fallback(inputs)

        if not isinstance(attn_metadata, dict):
            raise RuntimeError(
                "PLE short-conv expects per-layer attention metadata dict "
                f"during inference, got {type(attn_metadata).__name__}."
            )

        layer_attn_metadata = attn_metadata.get(self.prefix)
        if layer_attn_metadata is None:
            raise RuntimeError(
                f"Missing short-conv metadata for layer '{self.prefix}'. "
                "This would bypass conv-state updates and is not allowed."
            )
        if not isinstance(layer_attn_metadata, PleShortConvAttentionMetadata):
            raise TypeError(
                "Expected PleShortConvAttentionMetadata for layer "
                f"'{self.prefix}', got "
                f"{type(layer_attn_metadata).__name__}."
            )

        conv_state = self.kv_cache[0]
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weights = self.conv1d.weight.squeeze(1)

        state_capacity = self.conv_state_len + self.num_spec_tokens
        if state_capacity > 0:
            if conv_state.size(-1) < state_capacity:
                raise RuntimeError(
                    "PLE short-conv cache is smaller than expected for "
                    f"dilated convolution: got {conv_state.size(-1)}, "
                    f"expect at least {state_capacity}."
                )
            conv_state = conv_state[..., -state_capacity:]
        return self._short_conv_dilated_dispatch(
            inputs,
            layer_attn_metadata,
            conv_state,
            conv_weights,
            output=output,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same "
                f"token length, got {input_ids.shape[0]} and "
                f"{hidden_states.shape[0]}"
            )
        embeddings = self.ple_embedding(input_ids, query_start_loc, ngram_context)
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)
        token_count = hidden_states.shape[0]
        key = key.reshape(token_count, self.hc_count, self.hidden_size)
        query = hidden_states.reshape(token_count, self.hc_count, self.hidden_size)
        gated_value, normalized_grouped = ple_gate_norm(
            key,
            query,
            value,
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            self.hc_count,
            self.norm_conv.eps,
        )
        normalized = normalized_grouped.flatten(-2)
        conv_output = torch.zeros_like(normalized)
        torch.ops.vllm.qwen3_8_flash_next_ple_short_conv(
            normalized,
            conv_output,
            self.prefix,
        )
        return gated_value.flatten(-2) + conv_output


def qwen3_8_flash_next_ple_short_conv(
    inputs: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    result = layer._short_conv(inputs, output=output)
    if result is not None:
        output[: result.shape[0]].copy_(result)


def qwen3_8_flash_next_ple_short_conv_fake(
    inputs: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen3_8_flash_next_ple_short_conv",
    op_func=qwen3_8_flash_next_ple_short_conv,
    mutates_args=["output"],
    fake_impl=qwen3_8_flash_next_ple_short_conv_fake,
)


__all__ = [
    "Qwen3_8FlashNextNGramEmbedding",
    "Qwen3_8FlashNextPLEGroupedNorm",
    "Qwen3_8FlashNextPLELayer",
]
