# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 model, attention projections and BF16 indexer orchestration.

Kernels and platform choices belong to dispatch backends, not this module.
"""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable
from itertools import islice

import regex as re
import torch
import torch.nn.functional as F
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config, PretrainedConfig

import vllm.model_executor.layers.fused_moe as fused_moe
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.fused_moe import (
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsPP
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.attention.selector import get_attn_backend

from vllm_fl.configs.hy_v4 import HYV4Config
from vllm_fl.dispatch import CachedOp

logger = init_logger(__name__)


_indexer_cache_write = CachedOp("bf16_indexer_cache_write")


_indexer_decode = CachedOp("bf16_indexer_decode")


_top_k_per_row_prefill = CachedOp("top_k_per_row_prefill")


def _paged_sequence(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Gather one request's BF16 indexer keys in request-token order."""
    if seq_len <= 0:
        return kv_cache.new_empty((0, kv_cache.shape[-1]))
    block_size = kv_cache.shape[1]
    num_blocks = (seq_len + block_size - 1) // block_size
    blocks = block_table[:num_blocks].to(torch.long)
    offsets = torch.arange(seq_len, device=kv_cache.device, dtype=torch.long)
    slots = blocks.index_select(0, offsets // block_size) * block_size
    slots = slots + offsets.remainder(block_size)
    return kv_cache.reshape(-1, kv_cache.shape[-1]).index_select(0, slots)


def _select_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    topk: int,
    output: torch.Tensor,
) -> None:
    """Select request-local key positions for one query token."""
    output.fill_(-1)
    if keys.shape[0] == 0:
        return

    # HYV4 lightning-indexer score: weighted sum of per-head ReLU(QK).
    # Keep the non-linearity before the head reduction; moving the reduction
    # ahead of ReLU changes the selected tokens when individual heads disagree.
    scores = torch.matmul(keys, q.transpose(0, 1))
    logits = (torch.relu(scores) * weights.to(scores.dtype).unsqueeze(0)).sum(dim=-1)
    row_starts = torch.zeros_like(output[:1])
    row_ends = torch.full_like(output[:1], keys.shape[0])
    # Sparse attention consumes the selected positions as a set, so provider
    # implementations need not return them in descending-score order.
    _top_k_per_row_prefill(
        logits.unsqueeze(0),
        row_starts,
        row_ends,
        output[:topk].unsqueeze(0),
    )


def _hyv4_bf16_sparse_attn_indexer_impl(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    max_model_len: int,
) -> torch.Tensor:
    """Stateful BF16 Indexer implementation behind an opaque graph node."""
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        topk_indices_buffer[: q.shape[0]].fill_(-1)
        return topk_indices_buffer

    layer_name = _resolve_layer_name(k_cache_prefix)
    metadata = attn_metadata[layer_name]
    if not isinstance(metadata, DeepseekV32IndexerMetadata):
        raise TypeError(
            "HYV4 BF16 indexer expected DeepseekV32IndexerMetadata, "
            f"got {type(metadata).__name__}"
        )

    topk_indices_buffer[: q.shape[0]].fill_(-1)
    num_tokens = metadata.slot_mapping.shape[0]
    _indexer_cache_write(k[:num_tokens], kv_cache, metadata.slot_mapping)

    if metadata.num_prefills:
        assert metadata.prefill is not None
        for chunk in metadata.prefill.chunks:
            cu_seq_lens = chunk.cu_seq_lens.tolist()
            gathered = [
                _paged_sequence(
                    kv_cache,
                    chunk.block_table[request_id],
                    cu_seq_lens[request_id + 1] - cu_seq_lens[request_id],
                )
                for request_id in range(chunk.num_reqs)
            ]
            keys = torch.cat(gathered, dim=0)
            starts = chunk.cu_seqlen_ks.tolist()
            ends = chunk.cu_seqlen_ke.tolist()
            sequence_starts = chunk.cu_seq_lens[:-1]
            sequence_ids = torch.searchsorted(
                chunk.cu_seq_lens[1:], chunk.cu_seqlen_ks, right=True
            ).tolist()
            for local_row, (start, end, sequence_id) in enumerate(
                zip(starts, ends, sequence_ids, strict=True)
            ):
                token_row = chunk.token_start + local_row
                request_start = int(sequence_starts[sequence_id].item())
                _select_topk(
                    q[token_row],
                    weights[token_row],
                    keys[start:end],
                    topk_tokens,
                    topk_indices_buffer[token_row],
                )
                valid = topk_indices_buffer[token_row] >= 0
                topk_indices_buffer[token_row, valid] += start - request_start

    if metadata.num_decodes:
        assert metadata.decode is not None
        decode = metadata.decode
        decode_lens = decode.decode_lens
        num_decode_tokens = metadata.num_decode_tokens
        if decode.requires_padding:
            padded_q = pack_seq_triton(q[:num_decode_tokens], decode_lens, pad_value=0)
            padded_weights = pack_seq_triton(
                weights[:num_decode_tokens], decode_lens, pad_value=0
            )
        else:
            padded_q = q[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q.shape[1:]
            )
            padded_weights = weights[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, weights.shape[-1]
            )

        batch_size, next_n = padded_q.shape[:2]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode.seq_lens[:batch_size]
        if seq_lens.ndim == 1:
            seq_lens = seq_lens.unsqueeze(-1)
        count = min(topk_tokens, max_model_len)
        if decode.requires_padding:
            topk_indices = torch.empty(
                (num_padded_tokens, count),
                dtype=topk_indices_buffer.dtype,
                device=padded_q.device,
            )
        else:
            topk_indices = topk_indices_buffer[:num_padded_tokens, :count]
        _indexer_decode(
            padded_q,
            kv_cache.unsqueeze(-2),
            padded_weights.reshape(num_padded_tokens, -1),
            seq_lens,
            decode.block_table,
            decode.schedule_metadata,
            topk_indices,
            next_n=next_n,
            max_context_len=max_model_len,
            clean_logits=False,
        )
        if decode.requires_padding:
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, next_n, count), decode_lens
            )
        topk_indices_buffer[: topk_indices.shape[0], :count].copy_(topk_indices)

    return topk_indices_buffer


def _hyv4_bf16_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    max_model_len: int,
) -> torch.Tensor:
    del hidden_states, k_cache_prefix, kv_cache, q, k, weights
    del topk_tokens, max_model_len
    return topk_indices_buffer


direct_register_custom_op(
    op_name="hyv4_bf16_sparse_attn_indexer",
    op_func=_hyv4_bf16_sparse_attn_indexer_impl,
    mutates_args=["kv_cache", "topk_indices_buffer"],
    fake_impl=_hyv4_bf16_sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)

_hyv4_bf16_sparse_attn_indexer = (
    torch.ops.vllm.hyv4_bf16_sparse_attn_indexer.default
)


class HYV4BF16SparseAttnIndexer(nn.Module):
    """Paged BF16 indexer algorithm backed by dispatched operators."""

    def __init__(
        self,
        *,
        head_dim: int,
        topk_tokens: int,
        cache_config,
        topk_indices_buffer: torch.Tensor,
        prefix: str,
        max_model_len: int,
    ) -> None:
        super().__init__()
        if cache_config is None:
            raise ValueError("HYV4 BF16 indexer requires cache_config")
        self.head_dim = head_dim
        self.topk_tokens = topk_tokens
        self.topk_indices_buffer = topk_indices_buffer
        self.max_model_len = max_model_len
        if head_dim != 128:
            raise ValueError("HYV4 BF16 paged indexer requires head_dim=128")
        if cache_config.block_size != 64:
            raise ValueError("HYV4 BF16 paged indexer requires block_size=64")
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=head_dim,
            dtype=torch.bfloat16,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return _hyv4_bf16_sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q,
            k,
            weights,
            self.topk_indices_buffer,
            self.topk_tokens,
            self.max_model_len,
        )


_SPARSE_LAYER_TYPES = ("sparse_attention", "sparse", "deepseek_sparse_attention")


_WEIGHT_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def compute_skip_topk_layers(config: PretrainedConfig) -> set[int]:
    """Return the backbone layers that reuse a previous layer's top-k indices.

    A "shared" indexer layer performs sparse attention with the indices computed
    by the closest preceding "full" indexer layer, so it does not build its own
    indexer and its checkpoint indexer weights must be skipped.

    Args:
        config: The model config.

    Returns:
        The set of layer indices that share another layer's top-k indices.

    Raises:
        ValueError: If ``indexer_types`` has the wrong length or an unknown
            entry, or if ``index_topk_freq`` is not a positive integer.
    """
    if not hasattr(config, "index_topk"):
        return set()

    num_hidden_layers = config.num_hidden_layers
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is not None:
        if len(indexer_types) != num_hidden_layers:
            raise ValueError(
                "indexer_types must contain one entry per hidden layer: "
                f"expected {num_hidden_layers}, got {len(indexer_types)}."
            )
        invalid_types = sorted(set(indexer_types) - {"full", "shared"})
        if invalid_types:
            raise ValueError(
                f"indexer_types only supports 'full' and 'shared', got {invalid_types}."
            )
        return {
            layer_idx
            for layer_idx, indexer_type in enumerate(indexer_types)
            if indexer_type == "shared"
        }

    freq = getattr(config, "index_topk_freq", 1)
    if not isinstance(freq, int) or freq <= 0:
        raise ValueError(f"index_topk_freq must be a positive integer, got {freq!r}.")
    pattern = getattr(config, "index_topk_pattern", None)
    offset = getattr(config, "index_skip_topk_offset", 2)
    skip_layers: set[int] = set()
    for layer_idx in range(num_hidden_layers):
        if pattern is None:
            if max(layer_idx - offset + 1, 0) % freq != 0:
                skip_layers.add(layer_idx)
        elif 0 <= layer_idx < len(pattern) and pattern[layer_idx] == "S":
            skip_layers.add(layer_idx)
    return skip_layers


def is_skip_topk_indexer_weight(weight_name: str, skip_topk_layers: set[int]) -> bool:
    """Return whether an indexer weight belongs to a top-k sharing layer.

    Args:
        weight_name: Checkpoint weight name.
        skip_topk_layers: Result of `compute_skip_topk_layers`.

    Returns:
        True when the weight is an indexer weight of a layer that has no
        indexer module and therefore must be dropped.
    """
    if ".indexer." not in weight_name or not skip_topk_layers:
        return False
    match = _WEIGHT_LAYER_INDEX_RE.search(weight_name)
    return match is not None and int(match.group(1)) in skip_topk_layers


class Indexer(nn.Module):
    """Lightning indexer selecting the top-k tokens for sparse MLA."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = q_lora_rank

        # No tensor parallelism, just replicated.
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single BF16 GEMM producing
        # [head_dim + n_head]. Both checkpoint tensors are BF16.
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.register_buffer(
            "_weights_scale",
            torch.tensor(
                self.head_dim**-0.5 * self.n_head**-0.5,
                dtype=torch.bfloat16,
            ),
            persistent=False,
        )

        self.topk_indices_buffer = topk_indices_buffer
        if topk_indices_buffer is None:
            raise ValueError("HYV4 sparse attention requires a top-k buffer")
        self.indexer_op = HYV4BF16SparseAttnIndexer(
            head_dim=self.head_dim,
            topk_tokens=self.topk_tokens,
            cache_config=cache_config,
            topk_indices_buffer=topk_indices_buffer,
            prefix=prefix,
            max_model_len=vllm_config.model_config.max_model_len,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        hidden_states, q_quant, k, weights = self.prepare_inputs(
            hidden_states, qr, positions, rotary_emb
        )
        return self.indexer_op(hidden_states, q_quant, k, weights)

    def prepare_inputs(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the BF16 query, key and per-head weights of the indexer."""
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        # Checkpoint (PTM) layout: pe occupies the LAST rope_dim dims.
        q_nope, q_pe = torch.split(
            q, [self.head_dim - self.rope_dim, self.rope_dim], dim=-1
        )

        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        k_nope, k_pe = torch.split(
            k, [self.head_dim - self.rope_dim, self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # RoPE (NeoX) can introduce extra leading dims, so flatten back to the
        # token-major shapes.
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        # Reassemble with the original physical layout: no_pe first, pe last.
        q = torch.cat([q_nope, q_pe], dim=-1)
        # ``k_pe`` is [num_tokens, 1, rope_dim] (MQA).
        k = torch.cat([k_nope, k_pe.squeeze(-2)], dim=-1)

        # The paged-MQA kernels accumulate logits in FP32 and consume the
        # per-head weights as FP32.  W8A8 only changes the internal linear
        # operands; its output is still the model activation dtype.
        weights = (weights * self._weights_scale).float()

        return hidden_states, q, k, weights


class HYV4MLAAttention(nn.Module):
    """Multi-head latent attention with optional sparse lightning indexer.

    Main reference: the DeepSeek-V2 paper and the FlashInfer implementation
    (https://arxiv.org/abs/2405.04434). HY V4 additionally supports an output
    gate (``gated_mla``) and a per-head learnable attention sink.

    The sink is applied by the sink-capable sparse MLA backend selected through
    normal attention dispatch.  If no backend on this platform can consume
    sinks, model construction fails instead of silently changing the architecture.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.layer_idx = layer_idx
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.layer_id = int(prefix.split(".")[-2])
        layer_types = getattr(config, "layer_types", None)
        requested_sparse = (
            hasattr(config, "index_topk")
            and layer_types is not None
            and self.layer_id < len(layer_types)
            and layer_types[self.layer_id] in _SPARSE_LAYER_TYPES
        )
        # Only actual sparse layers may share another layer's top-k indices.
        self.skip_topk = requested_sparse and self.layer_id in compute_skip_topk_layers(
            config
        )
        # The skip pattern only governs backbone layers. MTP/nextn layers
        # (layer_id >= num_hidden_layers) always build a full indexer: they
        # compute indices at draft step 0 and toggle at runtime.
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        is_mtp_layer = (
            num_hidden_layers is not None and self.layer_id >= num_hidden_layers
        )
        self.create_indexer = requested_sparse and (not self.skip_topk or is_mtp_layer)
        self.is_sparse = requested_sparse

        # Do not silently degrade sparse layers into dense attention. Probe the
        # sparse MLA backend directly and fail fast with the real error.
        kv_cache_dtype = cache_config.cache_dtype if cache_config else "auto"
        if self.is_sparse:
            try:
                get_attn_backend(
                    head_size=self.kv_lora_rank + self.qk_rope_head_dim,
                    dtype=torch.get_default_dtype(),
                    kv_cache_dtype=kv_cache_dtype,
                    use_mla=True,
                    has_sink=bool(getattr(config, "learnable_sink", False)),
                    use_sparse=True,
                    num_heads=self.num_local_heads,
                )
            except Exception as exc:
                raise RuntimeError(
                    "HYV4 sparse attention was requested, but no valid sparse MLA "
                    "backend is available for current runtime/config. "
                    "Refusing to fall back to dense attention."
                ) from exc

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.q_a_proj = None
        self.kv_a_proj_with_mqa = None
        if self.q_lora_rank is not None:
            self.q_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_a_proj",
                disable_tp=True,
            )
            self.kv_a_proj_with_mqa = MergedColumnParallelLinear(
                self.hidden_size,
                [self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
                disable_tp=True,
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        self.q_a_layernorm = None
        self.q_b_proj = None
        self.q_proj = None
        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        self.indexer_rope_emb: nn.Module | None
        self.indexer: Indexer | None
        if self.create_indexer:
            # The checkpoint stores indexer q_pe/k_pe in interleaved
            # (Megatron/PTM) layout, so the indexer must use interleaved RoPE
            # (is_neox_style=False) like the main attention path. Using NeoX
            # here loses the relative-position dependence and corrupts the DSA
            # top-k selection.
            self.indexer_rope_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=False,
            )
            # The indexer projects its queries from the MLA q_lora activations,
            # so a sparse layer requires a query down-projection.
            assert q_lora_rank is not None, (
                "HYV4 sparse attention requires q_lora_rank to be set"
            )
            self.indexer = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )
        else:
            self.indexer_rope_emb = None
            self.indexer = None

        self.gated_mla = bool(getattr(config, "gated_mla", False))
        self.linear_gate: ColumnParallelLinear | None
        if self.gated_mla:
            if config.gating_type == "headwise":
                self.gate_projection_size_per_head = 1
            elif config.gating_type == "elementwise":
                self.gate_projection_size_per_head = self.v_head_dim
            else:
                raise ValueError(f"Unknown gating type: {config.gating_type}")
            self.linear_gate = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.gate_projection_size_per_head,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_gate",
            )
        else:
            self.linear_gate = None
        self.prefix = prefix

        # Per-head learnable attention sink. Created BEFORE ``MLAAttention`` so
        # it can be forwarded as the ``sinks`` impl kwarg. The parameter always
        # holds the local TP shard.
        self.learnable_sink = bool(getattr(config, "learnable_sink", False))
        sinks = None
        sink_backend: type[AttentionBackend] | None = None
        if self.learnable_sink:
            sink_backend = self._resolve_sink_backend(kv_cache_dtype)
            self.learnable_sink_param = nn.Parameter(
                torch.empty(
                    self.num_local_heads,
                    dtype=torch.float32,
                )
            )
            sinks = self.learnable_sink_param
            self._force_sparse_mqa()

        extra_impl_args = {} if sinks is None else {"sinks": sinks}
        self.mla_attn = MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            topk_indices_buffer=topk_indices_buffer,
            attn_backend=sink_backend,
            **extra_impl_args,
        )

    def _resolve_sink_backend(self, kv_cache_dtype: str) -> type[AttentionBackend]:
        """Return an MLA backend that can apply this layer's learnable sink.

        The sink is part of the architecture, so a backend that cannot apply it
        changes the model's output. Kernel selection belongs to the platform
        dispatch layer: this model accepts the selected backend only when it
        advertises sink support.  Vendor policies may select a vendor backend,
        a FlagGems backend, or another compatible implementation.

        Args:
            kv_cache_dtype: The layer's KV cache dtype string.

        Returns:
            The sink-capable backend class. Missing support raises an error.
        """
        head_size = self.kv_lora_rank + self.qk_rope_head_dim
        dtype = torch.get_default_dtype()
        try:
            selected_cls = get_attn_backend(
                head_size=head_size,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                use_mla=True,
                has_sink=True,
                use_sparse=self.is_sparse,
                num_heads=self.num_local_heads,
            )
        except Exception as exc:
            raise RuntimeError("HYV4 requires a sink-capable MLA backend") from exc
        if not selected_cls.supports_sink():
            raise NotImplementedError(
                f"HYV4 learnable sink is unsupported by {selected_cls.get_name()}"
            )
        return selected_cls

    def _force_sparse_mqa(self) -> None:
        """Keep every token on the sink-capable sparse MQA path.

        ``_resolve_sink_backend`` only binds the backend that serves decode.
        ``MLAAttention`` additionally routes short prefills to a separate dense
        MLA prefill backend, and none of those accept ``attn_sink``, so prefill
        would silently drop the sink while decode applies it. Such a partially
        applied sink is not the trained architecture and corrupts the output, so
        opt out of the dense split instead.

        Prefills up to ``index_topk`` keep every token inside the sparse top-k,
        making the sparse path numerically equivalent to the dense one apart from
        also applying the sink.
        """
        attention_config = get_current_vllm_config().attention_config
        # Stock vLLM 0.24 already routes every SparseMLAAttentionImpl token
        # through forward_mqa.  The explicit switch exists only on the newer
        # HY4 feature branch this adapter was supplied against.
        if not hasattr(attention_config, "sparse_mla_force_mqa"):
            logger.info_once(
                "HYV4 learnable sink enabled: vLLM 0.24 sparse MLA already "
                "uses MQA for prefill and decode."
            )
            return
        if attention_config.sparse_mla_force_mqa:
            return
        attention_config.sparse_mla_force_mqa = True
        logger.info_once(
            "HYV4 learnable sink enabled: forcing sparse MQA for prefill too, "
            "as the dense MLA prefill backends cannot apply sinks."
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        if self.q_lora_rank is not None:
            assert self.q_a_proj is not None
            assert self.q_a_layernorm is not None
            assert self.q_b_proj is not None
            q_c = self.q_a_proj(hidden_states)[0]
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.q_proj is not None
            q = self.q_proj(hidden_states)[0]

        assert self.kv_a_proj_with_mqa is not None
        kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        # Add a head dim of 1 to k_pe.
        k_pe = k_pe.unsqueeze(1)
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe
        )

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        output_shape = (
            hidden_states.shape[0],
            self.num_local_heads * self.v_head_dim,
        )
        # Pure decode uses static device metadata and can be captured as a
        # full graph. Prefill remains eager because the indexer uses host metadata.
        attn_out = torch.empty(
            output_shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        self._indexer_and_attn(
            hidden_states, q_c, positions, q, kv_c_normed, k_pe, attn_out
        )

        if self.gated_mla and self.linear_gate is not None:
            gate_score = self.linear_gate(hidden_states)[0]
            if self.config.gating_type == "headwise":
                gate_score = gate_score.unsqueeze(-1)
                attn_out = attn_out.reshape(*attn_out.shape[:-1], -1, self.v_head_dim)
                attn_out = attn_out * torch.sigmoid(gate_score)
                attn_out = attn_out.reshape(*attn_out.shape[:-2], -1)
            else:
                attn_out = attn_out * torch.sigmoid(gate_score)

        out, _ = self.o_proj(attn_out)
        return out

    def _indexer_and_attn(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor | None,
        positions: torch.Tensor,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, heads * v_head_dim], written in place
    ) -> None:
        """Run the indexer and sparse MLA; graph policy belongs to vLLM."""
        if self.indexer is not None and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
        out.copy_(
            self.mla_attn(
                q,
                kv_c_normed,
                k_pe,
                output_shape=out.shape,
            )
        )


class HYV4HyperConnection(nn.Module):
    """Four identity residual streams with token-dependent read/write gates."""

    def __init__(self, config: HYV4Config) -> None:
        super().__init__()
        self.num_streams = config.hc_mult
        width = config.hc_mult * config.hidden_size
        self.hc_fn = nn.Parameter(
            torch.empty(2 * config.hc_mult, width, dtype=torch.float32)
        )
        self.hc_base = nn.Parameter(
            torch.empty(2 * config.hc_mult, dtype=torch.float32)
        )
        self.hc_scale = nn.Parameter(torch.empty(2, dtype=torch.float32))
        self.magnitude = float(config.hc_magnitude)
        self.hc_eps = float(config.hc_eps)
        self.normalize_eps = float(config.rms_norm_eps)

    def read(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse streams and return the write gates for the branch output."""
        flat = streams.flatten(1).float()
        norm = torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + self.normalize_eps
        )
        gates = F.linear(flat, self.hc_fn) * norm
        read, write = gates.split(self.num_streams, dim=-1)
        read = torch.sigmoid(read * self.hc_scale[0] + self.hc_base[: self.num_streams])
        write = self.magnitude * torch.sigmoid(
            write * self.hc_scale[1] + self.hc_base[self.num_streams :]
        )
        read = read + self.hc_eps
        write = write + self.hc_eps
        collapsed = (read.unsqueeze(-1) * streams.float()).sum(dim=1)
        return collapsed.to(streams.dtype), write

    @staticmethod
    def write(
        streams: torch.Tensor,
        delta: torch.Tensor,
        write: torch.Tensor,
    ) -> torch.Tensor:
        """Write one branch result into all identity residual streams."""
        dtype = delta.dtype
        return (
            streams.float() + write.float().unsqueeze(-1) * delta.float().unsqueeze(1)
        ).to(dtype)


class HYV4HyperLayer(nn.Module):
    """Checkpoint-compatible wrapper around a pre-branch hyper-connection."""

    def __init__(self, config: HYV4Config) -> None:
        super().__init__()
        self.hc_pre = HYV4HyperConnection(config)


class HYV4HyperHead(nn.Module):
    """Collapse the four final residual streams before the final RMSNorm."""

    def __init__(self, config: HYV4Config) -> None:
        super().__init__()
        self.num_streams = config.hc_mult
        width = config.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(config.hc_mult, width, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(config.hc_mult, dtype=torch.float32)
        )
        self.hc_head_scale = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.hc_eps = float(config.hc_eps)
        self.normalize_eps = float(config.rms_norm_eps)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        flat = streams.flatten(1).float()
        norm = torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + self.normalize_eps
        )
        read = (
            torch.sigmoid(
                F.linear(flat, self.hc_head_fn) * norm * self.hc_head_scale
                + self.hc_head_base
            )
            + self.hc_eps
        )
        hidden = (read.unsqueeze(-1) * streams.float()).sum(dim=1)
        return hidden.to(streams.dtype)


class HYV4DenseMLP(nn.Module):
    """Tensor-parallel unclamped SwiGLU used by dense and shared branches."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class HYV4MoE(nn.Module):
    """HY4 no-aux sigmoid routed experts plus one shared expert."""

    def __init__(
        self,
        config: HYV4Config,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config
        if parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "HY4 does not yet support sequence-parallel MoE; disable it."
            )
        if config.hidden_act != "silu":
            raise ValueError("HY4 currently supports only the silu activation")

        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )
        # The SM90 DSV3 router kernel selected by vLLM for H=6144/E=256
        # requires BF16 weights. HY4 stores and evaluates the router in FP32,
        # so keep this model on GateLinear's FP32 F.linear fallback instead.
        self.gate.allow_dsv3_router_gemm = False
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )
        self.shared_experts = HYV4DenseMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.shared_experts",
        )

        eplb_config = parallel_config.eplb_config
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = config.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts

        # Keep the shared branch outside FusedMoE so HY4's FP32 routed/shared
        # residual addition is explicit and cannot be rounded early to BF16.
        # Resolve the factory from its owning module at construction time.
        # WorkerFL installs the FusedMoEFL factory before model construction;
        # binding FusedMoE during module import would make that hook dependent
        # on model-inspection/import order.
        self.experts = fused_moe.FusedMoE(
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            routed_scaling_factor=config.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            apply_routed_scale_to_output=False,
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            router_logits_dtype=torch.float32,
        )
        self.n_local_physical_experts = self.experts.routed_experts.local_num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, shape[-1])
        if self.experts.is_internal_router:
            routed = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            routed = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        shared = self.shared_experts(hidden_states)
        combined = (routed.float() + shared.float()).to(hidden_states.dtype)
        return combined.view(shape)


class HYV4Attention(HYV4MLAAttention):
    """HY4 compressed MLA plus DSA indexer, gate, and learnable sink."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: HYV4Config,
        layer_idx: int,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
            layer_idx=layer_idx,
        )


class HYV4DecoderLayer(nn.Module):
    """One HY4 attention/MLP block operating on four residual streams."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None,
    ) -> None:
        super().__init__()
        config = typing.cast(HYV4Config, vllm_config.model_config.hf_config)
        layer_idx = int(prefix.rsplit(".", 1)[-1])
        self.hc_attn_layer = HYV4HyperLayer(config)
        self.self_attn = HYV4Attention(
            vllm_config,
            config,
            layer_idx,
            f"{prefix}.self_attn",
            topk_indices_buffer,
        )
        self.hc_mlp_layer = HYV4HyperLayer(config)
        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = HYV4MoE(config, vllm_config, f"{prefix}.mlp")
        else:
            self.mlp = HYV4DenseMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        streams: torch.Tensor,
    ) -> torch.Tensor:
        hidden, write = self.hc_attn_layer.hc_pre.read(streams)
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn(positions, hidden)
        streams = self.hc_attn_layer.hc_pre.write(streams, hidden, write)

        hidden, write = self.hc_mlp_layer.hc_pre.read(streams)
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        streams = self.hc_mlp_layer.hc_pre.write(streams, hidden, write)
        return streams


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class HYV4Model(nn.Module):
    """HY4 decoder backbone."""

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = typing.cast(HYV4Config, vllm_config.model_config.hf_config)
        self.config = config
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=vllm_config.device_config.device,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: HYV4DecoderLayer(
                vllm_config,
                prefix,
                self.topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.hc_head = HYV4HyperHead(config)
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.hc_head = PPMissingLayer()
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["streams"],
            config.hc_mult * config.hidden_size,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                inputs_embeds = self.embed_input_ids(input_ids)
            streams = inputs_embeds.unsqueeze(1).expand(
                -1,
                self.config.hc_mult,
                -1,
            )
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline rank requires intermediate tensors")
            streams = intermediate_tensors["streams"].view(
                -1,
                self.config.hc_mult,
                self.config.hidden_size,
            )

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            streams = layer(positions, streams)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"streams": streams.flatten(1)})
        hidden_states = self.hc_head(streams)
        return self.norm(hidden_states)


class HYV4LogitsProcessor(LogitsProcessor):
    """Compute the excluded HY4 language head in FP32 when configured."""

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        hidden_states = hidden_states.to(lm_head.weight.dtype)
        logits = lm_head.quant_method.apply(
            lm_head,
            hidden_states,
            bias=embedding_bias,
        )
        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits


class HYV4MixtureOfExperts(MixtureOfExperts):
    """Expose HY4 expert metadata to vLLM EPLB infrastructure."""

    moe_mlp_layers: list[HYV4MoE]

    def extract_moe_parameters(self, example_moe: HYV4MoE | None) -> None:
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            return
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        if self.num_local_physical_experts != num_local_physical_experts:
            raise ValueError("HY4 local expert count changed unexpectedly")
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.routed_experts.update_expert_map()


class HYV4ForCausalLM(
    nn.Module,
    SupportsPP,
    HYV4MixtureOfExperts,
):
    """Native vLLM causal language model for the HY4 preview checkpoint."""

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = typing.cast(HYV4Config, vllm_config.model_config.hf_config)
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model = HYV4Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            lm_head_dtype = (
                torch.float32
                if config.enable_lm_head_fp32
                else vllm_config.model_config.dtype
            )
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                params_dtype=lm_head_dtype,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = HYV4LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.expert_weights = []
        self.num_expert_groups = 1
        self.moe_mlp_layers = []
        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, HYV4MoE):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts.routed_experts)
        self.num_moe_layers = len(self.moe_layers)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    @staticmethod
    def _packed_expert_target(name: str) -> tuple[str, bool] | None:
        gate_up_name = ".mlp.experts.gate_up_proj"
        down_name = ".mlp.experts.down_proj"
        if gate_up_name in name:
            return (
                name.replace(
                    gate_up_name,
                    ".mlp.experts.routed_experts.w13_weight",
                ),
                True,
            )
        if down_name in name:
            return (
                name.replace(
                    down_name,
                    ".mlp.experts.routed_experts.w2_weight",
                ),
                False,
            )
        return None

    def _load_all_experts(
        self,
        name: str,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_any = False
        expert_ids = getattr(self, "_hy4_local_expert_ids", range(num_experts))
        if loaded_weight.shape[0] != len(expert_ids):
            raise ValueError(
                f"HY4 packed expert leading dimension {loaded_weight.shape[0]} "
                f"does not match {len(expert_ids)} selected experts for {name}"
            )
        for loaded_expert_id, expert_id in enumerate(expert_ids):
            loaded_any |= loader(
                param,
                loaded_weight[loaded_expert_id],
                name,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=True,
            )
        return loaded_any

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load HY4 checkpoints with packed or per-expert MoE tensors."""
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        stacked_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("wk_weights_proj", "wk", 0),
            ("wk_weights_proj", "weights_proj", 1),
        ]
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=self.num_redundant_experts,
        )
        skip_topk_layers = compute_skip_topk_layers(self.config)
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        local_heads = self.config.num_attention_heads // tp_size

        for name, loaded_weight in weights:
            if name.startswith("model.mtp_layers."):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if is_skip_topk_indexer_weight(name, skip_topk_layers):
                continue
            if name.endswith("learnable_sink_param"):
                start = tp_rank * local_heads
                loaded_weight = loaded_weight.narrow(0, start, local_heads)

            packed_expert = self._packed_expert_target(name)
            if packed_expert is not None:
                mapped, is_gate_up = packed_expert
                if is_pp_missing_parameter(mapped, self):
                    continue
                if mapped not in params_dict:
                    raise ValueError(
                        f"HY4 packed expert tensor has no destination: {name} "
                        f"-> {mapped}"
                    )
                param = params_dict[mapped]
                if is_gate_up:
                    gate, up = loaded_weight.chunk(2, dim=-2)
                    loaded_gate = self._load_all_experts(
                        mapped,
                        param,
                        gate,
                        "w1",
                        self.config.n_routed_experts,
                    )
                    loaded_up = self._load_all_experts(
                        mapped,
                        param,
                        up,
                        "w3",
                        self.config.n_routed_experts,
                    )
                    loaded = loaded_gate or loaded_up
                else:
                    loaded = self._load_all_experts(
                        mapped,
                        param,
                        loaded_weight,
                        "w2",
                        self.config.n_routed_experts,
                    )
                if not loaded:
                    raise ValueError(f"No local HY4 expert accepted {name}")
                loaded_params.add(mapped)
                continue

            # compressed-tensors stores routed experts independently as
            # ``experts.<id>.<proj>.{weight,weight_scale}``. Route those names
            # into vLLM's fused local expert parameters while preserving the
            # expert id and w1/w2/w3 shard metadata expected by its loader.
            matched_expert = False
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in name:
                    continue
                matched_expert = True
                mapped = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(mapped, self):
                    break
                if mapped not in params_dict:
                    raise ValueError(
                        f"HY4 expert tensor has no destination: {name} -> {mapped}"
                    )
                param = params_dict[mapped]
                loader = typing.cast(Callable[..., bool], param.weight_loader)
                if loader(
                    param,
                    loaded_weight,
                    mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                ):
                    loaded_params.add(mapped)
                break
            if matched_expert:
                continue

            mapped_stacked = False
            for param_name, weight_name, shard_id in stacked_mapping:
                if weight_name not in name or "mlp.experts" in name:
                    continue
                if weight_name in ("wk", "weights_proj") and ".indexer." not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if is_pp_missing_parameter(mapped, self):
                    mapped_stacked = True
                    break
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                mapped_stacked = True
                break
            if mapped_stacked:
                continue

            if is_pp_missing_parameter(name, self):
                continue
            if name not in params_dict:
                raise ValueError(f"Unexpected HY4 checkpoint parameter: {name}")
            param = params_dict[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, loaded_weight)
            loaded_params.add(name)

        runtime_attention_scales = (
            ".self_attn.attn.q_scale",
            ".self_attn.attn.k_scale",
            ".self_attn.attn.v_scale",
            ".self_attn.attn.prob_scale",
            ".self_attn.mla_attn.q_scale",
            ".self_attn.mla_attn.k_scale",
            ".self_attn.mla_attn.v_scale",
            ".self_attn.mla_attn.prob_scale",
        )
        missing = sorted(
            name
            for name in set(params_dict).difference(loaded_params)
            if not name.endswith(runtime_attention_scales)
        )
        if missing:
            preview = ", ".join(missing[:20])
            suffix = " ..." if len(missing) > 20 else ""
            raise ValueError(
                f"HY4 checkpoint coverage failed: {len(missing)} model "
                f"parameters were not loaded: {preview}{suffix}"
            )
        return loaded_params
