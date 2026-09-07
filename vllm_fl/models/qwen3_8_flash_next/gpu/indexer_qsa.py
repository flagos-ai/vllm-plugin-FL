# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-vendor Qwen3.8-Flash-Next weight-free QSA indexer."""

from __future__ import annotations

import os
from typing import Any, cast

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig

from ..common.qsa_cache import (
    QSACompressedKeyCache,
    QSAForwardMetadata,
    QSAKeyStateCache,
    canonical_qsa_rope_positions,
)
from .nvidia_fast_paths import fast_gemma_rmsnorm, fast_qsa_rope

_QSA_FUSED_COMPRESS_ENABLED = os.environ.get(
    "QWEN4_QSA_FUSED_COMPRESS", "1"
) != "0"


def apply_qsa_rope(
    rotary_emb: nn.Module,
    positions: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Apply the main attention's exact 1D/MRoPE composition to QSA heads."""

    rotary_dim = rotary_emb.rotary_dim
    cache = rotary_emb._match_cos_sin_cache_dtype(tensor)  # noqa: SLF001
    cos_sin = cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    fast_result = fast_qsa_rope(rotary_emb, positions, tensor, cos, sin)
    if fast_result is not None:
        return fast_result
    if positions.ndim == 2:
        sections = rotary_emb.mrope_section
        if rotary_emb.mrope_interleaved:
            axis_cos, axis_sin = cos, sin
            channels = torch.arange(cos.shape[-1], device=cos.device)
            is_height = (channels % 3 == 1) & (channels < sections[1] * 3)
            is_width = (channels % 3 == 2) & (channels < sections[2] * 3)
            cos = torch.where(is_height, axis_cos[1], axis_cos[0])
            cos = torch.where(is_width, axis_cos[2], cos)
            sin = torch.where(is_height, axis_sin[1], axis_sin[0])
            sin = torch.where(is_width, axis_sin[2], sin)
        else:
            cos = torch.cat(
                [axis[index] for index, axis in enumerate(cos.split(sections, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [axis[index] for index, axis in enumerate(sin.split(sections, dim=-1))],
                dim=-1,
            )

    # Cross-vendor fallback: public/native tensor composition lets FlagGems or
    # the vendor PyTorch runtime dispatch every primitive appropriately.
    rotated = rotary_emb.apply_rotary_emb.forward_native(
        tensor[..., :rotary_dim],
        cos,
        sin,
    )
    return torch.cat((rotated, tensor[..., rotary_dim:]), dim=-1)


class QSAIndexer(nn.Module):
    """Replicated Q/K projection plus paged, weight-free QSA selection.

    ``prefix`` must be the checkpoint's indexer prefix, normally
    ``model.layers.N.self_attn.indexer``.  Consequently the trainable names are
    ``index_qk_proj``, ``q_layernorm`` and ``k_layernorm`` under that prefix.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Any,
        layer_id: int,
        rotary_emb: nn.Module,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if vllm_config.cache_config is None:
            raise ValueError("QSA requires a paged KV cache")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8-Flash-Next QSA currently requires BF16")

        self.layer_id = int(layer_id)
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.rotary_emb = rotary_emb
        self.prefix = prefix

        self.index_qk_proj = ReplicatedLinear(
            int(config.hidden_size),
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.index_qk_proj" if prefix else "index_qk_proj",
        )
        self.q_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )
        self.k_layernorm = GemmaRMSNorm(
            self.index_head_dim,
            eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )

        cache_config = vllm_config.cache_config
        cache_prefix = f"{prefix}." if prefix else ""
        self.raw_key_cache = QSAKeyStateCache(
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            cache_rope_positions=vllm_config.model_config.uses_mrope,
            prefix=f"{cache_prefix}raw_key_cache",
            cache_config=cache_config,
            vllm_config=vllm_config,
        )
        self.compressed_key_cache = QSACompressedKeyCache(
            head_size=self.index_head_dim,
            dtype=torch.bfloat16,
            compress_ratio=self.compress_ratio,
            prefix=f"{cache_prefix}compressed_key_cache",
            cache_config=cache_config,
            vllm_config=vllm_config,
        )

    @property
    def output_width(self) -> int:
        return self.token_topk + self.compress_ratio - 1

    def project_qk(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project replicated Q/K, normalize+rotate Q, and preserve raw K."""

        qk, _ = self.index_qk_proj(hidden_states)
        q_raw, token_k = qk.split(
            (
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ),
            dim=-1,
        )
        q = q_raw.reshape(-1, self.index_n_heads, self.index_head_dim)
        flat_q = q.reshape(-1, self.index_head_dim)
        normalized_q = fast_gemma_rmsnorm(self.q_layernorm, flat_q)
        if normalized_q is None:
            normalized_q = self.q_layernorm(flat_q)
        q = normalized_q.reshape_as(q)
        q = apply_qsa_rope(self.rotary_emb, positions, q)
        return q, token_k.reshape(-1, 1, self.index_head_dim)

    def normalize_compressed_keys(
        self,
        compressed_keys: torch.Tensor,
        first_rope_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize pooled K and apply the first token's exact group position."""

        keys = compressed_keys.reshape(-1, self.index_head_dim)
        normalized_keys = fast_gemma_rmsnorm(self.k_layernorm, keys)
        if normalized_keys is None:
            normalized_keys = self.k_layernorm(keys)
        keys = normalized_keys.reshape(-1, 1, self.index_head_dim)
        if getattr(self.rotary_emb, "mrope_section", None):
            positions = first_rope_positions.transpose(0, 1)
        else:
            positions = first_rope_positions[:, 0]
        return apply_qsa_rope(self.rotary_emb, positions, keys)

    def _metadata(
        self,
    ) -> tuple[QSAForwardMetadata, QSAForwardMetadata] | None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            return None
        raw = cast(QSAForwardMetadata, metadata[self.raw_key_cache.prefix])
        compressed = cast(
            QSAForwardMetadata, metadata[self.compressed_key_cache.prefix]
        )
        if raw.num_actual_tokens != compressed.num_actual_tokens:
            raise RuntimeError("QSA side-cache metadata token counts disagree")
        # The compressed cache owns the one replay-sensitive metadata build.
        # QSAMetadataBuilder defers this call only when the global vLLM config
        # already selected full CUDA graphs; the first QSA layer records it and
        # later layers share the prepared metadata object.
        compressed.prepare()
        if compressed.logical_positions.device.type == "cpu":
            raw.prepare()
            if not torch.equal(raw.logical_positions, compressed.logical_positions):
                raise RuntimeError("QSA side-cache metadata positions disagree")
        return raw, compressed

    def _update_and_compress(
        self,
        token_k: torch.Tensor,
        positions: torch.Tensor,
        raw_metadata: QSAForwardMetadata,
        compressed_metadata: QSAForwardMetadata,
    ) -> None:
        num_tokens = raw_metadata.num_actual_tokens
        raw_key_cache = self.raw_key_cache.key_cache
        rope_position_cache = self.raw_key_cache.rope_position_cache
        from .ops.qsa import (
            qsa_compress_groups_with_ratio,
            qsa_compress_norm_mrope_store_groups,
            qsa_store_cache_rows,
        )

        qsa_store_cache_rows(
            raw_key_cache,
            raw_metadata.slot_mapping,
            token_k[:num_tokens],
        )
        if rope_position_cache is not None:
            position_rows = canonical_qsa_rope_positions(positions)[:num_tokens].to(
                device=rope_position_cache.device
            )
            qsa_store_cache_rows(
                rope_position_cache,
                raw_metadata.slot_mapping,
                position_rows,
            )
        rotary_dim = int(self.rotary_emb.rotary_dim)
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        if mrope_section is None:
            mrope_section = (rotary_dim // 2, 0, 0)
        else:
            mrope_section = tuple(int(section) for section in mrope_section)
        if (
            _QSA_FUSED_COMPRESS_ENABLED
            and self.index_head_dim == 128
            and rotary_dim == 64
            and bool(getattr(self.rotary_emb, "is_neox_style", False))
            and len(mrope_section) == 3
            and sum(mrope_section) == rotary_dim // 2
        ):
            cos_sin_cache = self.rotary_emb._match_cos_sin_cache_dtype(  # noqa: SLF001
                token_k
            )
            qsa_compress_norm_mrope_store_groups(
                raw_key_cache,
                raw_metadata.block_table,
                compressed_metadata.token_to_req,
                compressed_metadata.logical_positions,
                compressed_metadata.slot_mapping,
                self.compressed_key_cache.kv_cache,
                self.k_layernorm.weight,
                cos_sin_cache,
                self.compress_ratio,
                self.k_layernorm.variance_epsilon,
                rotary_dim,
                mrope_section,
                bool(getattr(self.rotary_emb, "mrope_interleaved", False)),
                rope_position_cache,
            )
            return
        pooled, first_positions = qsa_compress_groups_with_ratio(
            raw_key_cache,
            raw_metadata.block_table,
            compressed_metadata.token_to_req,
            compressed_metadata.logical_positions,
            compressed_metadata.slot_mapping,
            self.compress_ratio,
            rope_position_cache,
        )
        normalized = self.normalize_compressed_keys(pooled, first_positions)
        qsa_store_cache_rows(
            self.compressed_key_cache.kv_cache,
            compressed_metadata.slot_mapping,
            normalized,
        )

    def _select(
        self,
        q: torch.Tensor,
        metadata: QSAForwardMetadata,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        from .ops.qsa import qsa_select_paged_tokens

        return qsa_select_paged_tokens(
            q,
            self.compressed_key_cache.kv_cache,
            metadata.block_table,
            metadata.token_to_req,
            metadata.logical_positions,
            metadata.seq_lens,
            self.token_topk,
            self.compress_ratio,
            out,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return fixed-width request-relative token indices padded with ``-1``."""

        metadata = self._metadata()
        if metadata is None:
            result = torch.full(
                (hidden_states.shape[0], self.output_width),
                -1,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            if out is not None:
                out.copy_(result)
                return out
            return result
        raw_metadata, compressed_metadata = metadata
        num_tokens = raw_metadata.num_actual_tokens
        q, token_k = self.project_qk(
            hidden_states[:num_tokens], positions[..., :num_tokens]
        )
        self._update_and_compress(
            token_k,
            positions[..., :num_tokens],
            raw_metadata,
            compressed_metadata,
        )
        return self._select(q, compressed_metadata, out)


__all__ = ["QSAIndexer", "apply_qsa_rope"]
