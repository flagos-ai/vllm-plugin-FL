"""DeepSeek-V4 short-context BF16 attention adapter for Ascend 910C."""

from __future__ import annotations

from typing import cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadata,
    DeepseekSparseSWAMetadataBuilder,
)


def _validate_short_context_attention(vllm_config) -> None:
    """Only admit contexts where C4A can select every compressed candidate."""
    config = vllm_config.model_config.hf_config
    for layer_id, ratio in enumerate(
        config.compress_ratios[: config.num_hidden_layers]
    ):
        if ratio not in (0, 1, 4, 128):
            raise NotImplementedError(
                "DeepSeek-V4 Ascend BF16 attention does not support "
                f"layer {layer_id} compress_ratio={ratio}."
            )
    max_model_len = vllm_config.model_config.max_model_len
    supported_context = min(config.sliding_window, 128)
    if max_model_len > supported_context:
        raise ValueError(
            "DeepSeek-V4 on Ascend 910C currently supports max_model_len <= "
            f"128 and <= sliding_window ({config.sliding_window}); set "
            f"--max-model-len {supported_context}."
        )
    if (
        4 in config.compress_ratios[: config.num_hidden_layers]
        and (max_model_len + 3) // 4 > config.index_topk
    ):
        raise NotImplementedError(
            "DeepSeek-V4 C4A on Ascend requires all compressed candidates "
            "to fit index_topk within the configured max_model_len."
        )


def _cache_rows(cache: torch.Tensor, slots: torch.Tensor, block_size: int):
    """Read logical cache rows without assuming contiguous page allocation."""
    if cache.ndim != 3 or cache.shape[1] != block_size:
        raise ValueError(
            f"Unexpected DeepSeek-V4 KV cache shape {tuple(cache.shape)} "
            f"for block size {block_size}"
        )
    return cache[slots // block_size, slots % block_size]


def _store_cache_rows(
    cache: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor, block_size: int
) -> None:
    """Write logical rows into potentially non-contiguous shared KV pages."""
    if cache.ndim != 3 or cache.shape[1] != block_size:
        raise ValueError(
            f"Unexpected DeepSeek-V4 KV cache shape {tuple(cache.shape)} "
            f"for block size {block_size}"
        )
    cache[slots // block_size, slots % block_size] = rows.to(cache.dtype)


class DeepseekV4FLMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    """Build metadata for the separate SWA cache; compressed MLA uses upstream."""

    def build_tile_scheduler(self, num_decode_tokens: int):
        del num_decode_tokens
        return {"swaonly": None, "c4a": None, "c128a": None}


class DeepseekV4FLSWABackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_name() -> str:
        return "FL_ASCEND_V4_BF16_SWA_CACHE"

    @staticmethod
    def get_builder_cls():
        return DeepseekV4FLMetadataBuilder


class DeepseekV4FLBackend(DeepseekV4SparseMLABackend):
    supported_kv_cache_dtypes = ["auto"]

    @staticmethod
    def get_name() -> str:
        return "FL_ASCEND_V4_BF16_SWA"

    @staticmethod
    def get_builder_cls():
        return DeepseekV4SparseMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        del capability
        return True


class DeepseekV4FLAttention(DeepseekV4Attention):
    """BF16 C4A/C128A/SWA attention for bounded 910C contexts."""

    backend_cls = DeepseekV4FLBackend
    swa_backend_cls = DeepseekV4FLSWABackend
    use_fp8_ds_mla_layout = False

    def __init__(self, *args, **kwargs) -> None:
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None and args:
            vllm_config = args[0]
        assert vllm_config is not None
        _validate_short_context_attention(vllm_config)

        original_event = torch.cuda.Event
        torch.cuda.Event = torch.npu.Event  # type: ignore[attr-defined,misc]
        try:
            super().__init__(*args, **kwargs)
        finally:
            torch.cuda.Event = original_event  # type: ignore[misc]

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del llama_4_scaling
        num_tokens = hidden_states.shape[0]
        o_padded = torch.empty(
            (num_tokens, self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self._run_parallel_input_projections(hidden_states)
        )
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr_float = qr.float()
        kv_float = kv.float()
        qr = (
            qr_float
            * torch.rsqrt(qr_float.pow(2).mean(-1, keepdim=True) + self.eps)
            * self.q_norm.weight.float()
        ).to(qr.dtype)
        kv = (
            kv_float
            * torch.rsqrt(kv_float.pow(2).mean(-1, keepdim=True) + self.eps)
            * self.kv_norm.weight.float()
        ).to(kv.dtype)
        self._prepare_and_attn_fn(
            hidden_states,
            qr,
            kv,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
        )
        return self._o_proj(o_padded[:, : self.n_local_heads, :], positions)

    def _run_parallel_input_projections(self, hidden_states):
        # torch.mm(..., out_dtype=torch.float32) in the upstream implementation
        # is unavailable in torch-npu. Keep the compressor projection in FP32.
        qr_kv = self._fused_wqa_wkv_gemm(hidden_states)
        kv_score = None
        if self.compressor is not None:
            weight = self.compressor.fused_wkv_wgate.weight
            kv_score = torch.mm(hidden_states.float(), weight.float().T)
        # C4A selects all compressed candidates in this bounded context, so
        # indexer ranking and its auxiliary projection do not affect attention.
        return qr_kv, kv_score, None, None

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        if not isinstance(attn_metadata, dict):
            return q
        metadata = cast(
            DeepseekSparseSWAMetadata,
            attn_metadata[self.swa_cache_layer.prefix],
        )
        q = q * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + self.eps).to(
            q.dtype
        )
        q, rotated_kv = self.rotary_emb.forward_native(positions, q, kv.unsqueeze(1))
        assert rotated_kv is not None
        kv = rotated_kv.squeeze(1)
        slots = metadata.slot_mapping
        valid = slots >= 0
        if valid.any():
            _store_cache_rows(
                self.swa_cache_layer.kv_cache,
                slots[valid].long(),
                kv[valid],
                metadata.block_size,
            )
        return q

    def _compress_kv_bf16(self, kv_score, positions) -> None:
        """Build the upstream C4A/C128A compressed key in the BF16 cache."""
        metadata_dict = get_forward_context().attn_metadata
        if not isinstance(metadata_dict, dict):
            return
        compressor = self.compressor
        assert compressor is not None
        ratio = self.compress_ratio
        overlap = ratio == 4
        coff = 1 + overlap
        state_width = coff * self.head_dim
        state_metadata = metadata_dict[compressor.state_cache.prefix]
        compressed_metadata = metadata_dict[self.prefix]

        kv, score = kv_score.split([state_width, state_width], dim=-1)
        ape = compressor.ape[positions.remainder(ratio)]
        packed_state = torch.cat((kv, score + ape), dim=-1)
        state_cache = compressor.state_cache.kv_cache
        state_slots = state_metadata.slot_mapping[: positions.shape[0]].long()
        valid_state = state_slots >= 0
        if valid_state.any():
            _store_cache_rows(
                state_cache,
                state_slots[valid_state],
                packed_state[valid_state],
                state_metadata.block_size,
            )

        num_tokens = positions.shape[0]
        compressed_slots = compressed_metadata.slot_mapping[:num_tokens].long()
        active = ((positions + 1).remainder(ratio) == 0) & (compressed_slots >= 0)
        if not active.any():
            return
        positions = positions[active]
        compressed_slots = compressed_slots[active]

        window = coff * ratio
        offsets = torch.arange(window, device=positions.device)
        history_pos = positions[:, None] - window + 1 + offsets[None, :]
        history_valid = history_pos >= 0
        safe_pos = history_pos.clamp_min(0)
        assert state_metadata.token_to_req_indices is not None
        request_ids = state_metadata.token_to_req_indices[:num_tokens][active].long()
        state_block_size = state_metadata.block_size
        state_blocks = state_metadata.block_table[
            request_ids[:, None], safe_pos // state_block_size
        ].long()
        history_valid &= state_blocks >= 0
        history_slots = (
            state_blocks.clamp_min(0) * state_block_size + safe_pos % state_block_size
        ).long()
        history = _cache_rows(state_cache, history_slots, state_block_size)

        channel_offsets = torch.arange(self.head_dim, device=positions.device)
        if overlap:
            channel_offsets = (
                channel_offsets[None, :]
                + (offsets[:, None] >= ratio).long() * self.head_dim
            )
        else:
            channel_offsets = channel_offsets[None, :].expand(window, -1)
        channel_offsets = channel_offsets[None, :, :].expand(positions.shape[0], -1, -1)
        state_kv = torch.gather(history[:, :, :state_width], 2, channel_offsets)
        state_score = torch.gather(history[:, :, state_width:], 2, channel_offsets)
        state_score = state_score.masked_fill(~history_valid[:, :, None], -float("inf"))
        compressed = (torch.softmax(state_score, dim=1) * state_kv).sum(dim=1)
        compressed = (
            compressed
            * torch.rsqrt(compressed.square().mean(dim=-1, keepdim=True) + self.eps)
            * compressor.norm.weight.float()
        )

        rope_dim = self.rope_head_dim
        compressed_pos = (positions // ratio) * ratio
        cos_sin = self.rotary_emb.cos_sin_cache[compressed_pos]
        rope = compressed[:, -rope_dim:]
        even, odd = rope[:, ::2], rope[:, 1::2]
        cos, sin = cos_sin[:, : rope_dim // 2], cos_sin[:, rope_dim // 2 :]
        rotated = torch.stack(
            (even * cos - odd * sin, odd * cos + even * sin), dim=-1
        ).flatten(-2)
        compressed = torch.cat((compressed[:, :-rope_dim], rotated), dim=-1)

        _store_cache_rows(
            self.kv_cache,
            compressed_slots,
            compressed,
            compressed_metadata.block_size // ratio,
        )

    def _prepare_and_attn(
        self,
        hidden_states,
        qr,
        kv,
        kv_score,
        indexer_kv_score,
        indexer_weights,
        positions,
        o_padded,
    ) -> None:
        del hidden_states, indexer_kv_score, indexer_weights
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
        q = self._fused_qnorm_rope_kv_insert(
            q, kv, positions, get_forward_context().attn_metadata
        )
        if self.compressor is not None:
            assert kv_score is not None
            self._compress_kv_bf16(kv_score, positions)
        if self.compress_ratio == 4:
            assert self.indexer is not None
            assert self.topk_indices_buffer is not None
            candidates = torch.arange(
                self.indexer.topk_tokens, device=positions.device, dtype=torch.int32
            )
            valid = candidates[None, :] < ((positions[:, None] + 1) // 4)
            self.topk_indices_buffer[: positions.shape[0]].copy_(
                torch.where(valid, candidates[None, :], -1)
            )
        self.forward_mqa(q, kv, positions, o_padded)

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        o, _ = self.rotary_emb.forward_native(positions, o, None, inverse=True)
        if self.n_local_groups != 1:
            raise ValueError(
                "DeepSeek-V4 FL output projection requires one group per TP rank"
            )
        z = self.wo_a(o.flatten(1))
        return self.wo_b(z)

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        del kv
        metadata_dict = get_forward_context().attn_metadata
        if not isinstance(metadata_dict, dict):
            output.zero_()
            return
        swa_metadata = cast(
            DeepseekSparseSWAMetadata,
            metadata_dict[self.swa_cache_layer.prefix],
        )
        assert swa_metadata.token_to_req_indices is not None
        request_ids = swa_metadata.token_to_req_indices[: positions.shape[0]].long()

        swa_block_size = swa_metadata.block_size
        swa_offsets = torch.arange(
            self.window_size - 1, -1, -1, device=positions.device
        )
        swa_pos = positions[:, None] - swa_offsets[None, :]
        swa_valid = swa_pos >= 0
        safe_pos = swa_pos.clamp_min(0)
        swa_blocks = swa_metadata.block_table[
            request_ids[:, None], safe_pos // swa_block_size
        ].long()
        swa_valid &= swa_blocks >= 0
        swa_slots = (
            swa_blocks.clamp_min(0) * swa_block_size + safe_pos % swa_block_size
        ).long()
        history = _cache_rows(self.swa_cache_layer.kv_cache, swa_slots, swa_block_size)

        if self.compress_ratio > 1:
            compressed_metadata = cast(
                DeepseekV4FlashMLAMetadata, metadata_dict[self.prefix]
            )
            compressed_block_size = (
                compressed_metadata.block_size // self.compress_ratio
            )
            candidate_count = (
                self.max_model_len + self.compress_ratio - 1
            ) // self.compress_ratio
            candidates = torch.arange(
                candidate_count, device=positions.device, dtype=torch.long
            )[None, :].expand(positions.shape[0], -1)
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                candidates = self.topk_indices_buffer[
                    : positions.shape[0], :candidate_count
                ].long()
            compressed_valid = (candidates >= 0) & (
                candidates < ((positions[:, None] + 1) // self.compress_ratio)
            )
            safe_candidates = candidates.clamp_min(0)
            compressed_blocks = compressed_metadata.block_table[
                request_ids[:, None], safe_candidates // compressed_block_size
            ].long()
            compressed_valid &= compressed_blocks >= 0
            compressed_slots = (
                compressed_blocks.clamp_min(0) * compressed_block_size
                + safe_candidates % compressed_block_size
            ).long()
            compressed_history = _cache_rows(
                self.kv_cache, compressed_slots, compressed_block_size
            )
            history = torch.cat((compressed_history, history), dim=1)
            valid = torch.cat((compressed_valid, swa_valid), dim=1)
        else:
            valid = swa_valid
        scores = torch.matmul(
            q.float().unsqueeze(-2), history.float().unsqueeze(1).transpose(-1, -2)
        ).squeeze(-2)
        scores.mul_(self.scale)
        scores.masked_fill_(~valid[:, None, :], -float("inf"))
        sink = self.attn_sink[: self.n_local_heads].float()[None, :, None]
        max_score = torch.maximum(scores.amax(-1, keepdim=True), sink)
        weights = torch.exp(scores - max_score) * valid[:, None, :]
        denominator = weights.sum(-1, keepdim=True) + torch.exp(sink - max_score)
        result = torch.matmul(
            weights.unsqueeze(-2), history.float().unsqueeze(1)
        ).squeeze(-2)
        output.copy_((result / denominator).to(output.dtype))


def _install_hc_head_fallback() -> None:
    from vllm.model_executor.layers.mhc import HCHeadOp

    if getattr(HCHeadOp, "_vllm_fl_ascend", False):
        return

    def forward_native(
        self,
        hidden_states,
        hc_fn,
        hc_scale,
        hc_base,
        rms_norm_eps,
        hc_eps,
    ):
        from flag_gems.fused.mhc import hc_head_fused_kernel

        hc_mult, hidden_size = hidden_states.shape[-2:]
        flat = hidden_states.reshape(-1, hc_mult, hidden_size)
        out = torch.empty(
            flat.shape[0],
            hidden_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        hc_head_fused_kernel(
            flat,
            hc_fn,
            hc_scale,
            hc_base,
            out,
            hidden_size,
            rms_norm_eps,
            hc_eps,
            hc_mult,
        )
        return out.view(*hidden_states.shape[:-2], hidden_size)

    HCHeadOp.forward_native = forward_native
    HCHeadOp._vllm_fl_ascend = True


if current_platform.device_type == "npu":
    _install_hc_head_fallback()

from vllm.models.deepseek_v4.xpu import model as _xpu_model  # noqa: E402

if current_platform.device_type == "npu":
    _xpu_model.DeepseekV4XPUAttention = DeepseekV4FLAttention


class DeepseekV4ForCausalLM(_xpu_model.DeepseekV4ForCausalLM):
    """vLLM 0.28 DeepSeek-V4 architecture with Ascend BF16 attention."""

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        _validate_short_context_attention(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)


__all__ = ["DeepseekV4ForCausalLM", "DeepseekV4FLAttention"]
