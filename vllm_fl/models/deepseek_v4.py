"""DeepSeek-V4 short-context inference adapter for Ascend 910C."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4SparseMLABackend
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadata,
    DeepseekSparseSWAMetadataBuilder,
)

if TYPE_CHECKING:
    pass


class DeepseekV4FLMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
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
        return DeepseekV4FLMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        del capability
        return True


class DeepseekV4FLAttention(DeepseekV4Attention):
    """Portable BF16 SWA path used when FP8 E4M3 is unavailable on 910C."""

    backend_cls = DeepseekV4FLBackend
    swa_backend_cls = DeepseekV4FLSWABackend
    use_fp8_ds_mla_layout = False

    def __init__(self, *args, **kwargs) -> None:
        vllm_config = kwargs.get("vllm_config")
        if vllm_config is None and args:
            vllm_config = args[0]
        assert vllm_config is not None
        config = vllm_config.model_config.hf_config
        if vllm_config.model_config.max_model_len > config.sliding_window:
            raise ValueError(
                "DeepSeek-V4 on Ascend 910C currently supports max_model_len <= "
                f"sliding_window ({config.sliding_window}); set --max-model-len "
                f"{config.sliding_window}."
            )

        prefix = kwargs.get("prefix", args[1] if len(args) > 1 else "")
        layer_id = int(prefix.split(".")[-2])
        original_ratio = config.compress_ratios[layer_id]
        config.compress_ratios[layer_id] = 1
        original_event = torch.cuda.Event
        torch.cuda.Event = torch.npu.Event  # type: ignore[attr-defined,misc]
        try:
            super().__init__(*args, **kwargs)
        finally:
            torch.cuda.Event = original_event  # type: ignore[misc]
            config.compress_ratios[layer_id] = original_ratio

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
            cache = self.swa_cache_layer.kv_cache.view(-1, self.head_dim)
            cache.index_copy_(0, slots[valid], kv[valid].to(cache.dtype))
        return q

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
        del hidden_states, kv_score, indexer_kv_score, indexer_weights
        q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
        q = self._fused_qnorm_rope_kv_insert(
            q, kv, positions, get_forward_context().attn_metadata
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
        metadata = cast(
            DeepseekSparseSWAMetadata,
            metadata_dict[self.swa_cache_layer.prefix],
        )
        assert metadata.token_to_req_indices is not None
        block_size = metadata.block_size
        offsets = torch.arange(self.window_size - 1, -1, -1, device=positions.device)
        history_pos = positions[:, None] - offsets[None, :]
        valid = history_pos >= 0
        safe_pos = history_pos.clamp_min(0)
        request_ids = metadata.token_to_req_indices[: positions.shape[0]].long()
        block_ids = metadata.block_table[
            request_ids[:, None], safe_pos // block_size
        ].long()
        valid &= block_ids >= 0
        slots = (block_ids.clamp_min(0) * block_size + safe_pos % block_size).long()
        cache = self.swa_cache_layer.kv_cache.view(-1, self.head_dim)
        history = cache[slots]
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
            weights.to(history.dtype).unsqueeze(-2), history.unsqueeze(1)
        ).squeeze(-2)
        output.copy_((result / denominator.to(result.dtype)).to(output.dtype))


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


_install_hc_head_fallback()

from vllm.models.deepseek_v4.xpu import model as _xpu_model  # noqa: E402

_xpu_model.DeepseekV4XPUAttention = DeepseekV4FLAttention


class DeepseekV4ForCausalLM(_xpu_model.DeepseekV4ForCausalLM):
    """vLLM 0.28 DeepSeek-V4 architecture with FL Ascend kernels."""

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        filtered = (
            (name, value)
            for name, value in weights
            if ".compressor." not in name and ".indexer." not in name
        )
        return super().load_weights(filtered)


__all__ = ["DeepseekV4ForCausalLM", "DeepseekV4FLAttention"]
