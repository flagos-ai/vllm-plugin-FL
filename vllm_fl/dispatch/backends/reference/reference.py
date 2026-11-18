# Copyright (c) 2026 BAAI. All rights reserved.

"""
Reference backend implementation using PyTorch.

This backend provides reference operator implementations using native PyTorch
operations. These implementations are always available when PyTorch is installed
and serve as fallback implementations.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend


class ReferenceBackend(Backend):
    """
    Reference backend for operator implementations.

    This backend uses native PyTorch operations to provide reference
    implementations that are always available as fallbacks.
    """

    _available: Optional[bool] = None

    def topk_softmax(
        self,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        token_expert_indices: torch.Tensor,
        gating_output: torch.Tensor,
        renormalize: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PyTorch fallback for the fused MoE top-k router (no CUDA kernel)."""
        topk = topk_weights.size(-1)
        scores = torch.softmax(gating_output.float(), dim=-1)
        weights, indices = torch.topk(scores, topk, dim=-1)
        if renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        topk_weights.copy_(weights.to(topk_weights.dtype))
        topk_indices.copy_(indices.to(topk_indices.dtype))
        # Upstream's kernel writes ``k_idx * num_tokens + token_idx`` here
        # (see ``topkGating``: ``source_rows[idx] = k_idx * num_rows + block_row``),
        # not the row-major ``token_idx * topk + k_idx`` ordering.
        num_tokens = gating_output.size(0)
        token_idx = torch.arange(num_tokens, device=gating_output.device).view(-1, 1)
        k_idx = torch.arange(topk, device=gating_output.device).view(1, -1)
        token_expert_indices.copy_(
            (k_idx * num_tokens + token_idx).to(token_expert_indices.dtype)
        )
        return topk_weights, topk_indices

    def moe_align_block_size(
        self,
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        expert_map: torch.Tensor | None = None,
        pad_sorted_ids: bool = False,
        ignore_invalid_experts: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PyTorch fallback for fused MoE block alignment (capture-safe).

        ``topk_ids`` holds global expert ids. With ``ignore_invalid_experts`` the
        map is applied while bucketing, so ids mapped to ``-1`` -- or otherwise
        out of range -- belong to another rank and are dropped instead of being
        routed into a neighbouring expert's bucket. Without the flag, buckets are
        built from global ids and the returned ``expert_ids`` are remapped
        through ``expert_map`` afterwards, matching upstream.
        """
        device = topk_ids.device
        flat = topk_ids.reshape(-1).to(torch.int64)
        # Upstream's top-k kernel writes -1 for padded rows, so screen the raw
        # ids before they index anything: a negative or out-of-range id must not
        # reach ``expert_map`` (where -1 selects its last entry) nor the bucket
        # ranking below (where it would overwrite the last expert's slots).
        # With a map the raw ids are global, so the map length is the bound.
        raw_count = expert_map.numel() if expert_map is not None else num_experts
        raw_valid = (flat >= 0) & (flat < raw_count)
        # Mirror upstream: with ``ignore_invalid_experts`` the map is applied
        # while bucketing, so ids owned by another rank are dropped; without it
        # buckets are built from global ids and the returned expert ids are
        # remapped at the end instead.
        remap_output = expert_map is not None and not ignore_invalid_experts
        if expert_map is not None and ignore_invalid_experts:
            mapped = expert_map.to(device=device, dtype=torch.int64)[
                torch.where(raw_valid, flat, torch.zeros_like(flat))
            ]
            valid = raw_valid & (mapped >= 0) & (mapped < num_experts)
            ids = mapped
        else:
            valid = raw_valid
            ids = flat
        local = torch.where(valid, ids, torch.zeros_like(ids))
        num_valid = flat.numel()
        same_expert = (
            local.unsqueeze(0)
            == torch.arange(num_experts, device=device).unsqueeze(1)
        ) & valid.unsqueeze(0)
        counts = same_expert.sum(dim=1)
        padded = (counts + block_size - 1) // block_size * block_size
        offsets = torch.cumsum(padded, 0) - padded
        ends = offsets + padded
        total = padded.sum()
        if pad_sorted_ids:
            total = (total + block_size - 1) // block_size * block_size
        # Static upper bound keeps every buffer size independent of the routing
        # data, which is required while capturing CUDA graphs. The extra slot is
        # the sink for tokens dropped by ``expert_map``.
        max_total = num_valid + num_experts * (block_size - 1) + 1
        max_total = (max_total + block_size - 1) // block_size * block_size
        sorted_ids = torch.full(
            (max_total,), num_valid, dtype=torch.int32, device=device
        )
        slot = torch.arange(num_valid, device=device)
        ranks = (torch.cumsum(same_expert.to(torch.int32), dim=1) - 1)[local, slot]
        dest = offsets[local] + ranks.to(torch.int64)
        dest = torch.where(valid, dest, torch.full_like(dest, max_total - 1))
        sorted_ids[dest] = torch.where(
            valid, slot, torch.full_like(slot, num_valid)
        ).to(torch.int32)
        num_blocks = max_total // block_size
        block_start = torch.arange(num_blocks, device=device) * block_size
        owner = (block_start.unsqueeze(0) >= offsets.unsqueeze(1)) & (
            block_start.unsqueeze(0) < ends.unsqueeze(1)
        )
        expert_of_block = torch.where(
            owner.any(dim=0),
            owner.to(torch.int32).argmax(dim=0).to(torch.int32),
            torch.full((num_blocks,), -1, dtype=torch.int32, device=device),
        )
        expert_ids = torch.where(
            block_start < total,
            expert_of_block,
            torch.full((num_blocks,), -1, dtype=torch.int32, device=device),
        )
        num_tokens_post_padded = total.reshape(1).to(torch.int32)
        if remap_output:
            safe = expert_ids.clamp(min=0)
            mapped = expert_map.to(device=device, dtype=torch.int32)[safe]
            expert_ids = torch.where(expert_ids >= 0, mapped, expert_ids)
        return sorted_ids, expert_ids, num_tokens_post_padded

    def moe_sum(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        """PyTorch fallback for the fused MoE reduction over top-k experts."""
        torch.sum(inp, dim=1, out=out)

    @property
    def name(self) -> str:
        return "reference"

    def is_available(self) -> bool:
        """Check if PyTorch is available."""
        if ReferenceBackend._available is None:
            try:
                import torch

                ReferenceBackend._available = True
            except ImportError:
                ReferenceBackend._available = False
        return ReferenceBackend._available

    # ==================== Operator Implementations ====================

    def dynamic_per_token_quant_int8(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .impl.quantization import dynamic_per_token_quant_int8_torch

        return dynamic_per_token_quant_int8_torch(x)

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_torch

        return silu_and_mul_torch(obj, x)

    def gelu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        GELU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import gelu_and_mul_torch

        return gelu_and_mul_torch(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization.

        Args:
            obj: The calling obj (e.g., RMSNorm layer)
            x: Input tensor
            residual: Optional residual tensor

        Returns:
            Normalized tensor, or tuple of (normalized, residual) if residual is provided
        """
        from .impl.normalization import rms_norm_torch

        return rms_norm_torch(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding.

        Args:
            obj: The calling obj (for interface consistency)
            query: Query tensor
            key: Key tensor
            cos: Cosine cache
            sin: Sine cache
            position_ids: Position indices
            rotary_interleaved: Whether to use interleaved rotary
            inplace: Whether to modify tensors in-place (ignored in reference impl)

        Returns:
            Tuple of (embedded_query, embedded_key)
        """
        from .impl.rotary import rotary_embedding_torch

        return rotary_embedding_torch(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for reference (vLLM native).

        This method returns the vLLM native flash attention backend path,
        which serves as a fallback implementation.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

        Returns:
            Fully qualified class path string (vLLM native backend)
        """
        # Return vLLM's native flash attention backend as reference
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if use_mla:
            # vLLM native MLA backend
            if use_sparse:
                return AttentionBackendEnum.FLASHMLA_SPARSE.get_path()
            return AttentionBackendEnum.FLASHMLA.get_path()
        return AttentionBackendEnum.FLASH_ATTN.get_path()

    def topk_softmax(
        self, topk_weights, topk_indices, token_expert_indices,
        gating_output, renormalize=False, e_score_correction_bias=None,
    ):
        from .impl.fused_moe import topk_softmax

        return topk_softmax(
            topk_weights, topk_indices, token_expert_indices,
            gating_output, renormalize, e_score_correction_bias,
        )

    def invoke_fused_moe_triton_kernel(
        self,
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        block_shape=None,
        B_bias=None,
    ):
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            invoke_fused_moe_triton_kernel,
        )

        invoke_fused_moe_triton_kernel(
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape=block_shape,
            B_bias=B_bias,
        )
