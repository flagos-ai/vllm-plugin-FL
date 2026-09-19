# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM5Next vision tower + multimodal processor.

Self-contained fork of the GLM-OCR / GLM-4V vision transformer so GLM5Next owns its
vision path (no inheritance from `glm_ocr`/`glm4_1v`, leaving those upstream models
untouched). Folds in:
  - P1: fused q/k RMSNorm (one kernel, two distinct weights) in the attention.
  - P7: a `forward(..., encoder_metadata=)` fast path so the ModelRunnerV2 encoder
    CUDA graph (PR #49852 + `--compilation-config cudagraph_mm_encoder=true`) can
    replay the tower instead of re-building rotary/cu_seqlens on the CPU each call.
  - the OCR-variant delta vs GLM-4V base (no abs-pos embeddings / post-conv norm;
    q/k norm eps=1e-5; qkv/proj/MLP bias=True).

Default (flag off) is bit-identical to the inherited eager forward.
"""

from collections.abc import Mapping
from functools import partial
from typing import Any, ClassVar, Literal

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    parallel_state,
    utils as dist_utils,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer, Conv3dLayer
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.models.glm4_1v import (
    Glm4vDummyInputsBuilder,
    Glm4vForConditionalGeneration,
    Glm4vMultiModalProcessor,
    Glm4vProcessingInfo,
    Glm4vVisionTransformer,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.model_executor.models.vision import (
    get_vit_attn_backend,
    is_vit_use_data_parallel,
)
from vllm.models.deepseek_v4.common.ops import fused_q_kv_rmsnorm
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_fl.kernels.glm5_next.provider import use_nvidia_reference
from vllm_fl.kernels.glm5_next.vision_attention import Glm5VisionAttention
from vllm_fl.models.glm5_next import SiluAndMulWithClamp

logger = init_logger(__name__)


class Glm5NextVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 1,
        in_channels: int = 3,
        hidden_size: int = 1536,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = Conv3dLayer(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = self.proj(x).view(L, self.hidden_size)
        return x


class Glm5NextVisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        swiglu_limit: float,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=in_features,
            output_sizes=[hidden_features] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=use_data_parallel,
        )
        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=use_data_parallel,
        )
        # GLM5Next clamps the vision SwiGLU gate/up (matches sglang
        # `swiglu_clamped`); GLM-OCR / GLM-4V do not. This is the behavioral
        # delta that makes the vision tower genuinely GLM5Next-specific.
        self.act_fn = SiluAndMulWithClamp(swiglu_limit=swiglu_limit)

    def forward(self, x: torch.Tensor):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class Glm5NextVisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.tp_rank = (
            0 if use_data_parallel else parallel_state.get_tensor_model_parallel_rank()
        )
        self.hidden_size_per_attention_head = dist_utils.divide(
            projection_size, num_heads
        )
        self.num_attention_heads_per_partition = dist_utils.divide(
            num_heads, self.tp_size
        )

        self.head_dim = embed_dim // num_heads

        # q/k norm eps hard-coded 1e-5 — distinct from block/post norm eps.
        self.q_norm = RMSNorm(self.head_dim, eps=1e-5)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-5)

        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj" if quant_config else f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )
        self.proj = RowParallelLinear(
            input_size=projection_size,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            bias=True,
            disable_tp=use_data_parallel,
        )

        attention_cls = (
            MMEncoderAttention if use_nvidia_reference() else Glm5VisionAttention
        )
        self.attn = attention_cls(
            num_heads=self.num_attention_heads_per_partition,
            head_size=self.hidden_size_per_attention_head,
            scale=self.hidden_size_per_attention_head**-0.5,
            prefix=f"{prefix}.attn",
        )
        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        seq_len, bs, _ = qkv.shape
        q, k, v = qkv.chunk(3, dim=2)
        new_shape = (
            seq_len,
            bs,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x, _ = self.qkv(x)
        q, k, v = self.split_qkv(x)

        q_shape, k_shape = q.shape, k.shape
        q_flat = q.reshape(-1, self.head_dim)
        k_flat = k.reshape(-1, self.head_dim)
        if use_nvidia_reference():
            q, k = fused_q_kv_rmsnorm(
                q_flat,
                k_flat,
                self.q_norm.weight,
                self.k_norm.weight,
                self.q_norm.variance_epsilon,
            )
        else:
            q = self.q_norm(q_flat)
            k = self.k_norm(k_flat)
        q = q.view(q_shape)
        k = k.view(k_shape)

        q, k, v = (rearrange(t, "s b ... -> b s ...").contiguous() for t in (q, k, v))
        if rotary_pos_emb_cos is not None and rotary_pos_emb_sin is not None:
            qk_concat = torch.cat([q, k], dim=0)
            qk_rotated = self.apply_rotary_emb(
                qk_concat,
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
            )
            q, k = torch.chunk(qk_rotated, 2, dim=0)

        context_layer = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        context_layer = rearrange(context_layer, "b s h d -> s b (h d)").contiguous()

        output, _ = self.proj(context_layer)
        return output


class Glm5NextVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        swiglu_limit: float,
        norm_layer: partial[nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Glm5NextVisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = Glm5NextVisionMLP(
            dim,
            mlp_hidden_dim,
            swiglu_limit=swiglu_limit,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        x_attn = self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
        )
        x_fused_norm, residual = self.norm2(x, residual=x_attn)
        x = residual + self.mlp(x_fused_norm)
        return x


class Glm5NextPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        swiglu_limit: float,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.hidden_size = d_model
        self.proj = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=bias,
            gather_output=True,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )
        self.post_projection_norm = nn.LayerNorm(self.hidden_size)
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[context_dim] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
            disable_tp=use_data_parallel,
        )
        self.down_proj = RowParallelLinear(
            context_dim,
            self.hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
            disable_tp=use_data_parallel,
        )
        # Merger SwiGLU is clamped too (matches sglang Glm5NextVisionPatchMerger);
        # GLM-OCR / GLM-4V mergers are unclamped.
        self.act_fn = SiluAndMulWithClamp(swiglu_limit=swiglu_limit)
        self.extra_activation_func = nn.GELU()

    def forward(self, x: torch.Tensor):
        x, _ = self.proj(x)
        x = self.extra_activation_func(self.post_projection_norm(x))
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Glm5NextVisionTransformer(nn.Module):
    def __init__(
        self,
        text_config,  # noqa: ANN001  (kept for call-signature parity; unused — GLM5Next
        # uses vision_config.projection_intermediate_size for the merger, not
        # text_config.intermediate_size like GLM-OCR does.)
        vision_config,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.tp_size = (
            1 if use_data_parallel else get_tensor_model_parallel_world_size()
        )

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads

        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.out_hidden_size = vision_config.out_hidden_size

        # GLM5Next applies a SwiGLU gate/up clamp in the vision encoder (block
        # MLP + patch merger) that GLM-OCR / GLM-4V do not. Falls back to the
        # text config's limit if the vision config omits it (matches sglang).
        swiglu_limit = getattr(vision_config, "swiglu_limit", None)
        if swiglu_limit is None:
            swiglu_limit = getattr(text_config, "swiglu_limit", None)
        assert swiglu_limit is not None, (
            "GLM5Next vision requires swiglu_limit (vision_config or text_config)"
        )

        # Single construction pass — no abs-pos embeddings / post-conv norm (OCR delta).
        self.patch_embed = Glm5NextVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
        )

        norm_layer = partial(RMSNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"partial_rotary_factor": 0.5},
        )
        self.blocks = nn.ModuleList(
            [
                Glm5NextVisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    swiglu_limit=swiglu_limit,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(depth)
            ]
        )
        # GLM5Next-specific merger bottleneck width.
        self.merger = Glm5NextPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.projection_intermediate_size,
            swiglu_limit=swiglu_limit,
            quant_config=quant_config,
            bias=False,
            prefix=f"{prefix}.merger",
        )

        self.downsample = Conv2dLayer(
            in_channels=vision_config.hidden_size,
            out_channels=vision_config.out_hidden_size,
            kernel_size=vision_config.spatial_merge_size,
            stride=vision_config.spatial_merge_size,
        )
        self.post_layernorm = RMSNorm(
            vision_config.hidden_size, eps=vision_config.rms_norm_eps
        )

        self.attn_backend = (
            get_vit_attn_backend(head_size=head_dim, dtype=torch.get_default_dtype())
            if use_nvidia_reference()
            else AttentionBackendEnum.FLASH_ATTN
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(
        self, grid_thw: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)

        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)

        pos_ids = pos_ids.to(cos.device, non_blocking=True)
        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)
        return cos_combined, sin_combined, pos_ids

    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor | None:
        max_seqlen = None
        if self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
            AttentionBackendEnum.TRITON_ATTN,
        }:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        return max_seqlen

    def prepare_encoder_metadata(
        self,
        grid_thw_list: list[list[int]],
        *,
        max_batch_size: int | None = None,
        max_frames_per_batch: int | None = None,
        max_seqlen_override: int | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute encoder metadata (shared by eager forward + CG capture/replay).

        Forked from Glm4vVisionTransformer with the ``pos_embeds`` entry removed —
        GLM5Next's tower has no abs-pos embeddings (``self.embeddings`` is never
        built), so the upstream ``pos_embeds_interpolate`` call would AttributeError.
        """
        if device is None:
            device = self.device

        metadata: dict[str, torch.Tensor | None] = {}

        rotary_cos, rotary_sin, _ = self.rot_pos_emb(grid_thw_list)
        metadata["rotary_pos_emb_cos"] = rotary_cos
        metadata["rotary_pos_emb_sin"] = rotary_sin

        grid_thw_np = np.array(grid_thw_list, dtype=np.int32)
        patches_per_frame = grid_thw_np[:, 1] * grid_thw_np[:, 2]
        cu_seqlens = np.repeat(patches_per_frame, grid_thw_np[:, 0]).cumsum(
            dtype=np.int32
        )
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])

        pad_to = (
            max_frames_per_batch if max_frames_per_batch is not None else max_batch_size
        )
        if pad_to is not None:
            num_seqs = len(cu_seqlens) - 1
            if num_seqs < pad_to:
                cu_seqlens = np.concatenate(
                    [
                        cu_seqlens,
                        np.full(
                            pad_to - num_seqs,
                            cu_seqlens[-1],
                            dtype=np.int32,
                        ),
                    ]
                )

        metadata["sequence_lengths"] = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend, cu_seqlens, device
        )

        if max_seqlen_override is not None:
            max_seqlen_val = max_seqlen_override
        else:
            max_seqlen_val = MMEncoderAttention.compute_max_seqlen(
                self.attn_backend, cu_seqlens
            )
        metadata["max_seqlen"] = torch.tensor(max_seqlen_val, dtype=torch.int32)

        metadata["cu_seqlens"] = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend,
            cu_seqlens,
            self.hidden_size,
            self.tp_size,
            device,
        )

        return metadata

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
        *,
        encoder_metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        if encoder_metadata is not None:
            # Encoder CUDA-graph path (PR #49852): rotary/cu_seqlens/max_seqlen are
            # precomputed by prepare_encoder_metadata (which uses rot_pos_emb exactly
            # as the eager rebuild does), so reuse them and skip the per-call CPU
            # rebuild (the low-GPU-util culprit on multimodal workloads).
            rotary_pos_emb_cos = encoder_metadata["rotary_pos_emb_cos"]
            rotary_pos_emb_sin = encoder_metadata["rotary_pos_emb_sin"]
            cu_seqlens = encoder_metadata["cu_seqlens"]
            max_seqlen = encoder_metadata["max_seqlen"]
        else:
            if isinstance(grid_thw, list):
                grid_thw = torch.tensor(grid_thw, dtype=torch.int32)
            rotary_pos_emb_cos, rotary_pos_emb_sin, _ = self.rot_pos_emb(grid_thw)
            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])
            cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)
            max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)

        # transformers
        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen,
            )

        # adapter
        x = self.post_layernorm(x)
        x = x.view(-1, self.spatial_merge_size, self.spatial_merge_size, x.shape[-1])
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x).view(-1, self.out_hidden_size)
        x = self.merger(x)
        return x

    def load_weights(self, weights) -> set[str]:
        # vLLM 0.24's mapper predates stacked-weight mappings. Reuse the
        # GLM-4V loader, whose q/k/v and gate/up shard rules match this tower.
        return Glm4vVisionTransformer.load_weights(self, weights)


class Glm5NextProcessingInfo(Glm4vProcessingInfo):
    """Build the checkpoint's custom image/video processor locally."""

    def get_hf_processor(self, **kwargs: object):
        processor = getattr(self, "_glm5_hf_processor", None)
        if processor is None:
            from vllm_fl.transformers_utils.processors.glm5_next import (
                Glm5NextProcessor,
            )

            processor = Glm5NextProcessor.from_pretrained(self.ctx.model_config.model)
            processor.configure_serving(self.ctx.get_merged_mm_kwargs({}))
            self._glm5_hf_processor = processor
        processor.resolve_serving_kwargs(kwargs)
        return processor

    def _vision_budget(self, modality):
        from vllm_fl.transformers_utils.processors.glm5_next_budget import (
            modality_kwargs,
            resolve_vision_budget,
        )

        processor = self.get_hf_processor()
        return resolve_vision_budget(
            getattr(processor, modality + "_processor"),
            modality_kwargs(self.ctx.get_merged_mm_kwargs({}), modality),
        )

    def _get_image_max_pixels(self) -> int:
        return self._vision_budget("image").max_pixels

    def _get_video_max_pixels(self) -> int:
        return self._vision_budget("video").max_pixels

    def get_max_image_tokens(self) -> int:
        return self._vision_budget("image").max_tokens

    def get_image_size_with_most_features(self) -> ImageSize:
        height, width = self._vision_budget("image").largest_canvas()
        return ImageSize(width=width, height=height)

    def get_num_image_tokens(self, *, image_width, image_height):
        budget = self._vision_budget("image")
        return self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=1,
            min_image_pixels=budget.min_pixels,
            max_image_pixels=budget.max_pixels,
        )[1]

    def get_num_video_tokens(self, *, image_width, image_height, num_frames):
        return self._get_vision_info(
            image_width=image_width,
            image_height=image_height,
            num_frames=num_frames,
            max_image_pixels=self._get_video_max_pixels(),
            modality="video",
            min_image_pixels=self._vision_budget("video").min_pixels,
        )[1]

    def _video_frame_cap(self):
        options = self.get_hf_processor().resolve_serving_kwargs({})["videos_kwargs"]
        return options["max_frames"]

    def _get_max_video_frames(self, max_tokens):
        # Bound the inherited search even when the total video pixel cap is
        # below max_tokens: that case never exceeds the token cap as t grows.
        cap = self._video_frame_cap()
        width, height = self.get_image_size_with_most_features()
        frames = 0
        for candidate in range(1, cap + 1):
            if (
                self.get_num_video_tokens(
                    image_width=width, image_height=height, num_frames=candidate
                )
                > max_tokens
            ):
                break
            frames = candidate
        return frames

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 16,
        do_resize: bool = True,
        max_image_pixels: int = 28 * 28 * 2 * 30000,
        modality: str = "image",
        min_image_pixels: int = 1,
    ) -> tuple[ImageSize, int]:
        from vllm_fl.transformers_utils.processors.glm5_next import smart_resize

        vision_config = self.get_hf_config().vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size
        processor = getattr(self.get_hf_processor(), modality + "_processor")
        factor = patch_size * merge_size * processor.patch_expand_factor
        # The inherited GLM-4V dummy-video token estimator probes frame counts
        # just beyond the pixel budget while finding the largest fitting video.
        # Keep one aligned spatial patch available per probed frame so that the
        # estimator returns an over-budget token count instead of raising.
        padded_frames = num_frames + (-num_frames % temporal_patch_size)
        max_image_pixels = max(max_image_pixels, padded_frames * factor * factor)

        if do_resize:
            frames = max(padded_frames, temporal_patch_size)
            resized_height, resized_width = smart_resize(
                t=frames,
                h=image_height,
                w=image_width,
                t_factor=temporal_patch_size,
                h_factor=factor,
                w_factor=factor,
                min_pixels=min_image_pixels,
                max_pixels=max_image_pixels,
            )
            preprocessed_size = ImageSize(width=resized_width, height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width, height=image_height)

        grid_t = max(padded_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size
        num_patches = grid_t * grid_h * grid_w
        return preprocessed_size, num_patches // (merge_size**2)

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        """Video token ceiling from the token-budget pixel cap.

        vLLM 0.24's inherited implementation reads
        ``video_processor.size["longest_edge"]``, but the GLM-5-Next
        token-budget processor deliberately leaves ``size`` unused
        (``longest_edge=1``). That understates the video encoder ceiling by
        orders of magnitude, so recompute it from ``_get_video_max_pixels``
        and the sampler's own frame cap.
        """
        result: dict[str, int] = {}

        if mm_counts.get("image", 0) > 0:
            result["image"] = self.get_max_image_tokens()

        if mm_counts.get("video", 0) > 0:
            max_pixels = self._get_video_max_pixels()

            vision_config = self.get_hf_config().vision_config
            temporal_patch_size = vision_config.temporal_patch_size
            patch_size = vision_config.patch_size
            merge_size = vision_config.spatial_merge_size

            max_vision_tokens = max_pixels // (
                temporal_patch_size * patch_size**2 * merge_size**2
            )

            max_grid_t = max(
                self._video_frame_cap() // temporal_patch_size,
                1,
            )

            tokenizer = self.get_tokenizer()
            max_ts_tokens = max(
                len(tokenizer.encode(f"{t:.1f} seconds", add_special_tokens=False))
                for t in range(min(max_grid_t, 300))
            )

            result["video"] = max_vision_tokens + max_grid_t * (2 + max_ts_tokens) + 2

        return result

    def _get_video_second_idx_glm46v(
        self, metadata: dict[str, Any], total_frames: int, mm_kwargs=None
    ) -> list[int]:
        """Align video prompt timestamps with the processor's frame sampler.

        vLLM's GLM-4.6V re-derivation uses a different duration-threshold
        policy than GLM5Next's ``fps_interval`` sampler, so the placeholder
        frame count would not match the encoded ``grid_t``. Reuse the
        processor's sampler through :func:`glm_video_timestamp_seconds`.
        """
        from types import SimpleNamespace

        from vllm_fl.transformers_utils.processors.glm5_next import (
            glm_select_decoded_frames,
            glm_video_timestamp_seconds,
        )

        video_metadata = SimpleNamespace(
            fps=metadata.get("fps"),
            duration=metadata.get("duration"),
            total_num_frames=metadata.get("total_num_frames", total_frames),
            frames_indices=metadata.get("frames_indices"),
            do_sample_frames=metadata.get("do_sample_frames", True),
        )
        options = self.get_hf_processor().resolve_serving_kwargs(mm_kwargs or {})
        if not video_metadata.do_sample_frames:
            _, video_metadata.frames_indices = glm_select_decoded_frames(
                self.get_video_processor(),
                video_metadata,
                total_frames,
                **options["videos_kwargs"],
            )
        return glm_video_timestamp_seconds(
            self.get_video_processor(), video_metadata, **options["videos_kwargs"]
        )

    def _construct_glm5_video_placeholder(self, video, metadata, grid, mm_kwargs):
        processor = self.get_hf_processor(**mm_kwargs)
        config = self.get_hf_config()
        timestamps = self._get_video_second_idx_glm46v(metadata, len(video), mm_kwargs)
        frames, height, width = map(int, grid)
        if len(timestamps) != frames:
            raise ValueError("GLM5-Next video sampling and encoder grid disagree")
        num_tokens = height * width // processor.video_processor.merge_size**2
        placeholder = [config.video_start_token_id]
        for timestamp in timestamps:
            placeholder.append(config.image_start_token_id)
            placeholder.extend([processor.image_token_id] * num_tokens)
            placeholder.append(config.image_end_token_id)
            placeholder.extend(
                self.get_tokenizer().encode(
                    f"{timestamp:.1f} seconds", add_special_tokens=False
                )
            )
        placeholder.append(config.video_end_token_id)
        return placeholder


class Glm5NextMultiModalProcessor(Glm4vMultiModalProcessor):
    """Let vLLM, rather than the feature-only HF processor, update prompts."""

    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        if not mm_data:
            return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)
        from types import SimpleNamespace

        from vllm.multimodal.processing import BaseMultiModalProcessor

        from vllm_fl.transformers_utils.processors.glm5_next import (
            glm_select_decoded_frames,
        )

        processor = self.info.get_hf_processor(**mm_kwargs)
        options = processor.resolve_serving_kwargs(mm_kwargs)["videos_kwargs"]
        # vLLM may have already sampled a video to its media-loader cap. Apply
        # deployment/request sampling to that available subset before HF sees
        # do_sample_frames=False, retaining the source timeline for prompts.
        videos = mm_data.get("videos")
        if isinstance(videos, list):
            prepared = []
            for item in videos:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[1], Mapping)
                    and item[1].get("do_sample_frames") is False
                ):
                    video, metadata = item
                    rows, source_indices = glm_select_decoded_frames(
                        processor.video_processor,
                        SimpleNamespace(**metadata),
                        len(video),
                        **options,
                    )
                    item = (
                        video[rows],
                        dict(metadata, frames_indices=source_indices),
                    )
                prepared.append(item)
            mm_data = dict(mm_data, videos=prepared)
        data, kwargs = self._get_direct_path_inputs(mm_data, mm_kwargs)
        # Resolve request precedence before InputProcessingContext re-merges
        # deployment kwargs. Otherwise a nested deployment default can mask a
        # flat request override in Transformers' modality merge.
        kwargs = processor.resolve_serving_kwargs(kwargs)
        return BaseMultiModalProcessor._call_hf_processor(
            self, prompt, data, kwargs, tok_kwargs
        )

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        from dataclasses import replace
        from vllm.multimodal.processing import PromptUpdateDetails

        updates = super()._get_prompt_updates(
            mm_items, hf_processor_mm_kwargs, out_mm_kwargs
        )
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)

        def video_replacement(item_idx):
            video, metadata = mm_items["video"][item_idx]
            grid = out_mm_kwargs["video"][item_idx]["video_grid_thw"].data
            placeholder = self.info._construct_glm5_video_placeholder(
                video, metadata, grid, hf_processor_mm_kwargs
            )
            return PromptUpdateDetails.select_token_id(
                placeholder, embed_token_id=processor.image_token_id
            )

        return [
            replace(update, replacement=video_replacement)
            if update.modality == "video"
            else update
            for update in updates
        ]

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False


class Glm5NextDummyInputsBuilder(Glm4vDummyInputsBuilder):
    def get_dummy_processor_inputs(self, seq_len, mm_counts, mm_options):
        from dataclasses import replace

        inputs = super().get_dummy_processor_inputs(seq_len, mm_counts, mm_options)
        # The dummy already has the maximal reserved frame count. Profiling
        # must not shrink it according to a deployment's sampling rate.
        kwargs = dict(inputs.hf_processor_mm_kwargs)
        kwargs["videos_kwargs"] = {
            **kwargs.get("videos_kwargs", {}),
            "do_sample_frames": False,
        }
        return replace(inputs, hf_processor_mm_kwargs=kwargs)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options):
        # Build video directly at its effective canvas. Repeating the maximal
        # still-image canvas hundreds of times wastes host memory before resize.
        image_size = self.info.get_image_size_with_most_features()
        frames = self.info._video_frame_cap()
        video_h, video_w = self.info._vision_budget("video").largest_canvas(frames)
        return {
            "image": self._get_dummy_images(
                width=image_size.width,
                height=image_size.height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            ),
            "video": self._get_dummy_videos(
                width=video_w,
                height=video_h,
                num_frames=frames,
                num_videos=mm_counts.get("video", 0),
                overrides=mm_options.get("video"),
            ),
        }


@MULTIMODAL_REGISTRY.register_processor(
    Glm5NextMultiModalProcessor,
    info=Glm5NextProcessingInfo,
    dummy_inputs=Glm5NextDummyInputsBuilder,
)
class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid
):
    """GLM5-Next VLM: ViT-DP plus TP language layers and EP experts."""

    has_inner_state: ClassVar[Literal[True]] = True
    is_hybrid: ClassVar[Literal[True]] = True
    supports_encoder_tp_data = True
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        from vllm_fl.models.glm5_next import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        from vllm_fl.models.glm5_next import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm_fl.models.glm5_next import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        from vllm_fl.patches.glm5_next_v024 import validate_glm5_config

        validate_glm5_config(vllm_config)
        # Bypass Glm4vForConditionalGeneration.__init__: its language-model
        # architecture selection does not know GLM5-Next.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Glm5NextVisionTransformer(
                config.text_config,
                config.vision_config,
                norm_eps=config.vision_config.rms_norm_eps,
                # The vision checkpoint is BF16 even when the language model
                # comes from the separately published FP8 directory.
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Glm5NextForCausalLM"],
            )

    def get_encoder_cudagraph_config(self):
        config = super().get_encoder_cudagraph_config()
        config.buffer_keys = [key for key in config.buffer_keys if key != "pos_embeds"]
        return config

    def load_weights(self, weights):
        from vllm_fl.model_loader.glm5_next import (
            audit_text_weights,
            unquantized_weights,
        )

        with audit_text_weights(self.language_model):
            return super().load_weights(unquantized_weights(weights))


__all__ = [
    "Glm5NextForConditionalGeneration",
    "Glm5NextMultiModalProcessor",
    "Glm5NextProcessingInfo",
    "Glm5NextVisionTransformer",
]
