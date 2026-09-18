# SPDX-License-Identifier: Apache-2.0
"""Install M3 model hooks without rewriting files in the vLLM installation."""

_INSTALLED = False


def _gemma_forward(self, x, residual=None):
    from .layers import gemma

    return gemma(x, self.weight, self.variance_epsilon, residual)


def _vision_rotary(
    self, qk_reshaped, rotary_cos, rotary_sin, seq_len, rotary_segment_lengths
):
    from .vision import vision_rope_op

    return vision_rope_op(qk_reshaped, rotary_cos, rotary_sin)


def _patch_embed_forward(self, pixel_values):
    from .layers import patch_embed

    projection = self.patch_embedding
    if projection.weight.dtype != pixel_values.dtype:
        self.patch_embedding = projection = projection.to(pixel_values.dtype)
    # Only non-overlapping, complete Conv3d patches are a row-wise GEMM.
    assert projection.bias is None and projection.groups == 1
    assert projection.padding == (0, 0, 0)
    assert projection.dilation == (1, 1, 1)
    assert projection.kernel_size == projection.stride
    return patch_embed(pixel_values, projection.weight)


def install() -> bool:
    global _INSTALLED
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "metax":
        return False
    if _INSTALLED:
        return False

    from vllm import _custom_ops
    from vllm.models.minimax_m3.common import indexer, sparse_attention, vision_tower
    from vllm.models.minimax_m3.nvidia import model

    from . import layers, msa
    from .sparse_config import install_safe_launches
    from .vision import M3VisionAttentionFL
    from .vision_chunk import vision_forward

    # The public Python wrapper is model-specific. Do not fabricate a private
    # torch.ops._C ABI or override unrelated CUDA operators.
    _custom_ops.fused_minimax_m3_qknorm_rope_kv_insert = layers.qknorm_rope_insert
    model.MiniMAXGemmaRMSNorm.forward = _gemma_forward
    model.SiluAndMulWithClamp = layers.SiluAndMulWithClampFL

    install_safe_launches()
    for name in (
        "minimax_m3_index_score",
        "minimax_m3_index_topk",
        "minimax_m3_index_decode",
    ):
        setattr(indexer, name, getattr(msa, name))
    for name in ("minimax_m3_sparse_attn", "minimax_m3_sparse_attn_decode"):
        setattr(sparse_attention, name, getattr(msa, name))

    vision_tower.MMEncoderAttention = M3VisionAttentionFL
    vision_tower.MiniMaxVLAttention._apply_rotary_emb = _vision_rotary
    vision_tower.MiniMaxVLPatchEmbed.forward = _patch_embed_forward
    tower = vision_tower.MiniMaxVLVisionModel
    tower._m3_forward_unbounded = tower.forward
    tower.forward = vision_forward

    from vllm_fl.patches.minimax_m3 import install_collective_boundary

    install_collective_boundary()
    _INSTALLED = True
    return True
