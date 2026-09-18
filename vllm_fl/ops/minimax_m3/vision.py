# SPDX-License-Identifier: Apache-2.0
import flag_gems
import torch

from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention as _MMEncoderBase,
)

from vllm_fl.dispatch import CachedOp

vision_rope_op = CachedOp("m3_vision_rope")
vision_attention_op = CachedOp("m3_vision_attention")


def vision_rope(x, cos, sin):
    half = cos.shape[-1]
    rd = 2 * half
    assert rd <= x.shape[-1] and cos.shape == sin.shape
    rot = x[..., :rd].float()
    c = cos.float().unsqueeze(-2)
    s = sin.float().unsqueeze(-2)
    a, b = rot.split(half, dim=-1)
    y = torch.cat((a * c - b * s, b * c + a * s), dim=-1).to(x.dtype)
    return torch.cat((y, x[..., rd:]), dim=-1) if rd < x.shape[-1] else y


def vision_attention(query, key, value, cu_seqlens, max_seqlen, scale):
    assert query.ndim == 4 and query.shape[0] == 1 and query.dtype == torch.bfloat16
    assert key.shape == value.shape and key.shape[:2] == query.shape[:2]
    n = query.shape[1]
    cu = (
        cu_seqlens.to(device=query.device, dtype=torch.int32)
        if cu_seqlens is not None
        else torch.tensor([0, n], device=query.device, dtype=torch.int32)
    )
    maxlen = (
        int(max_seqlen.item())
        if isinstance(max_seqlen, torch.Tensor)
        else int(max_seqlen or n)
    )
    q = query.squeeze(0).contiguous()
    k = key.squeeze(0).contiguous()
    v = value.squeeze(0).contiguous()
    result = flag_gems.flash_attn_varlen_func(
        q,
        k,
        v,
        maxlen,
        cu,
        maxlen,
        cu,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=False,
    )
    assert isinstance(result, torch.Tensor)
    return result.unsqueeze(0)


class M3VisionAttentionFL(_MMEncoderBase):
    def __init__(self, num_heads, head_size, scale=None, num_kv_heads=None, prefix=""):
        torch.nn.Module.__init__(self)
        self.fp8_enabled = False
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.scale = head_size**-0.5 if scale is None else scale
        self.layer_name = prefix

    def forward(
        self, query, key, value, cu_seqlens=None, max_seqlen=None, sequence_lengths=None
    ):
        return vision_attention_op(
            query, key, value, cu_seqlens, max_seqlen, self.scale
        )


def register(registry):
    from . import torch_ops
    from vllm_fl.dispatch import BackendImplKind, OpImpl

    for name, fn in [
        ("vision_rope", vision_rope),
        ("vision_attention", vision_attention),
    ]:
        registry.register_impl(
            OpImpl(
                op_name="m3_" + name,
                impl_id="m3.flagos." + name,
                kind=BackendImplKind.DEFAULT,
                fn=getattr(torch_ops, name),
                priority=150,
            )
        )
