# SPDX-License-Identifier: Apache-2.0
"""Model-local FlagGems FA2 adapter; never replaces public vLLM functions."""

import torch
from torch import nn

from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_fl.dispatch.capabilities import get_flaggems_vllm_operator

_BINDING = None


def _use_math_sdpa():
    return current_platform.vendor_name == "kunlunxin"


def _packed_math_sdpa(query, key, value, **kwargs):
    """Compose existing math ops when the XPU FA2 compiler rejects selects."""
    assert kwargs["dropout_p"] == 0 and not kwargs["causal"]
    assert kwargs["cu_seqlens_q"] is kwargs["cu_seqlens_k"]
    boundaries = kwargs["cu_seqlens_q"][1:]
    positions = torch.arange(query.shape[0], device=query.device)
    # Device comparison/reduction makes refreshed sequence boundaries explicit
    # inputs to the graph, including on XPU runtimes with host-side fallbacks.
    sequences = (positions[:, None] >= boundaries[None, :]).sum(dim=1)
    result = torch.empty_like(query)
    keys = key.permute(1, 2, 0).float()
    values = value.transpose(0, 1).float()
    # A small query tile bounds mask/intermediate memory independently of the
    # packed token count. GPU boundaries remain dynamic during graph replay.
    for start in range(0, query.shape[0], 32):
        mask = sequences[start : start + 32, None] == sequences[None, :]
        queries = query[start : start + 32].transpose(0, 1).float()
        scores = (queries @ keys) * kwargs["softmax_scale"]
        scores.masked_fill_(~mask, float("-inf"))
        output = scores.softmax(dim=-1) @ values
        result[start : start + 32] = output.transpose(0, 1)
    return result


def get_vision_binding():
    global _BINDING
    if _BINDING is None:
        from vllm_fl.dispatch.binding import OperatorBinding
        from vllm_fl.dispatch.manager import OpManager
        from vllm_fl.dispatch.types import BackendImplKind, OpImpl
        from vllm_fl.utils import use_flaggems_op

        def flag(*args, **kwargs):
            # This is a required model operator, like the MLA/indexer APIs.
            # Use the public packed/non-paged FA2 contract on every backend.
            flash_attn_varlen_func = get_flaggems_vllm_operator(
                "flash_attn_varlen_func"
            )
            return flash_attn_varlen_func(*args, **kwargs)

        def available():
            if _use_math_sdpa() or not use_flaggems_op("flash_attn_varlen_func"):
                return False
            return callable(get_flaggems_vllm_operator("flash_attn_varlen_func"))

        flag._is_available = available
        manager = OpManager()
        manager.registry.register_impl(
            OpImpl(
                "flash_attn_varlen_func",
                "glm5.vision.flaggems",
                BackendImplKind.DEFAULT,
                flag,
            )
        )
        _packed_math_sdpa._is_available = lambda: (
            _use_math_sdpa() and use_flaggems_op("flash_attn_varlen_func")
        )
        manager.registry.register_impl(
            OpImpl(
                "flash_attn_varlen_func",
                "glm5.vision.torch",
                BackendImplKind.REFERENCE,
                _packed_math_sdpa,
            )
        )
        manager.bump_policy_epoch()
        _BINDING = OperatorBinding(
            manager,
            "flash_attn_varlen_func",
            graph_capabilities={
                "glm5.vision.flaggems": True,
                "glm5.vision.torch": True,
            },
        )
    return _BINDING


def _vision_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    flash_attn_varlen_func = get_vision_binding()

    batch, seq_len, heads, head_dim = query.shape
    if cu_seqlens is None:
        cu_seqlens = torch.arange(
            0,
            (batch + 1) * seq_len,
            seq_len,
            dtype=torch.int32,
            device=query.device,
        )
    # The packed token count is a static upper bound for every sequence.
    # Do not read max_seqlen from a CUDA tensor inside graph capture.
    result = flash_attn_varlen_func(
        query.reshape(-1, heads, head_dim),
        key.reshape(-1, heads, head_dim),
        value.reshape(-1, heads, head_dim),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=batch * seq_len,
        max_seqlen_k=batch * seq_len,
        dropout_p=0.0,
        causal=False,
        softmax_scale=scale,
        fa_version=2,
    )
    # Only this vision caller consumes the output alone. Public FA callers
    # retain their original tuple/LSE contract.
    if isinstance(result, tuple):
        result = result[0]
    return result.reshape_as(query)


def _vision_attention_fake(query, key, value, cu_seqlens, scale):
    return torch.empty_like(query)


direct_register_custom_op(
    op_name="glm5_vision_attention",
    op_func=_vision_attention,
    fake_impl=_vision_attention_fake,
)


class Glm5VisionAttention(nn.Module):
    def __init__(self, num_heads, head_size, scale, prefix=""):
        super().__init__()
        self.scale = scale
        # This adapter requires FA2; a conflicting user policy is a startup
        # error, before loading weights or warming the encoder.
        get_vision_binding().preflight()

    def forward(self, query, key, value, cu_seqlens=None, max_seqlen=None):
        return torch.ops.vllm.glm5_vision_attention(
            query, key, value, cu_seqlens, self.scale
        )
