# Copyright (c) 2026 BAAI. All rights reserved.
"""Opaque custom op for DeepseekV4Attention.attention_impl (torch.compile).

Same design as the vendor fork's torch.ops.vllm.deepseek_v4_attention:
the whole attention_impl (wq_b GEMMs, KV insert, FlashMLA sparse, indexer,
workspace management) runs outside the compiled graph, looked up by layer
prefix via ForwardContext.no_compile_layers.
"""

import torch

from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op


def _fl_dsv4_attention(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    kv_score: torch.Tensor,
    indexer_kv_score: torch.Tensor,
    indexer_weights: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    self = get_forward_context().no_compile_layers[layer_name]
    self.attention_impl(
        hidden_states,
        qr,
        kv,
        kv_score,
        indexer_kv_score,
        indexer_weights,
        positions,
        out,
    )


def _fl_dsv4_attention_fake(
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    kv: torch.Tensor,
    kv_score: torch.Tensor,
    indexer_kv_score: torch.Tensor,
    indexer_weights: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="fl_dsv4_attention",
    op_func=_fl_dsv4_attention,
    mutates_args=["out"],
    fake_impl=_fl_dsv4_attention_fake,
)
