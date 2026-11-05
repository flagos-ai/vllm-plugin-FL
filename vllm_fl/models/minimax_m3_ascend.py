# SPDX-License-Identifier: Apache-2.0
"""Ascend-only MiniMax BF16 model entry; upstream classes remain unchanged."""

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.nvidia import model as upstream
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_fl.dispatch import CachedOp
from vllm_fl.models.minimax_m3_registration import add_fused_quantization_mapping

_get_kernels = CachedOp("minimax_m3_kernels")


class AscendGemmaRMSNorm(upstream.MiniMAXGemmaRMSNorm):
    def forward(self, x, residual=None):
        from vllm_fl.dispatch.backends.vendor.ascend.impl.minimax_m3 import gemma_norm

        if residual is None:
            return gemma_norm(x, self.weight, self.variance_epsilon)
        # Normalize the FP32 sum; only the returned residual is rounded to BF16.
        value = x.float() + residual.float()
        normalized = gemma_norm(value, self.weight, self.variance_epsilon)
        return normalized.to(x.dtype), value.to(residual.dtype)


def _sparse_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    context = get_forward_context()
    layer = context.no_compile_layers[layer_name]
    slots = context.slot_mapping
    if not isinstance(slots, dict) or layer_name not in slots:
        output.zero_()
        return
    layer._fl_kernels.insert_caches(
        layer,
        k,
        v,
        index_k,
        slots[layer_name],
        slots[layer.indexer.index_cache.prefix],
    )
    layer._fl_kernels.attend(layer, q, index_q, output)


def _sparse_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    pass


def _register_sparse_op():
    direct_register_custom_op(
        op_name="fl_minimax_m3_sparse_forward",
        op_func=_sparse_forward,
        mutates_args=["output"],
        fake_impl=_sparse_forward_fake,
        dispatch_key="PrivateUse1",
    )


class AscendMiniMaxM3Attention(upstream.MiniMaxM3Attention):
    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), -1)
        norm_rope = self._fl_kernels.norm_rope
        q = norm_rope(q, self.q_norm, positions, self.rotary_emb)
        k = norm_rope(k, self.k_norm, positions, self.rotary_emb)
        output = self.attn(q, k, v.contiguous())
        return self.o_proj(output)[0]


class AscendMiniMaxM3SparseAttention(upstream.MiniMaxM3SparseAttention):
    _fl_minimax_cache_owner = True

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, index_q, index_k = qkv.split(
            (
                self.q_size,
                self.kv_size,
                self.kv_size,
                self.index_q_size,
                self.idx_head_dim,
            ),
            -1,
        )
        norm_rope = self._fl_kernels.norm_rope
        q = norm_rope(q, self.q_norm, positions, self.rotary_emb)
        k = norm_rope(k, self.k_norm, positions, self.rotary_emb)
        index_q = norm_rope(index_q, self.index_q_norm, positions, self.rotary_emb)
        index_k = norm_rope(index_k, self.index_k_norm, positions, self.rotary_emb)
        output = torch.empty_like(q)
        torch.ops.vllm.fl_minimax_m3_sparse_forward(
            q,
            k,
            v.contiguous(),
            index_q,
            index_k,
            output,
            self.layer_name,
        )
        return self.o_proj(output)[0]


@support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "positions": 0, "inputs_embeds": 0},
)
class AscendMiniMaxM3Model(upstream.MiniMaxM3Model):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        kernels = _get_kernels()
        if getattr(kernels, "cache_contract", None) != "minimax_m3_block_major_bf16":
            raise ValueError("MiniMax provider must preserve BF16 block-major caches")
        # Adapt instances after upstream construction: weight names, registered
        # cache owners and upstream module globals keep their identity.
        for module in self.modules():
            if type(module) is upstream.MiniMAXGemmaRMSNorm:
                module.__class__ = AscendGemmaRMSNorm
            elif type(module) is upstream.MiniMaxM3SparseAttention:
                module.__class__ = AscendMiniMaxM3SparseAttention
                module._fl_kernels = kernels
                kernels.prepare_sparse(module)
                module.indexer.index_cache._fl_minimax_cache_owner = True
            elif type(module) is upstream.MiniMaxM3Attention:
                module.__class__ = AscendMiniMaxM3Attention
                module._fl_kernels = kernels


class MiniMaxM3SparseForCausalLM(upstream.MiniMaxM3SparseForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        nn.Module.__init__(self)
        if vllm_config.quant_config is not None:
            raise ValueError("Ascend MiniMax entry currently supports unquantized BF16")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("Ascend MiniMax entry currently requires BF16")
        if vllm_config.speculative_config is not None:
            raise ValueError("Ascend MiniMax speculative decoding is not supported")
        self.config = vllm_config.model_config.hf_text_config
        self.quant_config = None
        self.model = AscendMiniMaxM3Model(
            vllm_config=vllm_config,
            prefix=upstream.maybe_prefix(prefix, "model"),
        )
        self.lm_head = upstream.ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=None,
            prefix=upstream.maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = upstream.LogitsProcessor(self.config.vocab_size)


# The upstream VL wrapper creates its text backbone through ModelRegistry,
# which resolves to our CausalLM entry; its vision and processor stay upstream.
MiniMaxM3SparseForConditionalGeneration = (
    upstream.MiniMaxM3SparseForConditionalGeneration
)

add_fused_quantization_mapping(MiniMaxM3SparseForCausalLM)
_register_sparse_op()
