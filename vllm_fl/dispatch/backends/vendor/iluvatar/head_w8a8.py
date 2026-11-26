"""Native integer output projection for the untied MiniCPM5-2B head."""
import logging

import torch

from flag_gems.fused.minicpm.w8a8 import quantize_weight, shapes_ok

logger = logging.getLogger(__name__)
_HEAD_SHAPE = (130560, 2048)


def patch_output_projection(model_config, parallel_config):
    config = model_config.hf_config
    if (
        getattr(config, "tie_word_embeddings", None) is not False
        or (getattr(config, "vocab_size", None), getattr(config, "hidden_size", None)) != _HEAD_SHAPE
        or parallel_config.tensor_parallel_size != 1
    ):
        return

    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        UnquantizedEmbeddingMethod,
    )

    method = UnquantizedEmbeddingMethod
    if getattr(method.apply, "_e127_stock", None) is not None:
        return
    # Registered by patch_linear_w8a8_for_iluvatar; same op as E122 P0.
    project = torch.ops.vllm.iluvatar_w8a8_linear
    stock_process = method.process_weights_after_loading
    stock_apply = method.apply
    saw_b64 = False

    def process(self, layer):
        stock_process(self, layer)
        if not isinstance(layer, ParallelLMHead):
            return
        weight = layer.weight
        if weight.dtype != torch.bfloat16 or tuple(weight.shape) != _HEAD_SHAPE:
            return
        assert shapes_ok(weight.shape[1], weight.shape[0], False)
        codes, scales = quantize_weight(weight.detach(), False)
        layer.weight = torch.nn.Parameter(codes, requires_grad=False)
        layer.register_buffer("_e127_head_scale", scales, persistent=False)
        logger.info(
            "E127 untied W8A8 head loaded: shape=%s bytes=%d",
            tuple(codes.shape),
            codes.nbytes + scales.nbytes,
        )

    def apply(self, layer, x, bias=None):
        nonlocal saw_b64
        scales = getattr(layer, "_e127_head_scale", None)
        if scales is None:
            return stock_apply(self, layer, x, bias)
        flat = x.reshape(-1, x.shape[-1])
        if not saw_b64 and flat.shape[0] == 64:
            saw_b64 = True
            logger.info("E127 native W8A8 logits M64: %d columns", layer.weight.shape[0])
        logits = project(flat, layer.weight, scales, 0)
        logits = logits.reshape(*x.shape[:-1], logits.shape[-1])
        return logits if bias is None else logits + bias

    apply._e127_stock = stock_apply
    method.process_weights_after_loading = process
    method.apply = apply
