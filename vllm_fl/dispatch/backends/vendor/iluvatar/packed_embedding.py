# SPDX-License-Identifier: Apache-2.0
"""Register lossless storage for the model's independent input embedding."""

import logging
from functools import wraps

import torch


def register_packed_input_embedding():
    from flag_gems.fused.iluvatar_packed_embedding import encode, lookup
    from vllm.compilation import caching
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
        VocabParallelEmbedding,
    )
    from vllm.utils.torch_utils import direct_register_custom_op

    method = UnquantizedEmbeddingMethod
    if getattr(method, "_iluvatar_packed_input", False):
        return

    def gather(ids: torch.Tensor, low: torch.Tensor, planes: torch.Tensor,
               ends: torch.Tensor, side: torch.Tensor, base: int) -> torch.Tensor:
        return lookup(ids, low, planes, ends, side, base)

    def fake(ids, low, planes, ends, side, base):
        return torch.empty((*ids.shape, low.shape[1]), dtype=torch.bfloat16, device=low.device)

    direct_register_custom_op(
        op_name="iluvatar_lossless_embedding", op_func=gather, mutates_args=[],
        fake_impl=fake, dispatch_key="CUDA",
    )
    # A pre-existing serialized native embedding graph expects a BF16 weight.
    # The packed graph passes byte arrays and offsets to a different op. vLLM
    # selects its AOT artifact before Dynamo can inspect these arguments.
    native_factors = caching.aot_compile_hash_factors

    @wraps(native_factors)
    def embedding_graph_factors(config):
        return [*native_factors(config), "vllm_fl.iluvatar_lossless_embedding.external_kv_write.v2.decode_tiles_1119.dense_long_query.champ_k8v4_kv_v3"]

    caching.aot_compile_hash_factors = embedding_graph_factors
    native_process = method.process_weights_after_loading
    native_embedding = method.embedding

    @wraps(native_process)
    def process(self, layer):
        native_process(self, layer)
        if (type(layer) is not VocabParallelEmbedding or layer.tp_size != 1
                or not layer.weight.is_cuda or layer.weight.dtype != torch.bfloat16
                or not layer.weight.is_contiguous() or layer.weight.shape != (130560, 2048)):
            return
        config = get_current_vllm_config()
        hf = config.model_config.hf_config
        if (getattr(hf, "tie_word_embeddings", True)
                or getattr(hf, "architectures", None) != ["LlamaForCausalLM"]
                or config.lora_config is not None or config.model_config.enable_sleep_mode
                or getattr(config, "weight_transfer_config", None) is not None):
            return

        packed = encode(layer.weight)
        if packed is None:
            return
        low, planes, ends, side, base = packed
        original_bytes = layer.weight.nbytes
        layer.weight.data = low
        layer.register_buffer("_iluvatar_planes", planes)
        layer.register_buffer("_iluvatar_ends", ends)
        layer.register_buffer("_iluvatar_side", side)
        layer._iluvatar_exp_base = base
        retained = sum(t.nbytes for t in (low, planes, ends, side))
        logging.getLogger(__name__).info(
            "Lossless Iluvatar BF16 input embedding: %d bytes -> %d bytes, exponent base %d",
            original_bytes, retained, base,
        )

    @wraps(native_embedding)
    def embedding(self, layer, ids):
        if layer.weight.dtype == torch.uint8 and hasattr(layer, "_iluvatar_planes"):
            return torch.ops.vllm.iluvatar_lossless_embedding(
                ids, layer.weight, layer._iluvatar_planes, layer._iluvatar_ends,
                layer._iluvatar_side, layer._iluvatar_exp_base,
            )
        return native_embedding(self, layer, ids)

    method.process_weights_after_loading = process
    method.embedding = embedding
    method._iluvatar_packed_input = True
