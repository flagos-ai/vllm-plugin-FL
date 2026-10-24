"""One explicit registration entry for the K3 correctness candidate."""

from importlib.util import find_spec

import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.models import ModelRegistry
from vllm.v1.attention.backends.mla.prefill import (
    MLAPrefillBackendEnum,
    register_mla_prefill_backend,
)

import vllm_fl.dispatch.backends.flaggems.impl.kimi_k3_cache as cache_ops
from vllm_fl.ops.kimi_k3.activation import situ_out
from vllm_fl.dispatch import CachedOp

gather = CachedOp("kimi_k3_cache_gather")
from vllm_fl.quantization.mxfp4.kimi_k3 import K3Mxfp4Config
from vllm_fl.attention.kimi_k3.vision import K3MMEncoderAttention

_REGISTERED = False
_LIBRARY = None


def register():
    global _REGISTERED, _LIBRARY
    if _REGISTERED:
        return

    if torch._C._dispatch_has_kernel_for_dispatch_key(
        "_C::get_cuda_view_from_cpu_tensor", "CPU"
    ):
        raise RuntimeError("Unexpected duplicate UVA runtime provider")
    runtime_spec = find_spec("vllm_fl.dispatch.backends.vendor.metax.kimi_k3_runtime")
    if runtime_spec is None or runtime_spec.origin is None:
        raise RuntimeError("Build the K3 MACA host runtime with VLLM_VENDOR=metax")
    torch.ops.load_library(runtime_spec.origin)
    register_quantization_config("mxfp4")(K3Mxfp4Config)
    register_mla_prefill_backend(
        MLAPrefillBackendEnum.CUSTOM, "vllm_fl.attention.kimi_k3.backend.K3PrefillBackend"
    )
    CustomOp.register_oot(
        _decorated_op_cls=K3MMEncoderAttention, name="MMEncoderAttention"
    )
    library = torch.library.Library("_C", "IMPL", "CUDA")
    for name, implementation in (
        ("fused_kimi_k3_mla_key_concat_kv_cache_insert", cache_ops.prefill),
        ("fused_kimi_k3_mla_decode_q_concat_kv_cache_insert", cache_ops.decode),
        ("fused_kimi_k3_mla_kv_concat", cache_ops.concat),
        ("situ_and_mul", situ_out),
    ):
        if torch._C._dispatch_has_kernel_for_dispatch_key("_C::" + name, "CUDA"):
            raise RuntimeError(f"Unexpected duplicate implementation for {name}")
        library.impl(name, implementation)
    cache_library = torch.library.Library("_C_cache_ops", "FRAGMENT")
    cache_library.define(
        "gather_and_maybe_dequant_cache(Tensor src_cache, Tensor! dst, "
        "Tensor block_table, Tensor cu_seq_lens, Tensor token_to_seq, "
        "int num_tokens, str kv_cache_dtype, Tensor scale, Tensor? seq_starts) -> ()"
    )
    cache_library.impl("gather_and_maybe_dequant_cache", gather, "CUDA")
    ModelRegistry.register_model(
        "KimiK3ForConditionalGeneration",
        "vllm_fl.models.kimi_k3:K3ForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "KimiLinearForCausalLM",
        "vllm_fl.models.kimi_k3:K3LinearForCausalLM",
    )
    _LIBRARY = (library, cache_library)
    _REGISTERED = True
