"""Transactional K3 registration for the pinned native runtime."""

import inspect
import os
from contextlib import ExitStack
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
from vllm_fl.dispatch import CachedOp
from vllm_fl.ops.kimi_k3.activation import situ_out

gather = CachedOp("kimi_k3_cache_gather")
from vllm_fl.attention.kimi_k3.vision import K3MMEncoderAttention
from vllm_fl.quantization.mxfp4.kimi_k3 import K3Mxfp4Config

_REGISTERED = False
_LIBRARY = None
_MISSING = object()
_KERNELS = (
    ("fused_kimi_k3_mla_key_concat_kv_cache_insert", cache_ops.prefill),
    ("fused_kimi_k3_mla_decode_q_concat_kv_cache_insert", cache_ops.decode),
    ("fused_kimi_k3_mla_kv_concat", cache_ops.concat),
    ("situ_and_mul", situ_out),
)
_MODELS = {
    "KimiK3ForConditionalGeneration": "vllm_fl.models.kimi_k3:K3ForConditionalGeneration",
    "KimiLinearForCausalLM": "vllm_fl.models.kimi_k3:K3LinearForCausalLM",
}
_CACHE_SCHEMA = (
    "gather_and_maybe_dequant_cache(Tensor src_cache, Tensor! dst, "
    "Tensor block_table, Tensor cu_seq_lens, Tensor token_to_seq, "
    "int num_tokens, str kv_cache_dtype, Tensor scale, Tensor? seq_starts) -> ()"
)


def _restore_entry(mapping, key, previous):
    if previous is _MISSING:
        mapping.pop(key, None)
    else:
        mapping[key] = previous


def _restore_name(previous):
    if previous is _MISSING:
        if "name" in vars(K3MMEncoderAttention):
            delattr(K3MMEncoderAttention, "name")
    else:
        K3MMEncoderAttention.name = previous


def register():
    global _REGISTERED, _LIBRARY
    if _REGISTERED:
        return

    from vllm.model_executor import custom_op
    from vllm.model_executor.layers import quantization

    from vllm_fl.patches.kimi_k3 import sampling

    # Only this pinned compatibility entry touches the quantization registry:
    # native vLLM exposes registration but no removal API. Restore just our key.
    quant_configs = getattr(quantization, "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG", None)
    signatures = (
        (register_quantization_config, ("quantization",)),
        (register_mla_prefill_backend, ("backend", "class_path")),
        (CustomOp.register_oot, ("_decorated_op_cls", "name")),
        (ModelRegistry.register_model, ("model_arch", "model_cls")),
    )
    if any(
        tuple(inspect.signature(fn).parameters) != names for fn, names in signatures
    ):
        raise RuntimeError("K3 registration requires the pinned native registry APIs")
    if (
        not isinstance(quant_configs, dict)
        or not isinstance(ModelRegistry.models, dict)
        or not isinstance(custom_op.op_registry_oot, dict)
        or "mxfp4" not in quantization.QUANTIZATION_METHODS
        or not issubclass(K3Mxfp4Config, quantization.QuantizationConfig)
        or not callable(getattr(torch.library.Library, "_destroy", None))
    ):
        raise RuntimeError("Incompatible K3 registry or library lifetime contract")
    if "MMEncoderAttention" in custom_op.op_registry_oot:
        raise RuntimeError("Unexpected duplicate MMEncoderAttention provider")
    for name, _ in _KERNELS:
        torch._C._dispatch_find_schema_or_throw("_C::" + name, "")
        if torch._C._dispatch_has_kernel_for_dispatch_key("_C::" + name, "CUDA"):
            raise RuntimeError(f"Unexpected duplicate implementation for {name}")
    if (
        "_C_cache_ops::gather_and_maybe_dequant_cache"
        in torch._C._dispatch_get_all_op_names()
    ):
        raise RuntimeError("Unexpected duplicate K3 cache schema")
    sampler_targets = sampling.prepare()
    backend = MLAPrefillBackendEnum.CUSTOM
    old_mla = backend.get_path() if backend.is_overridden() else None
    runtime_spec = find_spec("vllm_fl.dispatch.backends.vendor.metax.kimi_k3_runtime")
    if runtime_spec is None or runtime_spec.origin is None:
        raise RuntimeError("Build the K3 MACA host runtime with VLLM_VENDOR=metax")
    origin = os.path.realpath(runtime_spec.origin)
    runtime_loaded = origin in torch.ops.loaded_libraries
    has_uva = torch._C._dispatch_has_kernel_for_dispatch_key(
        "_C::get_cuda_view_from_cpu_tensor", "CPU"
    )
    if has_uva and not runtime_loaded:
        raise RuntimeError("Unexpected duplicate UVA runtime provider")

    # A native library cannot be unloaded. Keep this independent host-runtime
    # prerequisite once per process; failed model registration can safely retry.
    if not runtime_loaded:
        torch.ops.load_library(origin)

    with ExitStack() as rollback:
        rollback.callback(
            _restore_entry, quant_configs, "mxfp4", quant_configs.get("mxfp4", _MISSING)
        )
        rollback.callback(
            _restore_entry,
            custom_op.op_registry_oot,
            "MMEncoderAttention",
            custom_op.op_registry_oot.get("MMEncoderAttention", _MISSING),
        )
        old_name = vars(K3MMEncoderAttention).get("name", _MISSING)
        rollback.callback(_restore_name, old_name)
        for name in _MODELS:
            rollback.callback(
                _restore_entry,
                ModelRegistry.models,
                name,
                ModelRegistry.models.get(name, _MISSING),
            )
        if old_mla is None:
            rollback.callback(backend.clear_override)
        else:
            rollback.callback(register_mla_prefill_backend, backend, old_mla)
        for target in sampler_targets or ():
            rollback.callback(
                setattr, target, "apply_top_k_top_p", target.apply_top_k_top_p
            )

        register_quantization_config("mxfp4")(K3Mxfp4Config)
        register_mla_prefill_backend(
            backend, "vllm_fl.attention.kimi_k3.backend.K3PrefillBackend"
        )
        CustomOp.register_oot(
            _decorated_op_cls=K3MMEncoderAttention, name="MMEncoderAttention"
        )
        library = torch.library.Library("_C", "IMPL", "CUDA")
        rollback.callback(library._destroy)
        for name, implementation in _KERNELS:
            library.impl(name, implementation)
        cache_library = torch.library.Library("_C_cache_ops", "FRAGMENT")
        rollback.callback(cache_library._destroy)
        cache_library.define(_CACHE_SCHEMA)
        cache_library.impl("gather_and_maybe_dequant_cache", gather, "CUDA")
        for name, target in _MODELS.items():
            ModelRegistry.register_model(name, target)
        sampling.install()
        _LIBRARY = (library, cache_library)
        _REGISTERED = True
        rollback.pop_all()
