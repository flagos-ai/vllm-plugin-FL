# Copyright (c) 2025 BAAI. All rights reserved.

import importlib
import os
import logging
import sys

# torch.float4_e2m1fn_x2 exists only in CUDA builds of PyTorch 2.7+.
# vllm.ir.tolerances references it at module level, so we inject a sentinel
# before any vllm.ir import can happen.
if "torch" in sys.modules:
    _torch = sys.modules["torch"]
    if not hasattr(_torch, "float4_e2m1fn_x2"):
        _torch.float4_e2m1fn_x2 = _torch.uint8
else:
    import torch as _torch
    if not hasattr(_torch, "float4_e2m1fn_x2"):
        _torch.float4_e2m1fn_x2 = _torch.uint8
del _torch

# --- torch 2.7.1+cpu (cambricon 4.4.3) compat shims ---------------------

import torch

# torch-mlu registers `_C::get_mlu_view_from_cpu_tensor` as a
# CompositeImplicitAutograd op that has no Python handle via torch.ops._C.
# torch._export.utils._materialize_cpp_cia_ops() getattr()s every CIA op and
# raises AttributeError on it, aborting torch._inductor init (first triggered
# by vllm.utils.deep_gemm's module-level @torch.compile). Skip CIA ops the
# dispatcher has no Python handle for.
try:
    import torch._export.utils as _export_utils

    def _tolerant_materialize_cpp_cia_ops():
        for op in torch._C._dispatch_get_registrations_for_dispatch_key(
            "CompositeImplicitAutograd"
        ):
            namespace, full = tuple(op.split("::"))
            parts = full.split(".")
            name = parts[0]
            overload = "default" if len(parts) == 1 else parts[1]
            try:
                _ = getattr(getattr(getattr(torch.ops, namespace), name), overload)
            except AttributeError:
                continue

    _export_utils._materialize_cpp_cia_ops = _tolerant_materialize_cpp_cia_ops
except Exception:
    pass

# flag_gems 5.3.5 populates current_work_registrar.torch_ops_map via
# torch.library.get_kernel(), which only exists in torch 2.8+. torch 2.7.1+cpu
# (cambricon 4.4.3) lacks it, so the map stays empty and the generated copy_
# pre/post hooks raise KeyError: 'aten::copy_'. Provide a torch 2.7.1-compatible
# get_kernel that redispatches to the native (CompositeExplicitAutograd) kernel.
if not hasattr(torch.library, "get_kernel"):
    _FALLBACK_KEYSET = torch._C.DispatchKeySet(
        torch._C.DispatchKey.CompositeExplicitAutograd
    )

    class _RedispatchKernel:
        def __init__(self, qualified_name):
            self._qualified_name = qualified_name

        def call_boxed(self, keyset, *args, **kwargs):
            namespace, name = self._qualified_name.split("::")
            op = getattr(getattr(torch.ops, namespace), name)
            return op.default.redispatch(_FALLBACK_KEYSET, *args, **kwargs)

    def _get_kernel(name_or_op, dispatch_key):
        if isinstance(name_or_op, str):
            qualified_name = name_or_op
        else:
            qualified_name = name_or_op._qualified_op_name
        return _RedispatchKernel(qualified_name)

    torch.library.get_kernel = _get_kernel

from vllm_fl.utils import get_op_config as _get_op_config

from . import version as version  # PyTorch-style: vllm_fl.version.git_version


logger = logging.getLogger(__name__)


def __getattr__(name):
    if name == "distributed":
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _patch_transformers_compat():
    """Patch transformers compatibility for ALLOWED_LAYER_TYPES and tokenizer."""
    import transformers.configuration_utils as cfg
    if not hasattr(cfg, "ALLOWED_LAYER_TYPES"):
        cfg.ALLOWED_LAYER_TYPES = getattr(
            cfg, "ALLOWED_ATTENTION_LAYER_TYPES", ()
        )


def _register_flagcx_connector():
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    for _alias in ("FlagCXConnector", "FlagcxConnector"):
        if _alias not in KVConnectorFactory._registry:
            KVConnectorFactory.register_connector(
                _alias,
                "vllm_fl.distributed.kv_transfer.flagcx_connector",
                "FlagCXConnector",
            )


def _patch_flash_attn_import():
    """Stub vllm.vllm_flash_attn if CUDA flash attention C extensions are missing."""
    import sys
    if "vllm.vllm_flash_attn" in sys.modules:
        return
    try:
        import vllm.vllm_flash_attn  # noqa: F401
    except ImportError:
        import types
        stub = types.ModuleType("vllm.vllm_flash_attn")
        stub.FA2_AVAILABLE = False
        stub.FA3_AVAILABLE = False
        stub.fa_version_unsupported_reason = lambda *a, **kw: "flash_attn C extensions not available"
        stub.flash_attn_varlen_func = None
        stub.get_scheduler_metadata = None
        stub.is_fa_version_supported = lambda *a, **kw: False
        sys.modules["vllm.vllm_flash_attn"] = stub


def _patch_custom_ops():
    """Register fallback schemas when neither vLLM extension ABI is present."""
    for module_name in ("vllm._C", "vllm._C_stable_libtorch"):
        try:
            importlib.import_module(module_name)
            return
        except (ImportError, OSError):
            continue

    try:
        import vllm_fl._C  # noqa: F401
    except (ImportError, OSError) as e:
        logger.debug("Failed to import vllm_fl._C: %s", e)

    from vllm_fl.ops._C_ops_registry import register_op_schemas
    register_op_schemas()


def register():
    """Register the FL platform."""
    _patch_custom_ops()
    _patch_flash_attn_import()
    _patch_transformers_compat()

    # Model-specific platform patches
    from vllm_fl.patches.glm_moe_dsa import apply_platform_patches as glm5_platform
    glm5_platform()

    # Note: FlagCX connector registration is deferred to register_model()
    # to avoid circular imports during VllmConfig.__post_init__ in spawned
    # subprocesses.

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _get_op_config()

    return "vllm_fl.platform.PlatformFL"

def register_quant_linear():
    from vllm.platforms import current_platform
    # vllm.model_executor.kernels.linear triggers cutlass_scaled_mm_supports_fp8
    # at module level, which requires torch.ops._C — not available on MUSA.
    if current_platform.device_type == "musa":
        return
    from vllm_fl.quantization.quant_linear import add_oot_quant_kernel
    add_oot_quant_kernel()

def register_router():
    from vllm.platforms import current_platform
    # fused_moe import chain triggers cutlass_scaled_mm_supports_fp8 on MUSA
    if current_platform.device_type == "musa":
        return
    from vllm_fl.utils import is_oot_enabled
    if not is_oot_enabled():
        return
    from vllm_fl.ops.fused_moe.router import replace_router_with_fl
    replace_router_with_fl()


def _register_gdn_packed_decode_patch() -> bool:
    """Install the packed GDN fix when this vLLM build provides it.

    Vendor images may omit vLLM's FLA package or route GDN through a different
    implementation. Keep the compatibility hook capability-based: any build
    carrying the vulnerable kernel is patched, while builds without the
    required module or symbol remain untouched.
    """
    try:
        patch_module = importlib.import_module("vllm_fl.patches.gdn_packed_decode")
        patch_fn = patch_module.patch_vllm_packed_gdn_beta
    except (ImportError, AttributeError) as exc:
        logger.debug("Packed GDN decode patch is unavailable: %s", exc)
        return False

    return patch_fn()


def register_model():
    """Register FL-specific models not yet upstream."""
    # General plugins are loaded independently in spawned model-inspection and
    # worker processes, so all runtime compatibility hooks must be idempotent.
    from vllm_fl.patches.moe_sum import patch_vllm_moe_sum
    from vllm_fl.patches.qwen3_5_text import apply_qwen3_5_text_patches

    apply_qwen3_5_text_patches()
    patch_vllm_moe_sum()

    _register_flagcx_connector()

    # Register OOT quant kernels so kernel selection can find them
    register_quant_linear()
    register_router()

    _register_gdn_packed_decode_patch()

    # Register GLM-5 (GlmMoeDsa) — config not yet upstream
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
        from vllm_fl.configs.glm_moe_dsa import GlmMoeDsaConfig
        _CONFIG_REGISTRY["glm_moe_dsa"] = GlmMoeDsaConfig

        #from vllm_fl.patches.glm_moe_dsa import apply_model_patches as glm5_model
        #glm5_model()
    except Exception as e:
        logger.error(f"Register GlmMoeDsa model error: {str(e)}")

# flag_gems 5.3.5 cambricon backend emits a task_type='block' triton launch
# kwarg unsupported by triton 3.2.0+mlu1.7.2 (cambricon 4.4.3). Strip it from
# JITFunction.run. torch_mlu must be imported first — its _inductor module
# imports triton.Config during triton init, so patching triton earlier raises a
# circular-import error. Guarded on torch_mlu importability (cambricon only).
try:
    import torch_mlu  # noqa: F401
    import triton.runtime.jit as _tr_jit

    _orig_run = _tr_jit.JITFunction.run
    if not getattr(_orig_run, "_flagos_task_type_patched", False):

        def _run_no_task_type(self, *args, **kwargs):
            kwargs.pop("task_type", None)
            return _orig_run(self, *args, **kwargs)

        _run_no_task_type._flagos_task_type_patched = True
        _tr_jit.JITFunction.run = _run_no_task_type
except ImportError:
    pass