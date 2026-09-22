# Copyright (c) 2025 BAAI. All rights reserved.

import contextlib
import importlib
import logging
import os
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

# torch.distributed._symmetric_memory exists only in PyTorch 2.8+.
# vllm.distributed.parallel_state imports it at module level, so vendor
# torch builds < 2.8 (iluvatar corex on 2.7.x) die with ImportError before
# engine start. vllm's later uses of the module are lazy, so an empty
# pre-registered stub suffices to get past the import gate.
# Probe by importing it: importlib pulls the submodule in, whereas
# hasattr() on the parent reads False until something else imports it, and
# a stub registered on that reading would shadow the real module.
#
# Unavailable comes in two shapes, and both need the stub: the package is
# absent (ModuleNotFoundError, cambricon 4.4.3), or present but its own
# __init__ wants a torch 2.8 symbol (ImportError, iluvatar corex 4.4.0, #27).
# A ModuleNotFoundError naming anything else is a missing dependency of the
# real module, so it keeps raising.
try:
    importlib.import_module("torch.distributed._symmetric_memory")
except ModuleNotFoundError as _exc:
    if _exc.name != "torch.distributed._symmetric_memory":
        raise
    _symm_mem_available = False
except ImportError:
    _symm_mem_available = False
else:
    _symm_mem_available = True

if not _symm_mem_available:
    import types as _types

    _symm_mem_stub = _types.ModuleType("torch.distributed._symmetric_memory")
    sys.modules["torch.distributed._symmetric_memory"] = _symm_mem_stub
    # Importing a real submodule also sets the parent attribute; a hand-made
    # sys.modules entry does not, so mirror it for attribute access and
    # from-imports.
    importlib.import_module("torch.distributed")._symmetric_memory = _symm_mem_stub
    del _symm_mem_stub, _types
del _symm_mem_available

# --- torch 2.7.1+cpu (cambricon 4.4.3) compat shims ---------------------
import torch

from . import version as version  # PyTorch-style: vllm_fl.version.git_version

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
    # Vulkan/MUSA builds without the cpp CIA shim: nothing to wrap.
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

logger = logging.getLogger(__name__)


def _arm_cpu_platform() -> str | None:
    """Return vLLM's CPU platform on AArch64 hosts, if available."""
    import platform

    if platform.machine().lower() not in {"aarch64", "arm64"}:
        return None
    from vllm.platforms import cpu_platform_plugin

    return cpu_platform_plugin()


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
        cfg.ALLOWED_LAYER_TYPES = getattr(cfg, "ALLOWED_ATTENTION_LAYER_TYPES", ())


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
        stub.fa_version_unsupported_reason = lambda *a, **kw: (
            "flash_attn C extensions not available"
        )
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


def _patch_torch_accelerator():
    """Complete the metax (MACA) torch.accelerator API.

    The MACA torch fork ships only the device-management subset of
    torch.accelerator and omits the memory-stats subset that vLLM >= 0.24
    calls unconditionally (MemorySnapshot, memory_profiling, weight loaders).
    Bind the torch.cuda equivalents; current_accelerator() == "cuda" on
    metax. The mtgpu allocator reports 0/empty stats, which vllm tolerates.
    """
    import flag_gems
    import torch

    # MACA is the only vendor whose torch fork is missing this API; binding
    # torch.cuda functions onto another vendor's accelerator would mask a real
    # gap there.
    if flag_gems.vendor_name != "metax":
        return

    if not hasattr(torch, "accelerator"):
        return
    accel = torch.accelerator
    if hasattr(accel, "memory_stats"):
        # Full implementation present - nothing to do.
        return

    for _name in (
        "empty_cache",
        "memory_reserved",
        "memory_stats",
        "memory_allocated",
        "max_memory_allocated",
    ):
        if not hasattr(accel, _name) and hasattr(torch.cuda, _name):
            setattr(accel, _name, getattr(torch.cuda, _name))

    if not hasattr(accel, "reset_peak_memory_stats"):
        _cuda_reset = torch.cuda.reset_peak_memory_stats

        def _safe_reset_peak_memory_stats(device=None):
            try:
                _cuda_reset(device)
            except RuntimeError:
                # mtgpu backend may reject an explicit device before the
                # allocator is initialized; the no-arg variant is the fallback.
                # If the allocator is not initialized at all, there is nothing
                # to reset.
                with contextlib.suppress(RuntimeError):
                    _cuda_reset()

        accel.reset_peak_memory_stats = _safe_reset_peak_memory_stats


def _init_vendor_device():
    """Vendor-specific device initialization patches."""
    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().vendor_name == "kunlunxin":
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils import (
            _patch_xpu_get_device,
        )

        _patch_xpu_get_device()


def register():
    """Register the FL platform."""
    # PlatformFL is accelerator-shaped. For the standard FlagGems ARM target,
    # preserve vLLM's stock CPU platform and install kernels in register_model().
    arm_cpu_platform = _arm_cpu_platform()
    if arm_cpu_platform is not None:
        logger.info("[vllm_fl] ARM64 CPU target -> vLLM CPU platform")
        return arm_cpu_platform

    _patch_torch_accelerator()
    _init_vendor_device()
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
    from vllm_fl.patches.qwen3_5_text import apply_qwen3_5_text_patches

    apply_qwen3_5_text_patches()

    from vllm.platforms import current_platform

    if current_platform.device_type == "cpu" and _arm_cpu_platform() is not None:
        from vllm_fl.patches.arm_cpu_gdn import (
            apply_arm_cpu_gdn_state_indices_patch,
        )

        apply_arm_cpu_gdn_state_indices_patch()

        # FlagGems owns the generic Triton operator. This plugin owns vLLM's
        # checkpoint metadata and kernel-lifecycle integration.
        try:
            import flag_gems
        except ModuleNotFoundError as error:
            if error.name != "flag_gems":
                raise
            logger.warning(
                "[vllm_fl] FlagGems is not installed; ARM packed W4A8 "
                "integration is disabled and other vLLM paths are unchanged"
            )
            return
        if flag_gems.vendor_name != "arm":
            logger.warning(
                "[vllm_fl] FlagGems selected vendor %r, not 'arm'; ARM CPU "
                "runtime integration was not installed",
                flag_gems.vendor_name,
            )
            return

        from vllm_fl.quantization.arm_cpu_w4a8 import (
            install_arm_cpu_packed_w4a8,
        )

        install_arm_cpu_packed_w4a8()
        return

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

        # from vllm_fl.patches.glm_moe_dsa import apply_model_patches as glm5_model
        # glm5_model()
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

    # triton_unified_attention.py (vllm 0.24.0) references tl.make_tensor_descriptor
    # from module-level TD helper JITFunctions, but triton 3.2.0+mlu1.7.2
    # (cambricon 4.4.3) lacks the symbol. triton's dependency finder walks every
    # referenced JITFunction with an unconditional getattr, ignoring the
    # constexpr gate that keeps the TD path dead on MLU, so the symbol must
    # merely exist — inject a stub (never invoked) on forks that lack it.
    import triton.language as _tl

    if not hasattr(_tl, "make_tensor_descriptor"):
        _tl.make_tensor_descriptor = lambda *args, **kwargs: None
except ImportError:
    # No triton in this environment: there is no TD symbol to stub.
    pass
