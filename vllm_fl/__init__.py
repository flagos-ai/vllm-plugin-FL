# Copyright (c) 2025 BAAI. All rights reserved.

import importlib
import logging
import os
import sys


def _get_explicit_vendor_for_triton_compat():
    """Read an early vendor selection without importing vLLM or FlagGems."""
    platform = os.environ.get("VLLM_FL_PLATFORM", "").strip().lower()
    # ``cuda`` is a device type shared by NVIDIA and Kunlunxin, so it cannot
    # decide whether the Kunlunxin compatibility patch is needed.
    if platform and platform not in {"auto", "cuda"}:
        return platform

    vendor = os.environ.get("GEMS_VENDOR", "").strip().lower()
    return vendor or None


def _should_patch_flag_gems_triton_import_compat():
    """Keep auto detection, but respect an explicitly selected vendor."""
    vendor = _get_explicit_vendor_for_triton_compat()
    return vendor is None or vendor == "kunlunxin"


def _patch_flag_gems_triton_import_compat():
    """Allow newer FlagGems to load with the Kunlunxin Triton runtime.

    The hook runs before vLLM platform registration, so it must only use
    environment variables for early vendor selection.  When no vendor is
    explicit, retain the existing probe for Kunlunxin auto detection.

    FlagGems 5.4 registers ``_dirichlet_grad`` at import time and asks Triton
    to resolve ``tl.map_elementwise`` while computing the JIT cache key.  The
    Kunlunxin Triton runtime does not expose that builtin.  vLLM does not use
    this operator and the Kunlunxin dispatch config blacklists it, so provide
    only an import-time sentinel.  If it is ever invoked, fail explicitly
    instead of silently producing an incorrect result.
    """
    if not _should_patch_flag_gems_triton_import_compat():
        return

    try:
        import triton
        import triton.language as tl
    except ImportError:
        return
    if not hasattr(tl, "map_elementwise"):
        def _unsupported_map_elementwise(*args, **kwargs):
            raise NotImplementedError(
                "triton.language.map_elementwise is unavailable on Kunlunxin; "
                "the FlagGems _dirichlet_grad operator must remain blacklisted"
            )

        _unsupported_map_elementwise.__name__ = "map_elementwise"
        _unsupported_map_elementwise.__module__ = "triton.language"
        _unsupported_map_elementwise.__triton_builtin__ = True
        tl.map_elementwise = _unsupported_map_elementwise

    try:
        importlib.import_module("triton.knobs")
    except ModuleNotFoundError as exc:
        if exc.name != "triton.knobs":
            raise
        import types

        knobs = types.ModuleType("triton.knobs")
        knobs.autotuning = types.SimpleNamespace(adjust_block_size=True)
        sys.modules[knobs.__name__] = knobs
        triton.knobs = knobs


_patch_flag_gems_triton_import_compat()

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

from . import version as version  # PyTorch-style: vllm_fl.version.git_version
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

        # ``vllm_flash_attn.__init__`` imports ``flash_attn_interface`` before
        # checking whether the CUDA FA extensions are available.  When that
        # final check raises, Python removes the parent package but leaves the
        # successfully imported interface module cached.  Reusing that orphan
        # later makes its missing relative C extension look like a circular
        # import and emits one error per model layer.  Drop the failed probe's
        # child before installing the non-CUDA fallback package.
        sys.modules.pop("vllm.vllm_flash_attn.flash_attn_interface", None)
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


def _init_vendor_device():
    """Apply compatibility hooks that must run before vLLM model imports."""
    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().vendor_name == "kunlunxin":
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils import (
            _patch_xpu_get_device,
        )

        _patch_xpu_get_device()


def register():
    """Register the FL platform."""
    _init_vendor_device()

    # PlatformFL is accelerator-shaped. For the standard FlagGems ARM target,
    # preserve vLLM's stock CPU platform and install kernels in register_model().
    arm_cpu_platform = _arm_cpu_platform()
    if arm_cpu_platform is not None:
        logger.info("[vllm_fl] ARM64 CPU target -> vLLM CPU platform")
        return arm_cpu_platform

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
    # at module level, which requires torch.ops._C — not available on these
    # platforms.
    if current_platform.device_type in {"musa", "gcu"}:
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
    from vllm_fl.patches.deepseek_v4_thead import apply_deepseek_v4_thead_patches
    from vllm_fl.patches.moe_sum import patch_vllm_moe_sum
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

    patch_vllm_moe_sum()
    apply_deepseek_v4_thead_patches()

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
