# Copyright (c) 2025 BAAI. All rights reserved.

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

from . import version as version  # PyTorch-style: vllm_fl.version.git_version
from vllm_fl.utils import get_op_config as _get_op_config

logger = logging.getLogger(__name__)

# vLLM 0.28 relocated the Flash-Linear-Attention kernels from
# ``vllm.model_executor.layers.fla`` to
# ``vllm.third_party.flash_linear_attention``.  Alias the old spelling onto
# the new one so the plugin and vendor backends keep working verbatim on
# 0.28 (and stay a no-op on <= 0.24, where the real package exists).
def _install_xpu_device_guard() -> None:
    """Install the torch.xpu.get_device_name wrapper BEFORE any FLA import.

    ``_install_fla_relocation_alias()`` probes
    ``vllm.model_executor.layers.fla``, which on 0.28 pulls in the FLA ops
    package whose module body calls ``torch.xpu.get_device_name(0)``.  On
    Kunlunxin that raises "Torch not compiled with XPU enabled" and aborts the
    alias install, so the wrapper must already be in place.  Idempotent and a
    no-op when ``torch.xpu`` is unavailable.
    """
    try:
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils import (  # noqa: E501
            _install_xpu_get_device_wrapper,
        )

        _install_xpu_get_device_wrapper()
    except Exception as _xpu_guard_exc:  # pragma: no cover - vendor dependent
        logger.debug("[vllm_fl] XPU device guard unavailable: %s", _xpu_guard_exc)


def _install_fla_relocation_alias() -> None:
    """Alias the pre-0.28 FLA path onto the 0.28 layout (deferred to register).

    Probing ``vllm.model_executor.layers.fla`` imports the parent
    ``vllm.model_executor.layers`` package, which resolves ``current_platform``
    and re-enters the platform-plugin loader while this module is only
    partially initialised (``vllm_fl.register`` not yet bound), so vLLM drops
    the platform plugin.  Deferring to ``register()`` keeps the alias in place
    before any vendor/model code imports it.
    """
    try:
        from vllm_fl.patches.fla_module_relocation import install as _install

        _install()
    except Exception as _fla_reloc_exc:  # pragma: no cover - version dependent
        logger.debug("[vllm_fl] FLA module relocation unavailable: %s", _fla_reloc_exc)


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


_register_in_progress = False

# vLLM resolves ``vllm.platforms.current_platform`` from this qualname, so the
# value lives in one place: register() must always hand vLLM a non-None string.
_PLATFORM_QUALNAME = "vllm_fl.platform.PlatformFL"


def _preload_platform_class():
    """Import PlatformFL early so a nested resolution cannot fail mid-way.

    The class body resolves the vendor/device_type from the FlagGems detector.
    Importing it *inside* register() before the side effects below means a
    re-entrant resolve (which imports ``vllm.model_executor.layers`` and
    therefore ``vllm.platforms``) can instantiate it immediately instead of
    failing while this module is still executing.
    """
    try:
        from vllm_fl.platform import PlatformFL

        return PlatformFL
    except Exception as _platform_exc:  # pragma: no cover - vendor dependent
        logger.debug("[vllm_fl] PlatformFL unavailable: %s", _platform_exc)
        return None


def _run_best_effort(*steps) -> None:
    """Run compatibility hooks without ever letting one abort registration."""
    for step in steps:
        try:
            step()
        except Exception as _step_exc:  # pragma: no cover - vendor dependent
            logger.warning("[vllm_fl] %s failed: %s", step.__name__, _step_exc)


def _is_fallback_platform(platform) -> bool:
    """True when the cached platform cannot serve an accelerator.

    The class-name check is kept as a fallback because the import path of
    ``UnspecifiedPlatform`` has moved between vLLM releases, and a platform
    that reports an empty ``device_type`` is unusable whatever it is called.
    """
    try:
        from vllm.platforms.interface import UnspecifiedPlatform

        if isinstance(platform, UnspecifiedPlatform):
            return True
    except Exception as _iface_exc:  # pragma: no cover - layout dependent
        logger.debug("[vllm_fl] UnspecifiedPlatform unavailable: %s", _iface_exc)
    if type(platform).__name__ == "UnspecifiedPlatform":
        return True
    return not getattr(platform, "device_type", "")


def _force_adopt_platform() -> bool:
    """Re-assert PlatformFL when vLLM already cached a fallback platform.

    ``vllm.platforms`` resolves ``current_platform`` lazily but **only once**:
    its module-level ``__getattr__`` stores the first result in the global
    ``_current_platform``.  That resolution is re-entrant, and the *outer*
    resolution wins because it performs the final assignment.  So if any step of
    the outer ``register()`` raises, vLLM wraps the plugin callable in
    ``except Exception: pass``, counts this plugin as "not activated", and
    overwrites the good platform a nested resolution already installed with
    ``UnspecifiedPlatform``.

    Reading the module global ``_current_platform`` (not the ``current_platform``
    attribute) deliberately avoids triggering yet another nested resolution,
    and the assignment goes through vLLM's module ``__setattr__``, which is the
    supported way for a plugin to set the platform.
    """
    try:
        import vllm.platforms as _platforms
    except Exception as _platform_exc:  # pragma: no cover - version dependent
        logger.debug("[vllm_fl] cannot inspect current_platform: %s", _platform_exc)
        return False

    current = getattr(_platforms, "_current_platform", None)
    if current is None or not _is_fallback_platform(current):
        # Nothing resolved yet (vLLM will call register() again), or a real
        # platform is already in place.
        return False

    _platform_cls = _preload_platform_class()
    if _platform_cls is None:
        return False

    _platforms.current_platform = _platform_cls()
    logger.info(
        "[vllm_fl] vLLM had cached %s; re-adopted %s",
        type(current).__name__,
        _PLATFORM_QUALNAME,
    )
    return True


def register():
    """Register the FL platform.

    Must never raise and never return ``None``: vLLM 0.28 calls this entry
    point from ``resolve_current_platform_cls_qualname()`` inside a bare
    ``except Exception: pass``, so an escaping exception surfaces not as a
    traceback but as ``current_platform`` silently pinned to
    ``UnspecifiedPlatform`` for the life of the process.  Every optional patch
    is therefore best-effort -- a patch module that only ships on some plugin
    revisions must degrade to a debug log, not abort platform registration.
    """
    global _register_in_progress
    if _register_in_progress:
        # Re-entrant call while the outer register() is still installing
        # aliases / vendor hooks (vLLM loads platform plugins more than once,
        # and register() itself can trigger a nested platform resolution).
        return _PLATFORM_QUALNAME
    _register_in_progress = True
    try:
        _preload_platform_class()
        _run_best_effort(
            _install_xpu_device_guard,
            _install_fla_relocation_alias,
            _init_vendor_device,
        )

        # PlatformFL is accelerator-shaped. For the standard FlagGems ARM target,
        # preserve vLLM's stock CPU platform and install kernels in register_model().
        try:
            arm_cpu_platform = _arm_cpu_platform()
        except Exception as _arm_exc:  # pragma: no cover - version dependent
            logger.debug("[vllm_fl] ARM CPU platform probe unavailable: %s", _arm_exc)
            arm_cpu_platform = None
        if arm_cpu_platform is not None:
            logger.info("[vllm_fl] ARM64 CPU target -> vLLM CPU platform")
            return arm_cpu_platform

        _run_best_effort(
            _patch_custom_ops,
            _patch_flash_attn_import,
            _patch_transformers_compat,
        )

        # Model-specific platform patches. The patch module only exists on
        # plugin revisions that ship GLM-5/DSA support; a missing module means
        # "nothing to patch", not "the platform is broken".
        try:
            from vllm_fl.patches.glm_moe_dsa import (
                apply_platform_patches as glm5_platform,
            )

            glm5_platform()
        except ImportError as _glm_exc:
            logger.debug("[vllm_fl] GLM-5 platform patches unavailable: %s", _glm_exc)
        except Exception as _glm_exc:  # pragma: no cover - vendor dependent
            logger.warning("[vllm_fl] GLM-5 platform patches failed: %s", _glm_exc)

        # Note: FlagCX connector registration is deferred to register_model()
        # to avoid circular imports during VllmConfig.__post_init__ in spawned
        # subprocesses.

        multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
        if multiproc_method is None:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        _run_best_effort(_get_op_config)

        _force_adopt_platform()
        return _PLATFORM_QUALNAME
    except Exception as _register_exc:  # pragma: no cover - defensive
        logger.warning(
            "[vllm_fl] register() failed, still reporting the platform: %s",
            _register_exc,
        )
        return _PLATFORM_QUALNAME
    finally:
        _register_in_progress = False

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

        #from vllm_fl.patches.glm_moe_dsa import apply_model_patches as glm5_model
        #glm5_model()
    except Exception as e:
        logger.error(f"Register GlmMoeDsa model error: {str(e)}")
