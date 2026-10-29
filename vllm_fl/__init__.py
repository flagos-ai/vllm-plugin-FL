# Copyright (c) 2025 BAAI. All rights reserved.

import importlib
import logging
import os
import sys


def _patch_flag_gems_triton_import_compat():
    """Allow newer FlagGems to load with the Kunlunxin Triton runtime.

    FlagGems 5.4 registers ``_dirichlet_grad`` at import time and asks Triton
    to resolve ``tl.map_elementwise`` while computing the JIT cache key.  The
    Kunlunxin Triton runtime does not expose that builtin.  vLLM does not use
    this operator and the Kunlunxin dispatch config blacklists it, so provide
    only an import-time sentinel.  If it is ever invoked, fail explicitly
    instead of silently producing an incorrect result.
    """
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
        stub.fa_version_unsupported_reason = lambda *a, **kw: (
            "flash_attn C extensions not available"
        )
        stub.flash_attn_varlen_func = None
        stub.get_scheduler_metadata = None
        stub.is_fa_version_supported = lambda *a, **kw: False
        sys.modules["vllm.vllm_flash_attn"] = stub


def _is_metax_backend() -> bool:
    """True when this process runs on the MetaX (maca) backend."""
    from vllm_fl.ops._C_ops_registry import _is_metax_backend as _impl

    return _impl()


def _metax_plugin_installed() -> bool:
    """True when vllm_metax (the vendor platform plugin) is importable."""
    import importlib.util

    try:
        return importlib.util.find_spec("vllm_metax") is not None
    except Exception:
        return False


def _patch_custom_ops():
    """Register fallback schemas when neither vLLM extension ABI is present."""
    # On MetaX the maca mcoplib._C already registered every _C operator.
    # Importing vLLM's own vllm._C .so on top of it raises a
    # duplicate-registration c10::Error, so skip the probe entirely.
    if _is_metax_backend():
        logger.debug(
            "[vllm_fl] MetaX backend owns torch.ops._C; skipping vllm._C import"
        )
        return
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


def _install_metax_compat():
    """Apply the MetaX vLLM 0.28 compatibility shims before any kernel runs.

    These shims used to be installed as ``.pth`` hooks in ``site-packages`` so
    that they ran at interpreter start-up. Every one of them either arms a
    ``sys.meta_path`` hook for a lazily imported module or rebinds a module
    that is already imported, so the plugin entry point runs early enough and
    a plain ``pip install`` is sufficient.

    This runs before the platform-slot decision below: under
    ``VLLM_PLUGINS=metax`` FL yields the platform slot and
    ``PlatformFL.import_kernels`` is never called, yet the vendor modules
    still need these fixes.
    """
    try:
        from vllm_fl.patches.metax_028_compat import install, metax_active

        if metax_active():
            install()
    except Exception as exc:  # pragma: no cover - never block registration
        logger.warning("[vllm_fl] MetaX compatibility shims not installed: %s", exc)


def register():
    """Register the FL platform."""
    _install_metax_compat()
    # MetaX (maca) ships its own vLLM platform plugin (vllm_metax) and vLLM
    # 0.28 rejects more than one platform plugin
    # (RuntimeError: Only one platform plugin can be activated, but got
    # ['metax', 'fl']).  On that backend FL must not claim the platform slot:
    # the general plugin (vllm_fl:register_model) keeps FL's model and kernel
    # registrations.
    # Only yield the platform slot when the user actually asked for the
    # vendor plugin (VLLM_PLUGINS allows "metax" and not "fl") and it is
    # installed.  Keying this on the *hardware* instead would leave a MetaX
    # host on UnspecifiedPlatform whenever VLLM_PLUGINS=fl: vLLM 0.28 would
    # then activate neither plugin and no builtin plugin matches MACA.
    _allowed = [
        p.strip()
        for p in (os.environ.get("VLLM_PLUGINS") or "").split(",")
        if p.strip()
    ]
    if "metax" in _allowed and "fl" not in _allowed and _metax_plugin_installed():
        logger.info(
            "[vllm_fl] VLLM_PLUGINS requests metax -> vLLM metax platform; "
            "FL platform registration skipped"
        )
        return None

    _init_vendor_device()

    # PlatformFL is accelerator-shaped. For the standard FlagGems ARM target,
    # preserve vLLM's stock CPU platform and install kernels in register_model().
    arm_cpu_platform = _arm_cpu_platform()
    if arm_cpu_platform is not None:
        logger.info("[vllm_fl] ARM64 CPU target -> vLLM CPU platform")
        return arm_cpu_platform

    _patch_custom_ops()
    _patch_flash_attn_import()

    # Note: FlagCX connector registration is deferred to register_model()
    # to avoid circular imports during VllmConfig.__post_init__ in spawned
    # subprocesses.

    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _get_op_config()

    from vllm_fl.utils import DeviceInfo

    vendor_name = DeviceInfo().vendor_name
    if vendor_name in ("ascend", "enflame", "gcu"):
        # v0.28 defaults some models to the upstream V2 runner. These backends
        # use ModelRunnerFL; choose it before VllmConfig reads model defaults.
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    if vendor_name == "nvidia":
        return "vllm_fl.nvidia_platform.NvidiaPlatformFL"

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
    """Register FL runtime extensions after vLLM model discovery."""

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

    from vllm_fl.utils import DeviceInfo

    if DeviceInfo().vendor_name == "ascend":
        # Import registers the ModelSlim quantization config before model
        # configuration resolution in worker processes.
        from vllm.model_executor.models import ModelRegistry

        from vllm_fl.quantization import modelslim_w8a8  # noqa: F401

        # Reuse vLLM's DeepSeek-V4 model and loader with an Ascend-only
        # attention/cache adapter for the torch_npu empty build.
        ModelRegistry.register_model(
            "DeepseekV4ForCausalLM",
            "vllm_fl.models.deepseek_v4:DeepseekV4ForCausalLM",
        )
        ModelRegistry.register_model(
            "GlmMoeDsaForCausalLM",
            "vllm_fl.models.glm_moe_dsa:GlmMoeDsaForCausalLM",
        )
