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

try:
    from vllm_fl.utils import get_op_config as _get_op_config
except ModuleNotFoundError as exc:
    # The native GLM5-Next baseline reuses only this package's config/model
    # modules and deliberately does not install or activate FlagGems.
    if exc.name != "flag_gems":
        raise

    def _get_op_config():
        return None


from . import version as version  # PyTorch-style: vllm_fl.version.git_version

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

    vendor_name = DeviceInfo().vendor_name
    if vendor_name == "nvidia":
        # FlashInfer chooses the first loaded libcudart by name. Bind it to
        # Torch's CUDA runtime before optional TileLang probes load a stub
        # with the same prefix but an incomplete CUDA runtime ABI.
        try:
            importlib.import_module("flashinfer.comm")
        except (ImportError, OSError) as error:
            logger.debug("Optional FlashInfer communication is unavailable: %s", error)
    elif vendor_name == "kunlunxin":
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


def _patch_ascend_torch_accelerator() -> None:
    """Install the Ascend memory shim in every general-plugin process."""
    from vllm.platforms import current_platform

    if (
        current_platform.device_type == "npu"
        and getattr(current_platform, "vendor_name", None) == "ascend"
    ):
        from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_torch_accelerator import (
            patch_torch_accelerator,
        )

        patch_torch_accelerator()


def _patch_multi_stream_capture_guard():
    """Plugin-side capture guard for both multi-stream helpers (Path C fix).

    aux streams 的 fork/join 事件在 CUDA graph 捕获期间会让下游 MoE 的首调/
    tuner 异常并静默走未融合路径, 退化形态被冻进图里永久复放 (2026-09-11
    T2/T3/T5 对照实验)。此守卫只约束捕获期, 不保证运行期多流正确性。
    同时覆盖 R128 compressor 使用的 maybe_execute_in_parallel。替代对
    vllm/utils/multi_stream_utils.py 的源码修改 —— 插件内完成, 不动 vllm。
    VLLM_MULTI_STREAM_DURING_CAPTURE=1 可关掉守卫(实验用)。
    """
    if getattr(_patch_multi_stream_capture_guard, "_installed", False):
        return
    import torch
    from vllm.utils import multi_stream_utils

    _orig = multi_stream_utils.execute_in_parallel
    _orig_maybe = multi_stream_utils.maybe_execute_in_parallel

    def _disable_capture_aux():
        return (
            torch.cuda.is_current_stream_capturing()
            and os.environ.get("VLLM_MULTI_STREAM_DURING_CAPTURE", "0") != "1"
        )

    def _guarded(default_fn, aux_fns, start_event, done_events, aux_streams=None, enable=False):
        if (
            aux_streams is not None
            and enable
            and _disable_capture_aux()
        ):
            enable = False
        return _orig(
            default_fn, aux_fns, start_event, done_events,
            aux_streams=aux_streams, enable=enable,
        )

    def _guarded_maybe(fn0, fn1, event0, event1, aux_stream=None):
        if aux_stream is not None and _disable_capture_aux():
            aux_stream = None
        return _orig_maybe(fn0, fn1, event0, event1, aux_stream)

    multi_stream_utils.execute_in_parallel = _guarded
    multi_stream_utils.maybe_execute_in_parallel = _guarded_maybe
    _patch_multi_stream_capture_guard._installed = True
    logger.info("Patched both multi-stream helpers with CUDA graph capture guard")


def register_model():
    """Register FL-specific models not yet upstream."""
    _patch_ascend_torch_accelerator()

    # General plugins are loaded independently in spawned model-inspection and
    # worker processes, so all runtime compatibility hooks must be idempotent.
    from vllm_fl.models.glm5.runtime import (
        register_glm5_next_support,
    )
    from vllm_fl.patches.qwen3_5_text import apply_qwen3_5_text_patches
    from vllm_fl.patches.qwen3_8_flash_next import (
        apply_qwen3_8_flash_next_patches,
    )

    # GLM5-Next is intentionally registered only by its vLLM-0.24 adapter;
    # the plugin main branch is ABI-pinned to that release line.
    register_glm5_next_support()
    apply_qwen3_5_text_patches()
    apply_qwen3_8_flash_next_patches()

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

    from vllm_fl.models.hy4.registration import register_hy_v4_support

    register_hy_v4_support()

    from vllm_fl.models.minimax_m3_registration import register_minimax_m3_support

    register_minimax_m3_support()

    _register_flagcx_connector()

    # Register OOT quant kernels so kernel selection can find them
    register_quant_linear()
    register_router()

    _register_gdn_packed_decode_patch()

    # DeepSeek-V4 is available upstream, but Hygon needs a compile-friendly
    # orchestration layer around its validated ROCm attention and MHC kernels.
    # Keep the registration lazy and vendor-scoped so other FL platforms
    # continue to use the upstream architecture unchanged.
    from vllm import ModelRegistry
    from vllm.platforms import current_platform

    use_hygon_dv4_model = os.environ.get(
        "VLLM_FL_HYGON_DV4_COMPILE_MODEL", "1"
    ).lower() not in {"0", "false", "off"}
    if (
        str(getattr(current_platform, "vendor_name", "")).lower() == "hygon"
        and use_hygon_dv4_model
    ):
        ModelRegistry.register_model(
            "DeepseekV4ForCausalLM",
            "vllm_fl.models.hygon.deepseek_v4:DeepseekV4ForCausalLM",
        )
        _patch_multi_stream_capture_guard()
        logger.info("Registered Hygon compile-friendly DeepseekV4ForCausalLM")

    # Register GLM-5 (GlmMoeDsa) — config not yet upstream
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY

        from vllm_fl.configs.glm_moe_dsa import GlmMoeDsaConfig

        _CONFIG_REGISTRY["glm_moe_dsa"] = GlmMoeDsaConfig

        # from vllm_fl.patches.glm_moe_dsa import apply_model_patches as glm5_model
        # glm5_model()
    except Exception as e:
        logger.error(f"Register GlmMoeDsa model error: {str(e)}")

    if os.environ.get("VLLM_FL_HYGON_FP8_HCU") == "1":
        from vllm_fl.dispatch.backends.vendor.hygon.patches.fp8_hcu import install
        install()
