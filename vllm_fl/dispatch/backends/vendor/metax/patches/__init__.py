"""MetaX patches installed during vLLM import and before model creation.

``PlatformFL.import_kernels()`` imports this package while vLLM is still
initialising. Low-level patches are installed immediately; patches that import
model layers are deferred to ``ensure_patches()`` to avoid circular imports.
Failures remain recorded in ``PATCH_ERRORS`` for reporting and retry.
"""

import importlib
import logging
import os

import vllm.utils.import_utils as iu

from .._compat import legacy_torch_runtime

logger = logging.getLogger(__name__)

# Order matters: pynccl_wrapper depends on utils_patch.
_PATCH_MODULES = (
    "accelerator_compat",
    "functorch_config_patch",
    "fix_standalone_compile",
    "utils_patch",
    "pynccl_wrapper",
    "cuda_wrapper",
    "chunk_delta_h",
    "moe_topk_compat",
)

# These patches import model layers and must wait until vLLM has finished
# importing its core modules, but run before any model layer is instantiated.
_DEFERRED_PATCH_MODULES = ("gdn_linear_attn",)
_V41_PATCH_MODULES = (
    "compressed_tensors_moe_targets",
    "w4a8_moe",
    "deepseek_v41.ep_router_ids",
    "sqrtsoftplus_router",
    "deepseek_v41.graph",
)


def _install_v41_patches():
    """Prepare the complete dependent group, then install or roll it back."""
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "metax":
        return
    bindings = []
    for name in _V41_PATCH_MODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        bindings.extend(module.prepare())
    originals = [(target, name, getattr(target, name)) for target, name, _ in bindings]
    try:
        for target, name, replacement in bindings:
            setattr(target, name, replacement)
    except Exception:
        for target, name, original in reversed(originals):
            setattr(target, name, original)
        raise


# Keep the optional third-party triton_kernels package disabled, matching
# vLLM-metax 0.32. vLLM's bundled FLA/Triton kernels remain available in the
# pure FlagOS runtime; this does not require mcoplib to be installed.
#
# Assigned before any patch below is imported: vLLM modules bind this predicate
# with ``from vllm.utils.import_utils import has_triton_kernels`` when they
# load, so replacing it as early as possible keeps more of them on the fallback
# path.
# Remove when the optional package is validated on MetaX.
# vLLM-metax 0.32 still disables this third-party package on Triton 3.6.
# This restriction is independent of the old Triton 3.0 compiler workarounds.
iu.has_triton_kernels = lambda: False


def _import_patches() -> dict[str, Exception]:
    errors: dict[str, Exception] = {}
    for name in _PATCH_MODULES:
        if (
            name == "gdn_linear_attn"
            and os.environ.get("MACA_VLLM_USE_EXTERNAL_FLA") != "1"
        ):
            continue
        if (
            name in ("fix_standalone_compile", "functorch_config_patch")
            and not legacy_torch_runtime()
        ):
            # These replace private torch 2.8 APIs and the matmul class's
            # __setattr__; importing them on torch 2.10 would undo native fixes.
            continue
        try:
            importlib.import_module(f"{__name__}.{name}")
        except Exception as exc:  # keep the remaining patches alive
            errors[name] = exc
            logger.warning("metax patch %s failed to apply", name, exc_info=True)
    return errors


# Patch module name -> exception raised while importing it. Empty on a healthy
# MetaX run; read by tests/unit_tests/dispatch/test_metax_patches.py.
PATCH_ERRORS: dict[str, Exception] = _import_patches()


def ensure_patches() -> dict[str, Exception]:
    """Install deferred patches and retry earlier failures before model creation.

    The platform configuration hook and worker runtime call this after vLLM
    core imports have completed. Successful imports are cached by Python, so
    repeated calls do not register the GDN replacement more than once. Failed
    imports remain in ``PATCH_ERRORS`` and are retried on the next call.
    """
    for name in dict.fromkeys((*PATCH_ERRORS, *_DEFERRED_PATCH_MODULES)):
        if name == "deepseek_v41":
            continue
        try:
            importlib.import_module(f"{__name__}.{name}")
        except Exception as exc:
            PATCH_ERRORS[name] = exc
            continue
        PATCH_ERRORS.pop(name, None)
    # _custom_ops invokes import_kernels() before defining its top-k helpers.
    # A successful early import can therefore be overwritten without raising
    # an exception. Restore this patch after the native module finishes loading.
    try:
        topk_compat = importlib.import_module(f"{__name__}.moe_topk_compat")
        if topk_compat.apply():
            PATCH_ERRORS.pop("moe_topk_compat", None)
        else:
            PATCH_ERRORS["moe_topk_compat"] = RuntimeError(
                "MetaX routed top-k compatibility could not be applied"
            )
    except Exception as exc:
        PATCH_ERRORS["moe_topk_compat"] = exc
    try:
        _install_v41_patches()
    except Exception as exc:
        PATCH_ERRORS["deepseek_v41"] = exc
    else:
        PATCH_ERRORS.pop("deepseek_v41", None)
    return dict(PATCH_ERRORS)
