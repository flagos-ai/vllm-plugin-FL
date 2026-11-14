"""MetaX patches applied into vLLM 0.32 at import time.

``PlatformFL.import_kernels()`` imports this package while vLLM is still
initialising, so a single failing module used to skip every patch below it.
Each patch is imported on its own and failures are recorded in
``PATCH_ERRORS`` instead of aborting the rest of the package.
"""

import importlib
import logging
import os

import vllm.utils.import_utils as iu

from .._compat import legacy_torch_runtime

logger = logging.getLogger(__name__)

# Order matters: pynccl_wrapper depends on utils_patch, and gdn_linear_attn
# must be imported before any model layer is instantiated.
_PATCH_MODULES = (
    "accelerator_compat",
    "functorch_config_patch",
    "fix_standalone_compile",
    "utils_patch",
    "pynccl_wrapper",
    "cuda_wrapper",
    "chunk_delta_h",
    "moe_topk_compat",
    "gdn_linear_attn",
)

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
    """Retry the patch modules that failed to import earlier.

    ``import_kernels()`` runs while vLLM may still be importing itself, so a
    patch that needs a model-layer module (``gdn_linear_attn`` imports the
    mamba/attention layers) can hit a circular import. Python drops a module
    whose import raised, so importing it again once vLLM is fully loaded
    succeeds. ``PlatformFL.apply_config_platform_defaults()`` calls this before
    any model layer is built.
    """
    for name in list(PATCH_ERRORS):
        try:
            importlib.import_module(f"{__name__}.{name}")
        except Exception as exc:
            PATCH_ERRORS[name] = exc
            continue
        del PATCH_ERRORS[name]
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
    return dict(PATCH_ERRORS)
