"""MetaX patches applied into vLLM 0.28 at import time.

``PlatformFL.import_kernels()`` imports this package while vLLM is still
initialising, so a single failing module used to skip every patch below it.
Each patch is imported on its own and failures are recorded in
``PATCH_ERRORS`` instead of aborting the rest of the package.
"""

import importlib
import logging

import vllm.utils.import_utils as iu

logger = logging.getLogger(__name__)

# Order matters: pynccl_wrapper depends on utils_patch, and gdn_linear_attn
# must be imported before any model layer is instantiated.
_PATCH_MODULES = (
    "accelerator_compat",
    "functorch_config_patch",
    "fix_standalone_compile",
    "pynccl_wrapper",
    "cuda_wrapper",
    "utils_patch",
    "chunk_delta_h",
    "moe_topk_compat",
    "gdn_linear_attn",
)

# MetaX C550 does not support third-party Triton kernels (Triton upgrade
# required). Disable them so FLA decode ops (fused_recurrent_gated_delta_rule
# etc.) fall back to the non-Triton path handled by mcoplib, producing correct
# output.
#
# Assigned before any patch below is imported: vLLM modules bind this predicate
# with ``from vllm.utils.import_utils import has_triton_kernels`` when they
# load, so replacing it as early as possible keeps more of them on the fallback
# path.
# TODO: remove when MetaX Triton support is available.
iu.has_triton_kernels = lambda: False


def _import_patches() -> dict[str, Exception]:
    errors: dict[str, Exception] = {}
    for name in _PATCH_MODULES:
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
    return dict(PATCH_ERRORS)
