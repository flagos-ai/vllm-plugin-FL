# Copyright (c) 2026 BAAI. All rights reserved.

"""
MUSA-specific patches for vLLM compatibility.
"""

import logging

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_musa_patches():
    """Apply MUSA patches that must run before model construction."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_mccl_backend()
    patch_topk_topp_sampler()
    patch_sync_device()
    patch_triton_reshape_and_cache_flash()
    patch_cuda_get_device_properties()


def patch_mccl_backend():
    """Register the mccl distributed backend for MUSA via torch_musa."""
    try:
        import torch_musa.distributed as musa_dist
        musa_dist._apply_distributed_patch()
        logger.info("Registered mccl distributed backend for MUSA")
    except Exception as e:
        logger.warning("Failed to register mccl backend for MUSA: %s", e)


def patch_topk_topp_sampler():
    """Force PyTorch-native top-k/top-p on MUSA.

    The vLLM Triton top-k/top-p kernel uses mixed uint32/int32 arithmetic
    that the MUSA Triton compiler rejects. Route through the PyTorch path
    instead, which works correctly on MUSA.
    """
    try:
        import vllm.v1.sample.ops.topk_topp_sampler as sampler_mod
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

        if getattr(sampler_mod, "_musa_topk_topp_patched", False):
            return

        def _apply_top_k_top_p_musa(logits, k, p):
            return apply_top_k_top_p_pytorch(logits, k, p)

        sampler_mod.apply_top_k_top_p = _apply_top_k_top_p_musa
        sampler_mod._musa_topk_topp_patched = True
        logger.info("Patched apply_top_k_top_p to use PyTorch-native path for MUSA")
    except Exception as e:
        # May fail in the main process due to circular imports during early init;
        # worker processes will retry and succeed independently.
        logger.debug("Failed to patch top-k/top-p sampler for MUSA: %s", e)


def patch_sync_device():
    """Patch _sync_device in ModelRunnerFL to use torch_musa.synchronize().

    torch.accelerator.synchronize() is not supported on MUSA; use the
    MUSA-native synchronize instead.
    """
    try:
        import torch_musa

        if getattr(patch_sync_device, "_musa_sync_device_patched", False):
            return

        def _sync_device_musa(self):
            torch_musa.synchronize()

        # Defer the import of model_runner to avoid circular imports during
        # early module initialization (fused_moe.modular_kernel is not yet
        # fully loaded when apply_musa_patches() is first called).
        import vllm_fl.worker.model_runner as mr_mod
        mr_mod.ModelRunnerFL._sync_device = _sync_device_musa
        patch_sync_device._musa_sync_device_patched = True
        logger.info("Patched ModelRunnerFL._sync_device to use torch_musa.synchronize()")
    except Exception as e:
        # May fail during early init due to circular imports; will be retried
        # in worker processes once all modules are fully initialized.
        logger.debug("Failed to patch _sync_device for MUSA: %s", e)


def patch_triton_reshape_and_cache_flash():
    """Patch triton_reshape_and_cache_flash to avoid torch.cuda calls on MUSA.

    The function calls torch.cuda.get_device_capability() unconditionally in
    the else branch, which fails on MUSA devices. Patch torch.cuda to handle
    MUSA devices gracefully by returning a safe capability value.
    """
    try:
        import torch.cuda as torch_cuda

        if getattr(torch_cuda, "_musa_get_device_capability_patched", False):
            return

        _orig_get_device_capability = torch_cuda.get_device_capability

        def _get_device_capability_musa(device=None):
            try:
                return _orig_get_device_capability(device)
            except (ValueError, RuntimeError):
                # MUSA device: return a safe capability that avoids fp8 paths
                return (8, 0)

        torch_cuda.get_device_capability = _get_device_capability_musa
        torch_cuda._musa_get_device_capability_patched = True
        logger.info("Patched torch.cuda.get_device_capability for MUSA")
    except Exception as e:
        logger.warning("Failed to patch torch.cuda.get_device_capability for MUSA: %s", e)


def patch_cuda_get_device_properties():
    """Patch vllm.utils.platform_utils.cuda_get_device_properties for MUSA.

    The original implementation spawns a subprocess via ProcessPoolExecutor
    when CUDA is not initialized. On MUSA this always triggers the subprocess
    path, which fails with AssertionError when called from a daemon thread
    (e.g. vllm's usage-reporting thread). Replace it with a direct
    torch_musa call so no subprocess is needed.
    """
    try:
        import torch_musa
        import vllm.utils.platform_utils as pu_mod

        if getattr(pu_mod, "_musa_cuda_get_device_properties_patched", False):
            return

        def _cuda_get_device_properties_musa(device, names, init_cuda=False):
            props = torch_musa.get_device_properties(device)
            return tuple(getattr(props, name) for name in names)

        pu_mod.cuda_get_device_properties = _cuda_get_device_properties_musa
        # Also patch the reference already imported into usage_lib.
        try:
            import vllm.usage.usage_lib as ul_mod
            ul_mod.cuda_get_device_properties = _cuda_get_device_properties_musa
        except Exception:
            pass
        pu_mod._musa_cuda_get_device_properties_patched = True
        logger.info("Patched cuda_get_device_properties to use torch_musa for MUSA")
    except Exception as e:
        logger.warning("Failed to patch cuda_get_device_properties for MUSA: %s", e)
