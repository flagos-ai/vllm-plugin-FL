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

    patch_topk_topp_sampler()
    patch_triton_reshape_and_cache_flash()
    patch_cuda_get_device_properties()
    patch_accelerator_missing_attrs()
    patch_cuda_stream_for_musa()


def patch_topk_topp_sampler():
    """Force PyTorch-native top-k/top-p on MUSA.

    The vLLM Triton top-k/top-p kernel uses mixed uint32/int32 arithmetic
    that the MUSA Triton compiler rejects. Route through the PyTorch path
    instead, which works correctly on MUSA.
    """
    try:
        import vllm.v1.sample.ops.topk_topp_sampler as sampler_mod
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

        # Direct assignment works: apply_top_k_top_p_pytorch accepts the same
        # (logits, k, p) positional args; its extra allow_cpu_sync=False is defaulted.
        sampler_mod.apply_top_k_top_p = apply_top_k_top_p_pytorch
        logger.info("Patched apply_top_k_top_p to use PyTorch-native path for MUSA")
    except Exception as e:
        # May fail in the main process due to circular imports during early init;
        # worker processes will retry and succeed independently.
        logger.debug("Failed to patch top-k/top-p sampler for MUSA: %s", e)



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


def patch_accelerator_missing_attrs():
    """Patch missing torch.accelerator attributes for MUSA compatibility.

    Some vLLM modules call APIs that were added to torch.accelerator in newer
    PyTorch versions but are absent on the MUSA build:

    - ``torch.accelerator.empty_cache()`` — called by gdn_linear_attn.py after
      prefill kernel warmup. Delegated to torch_musa.empty_cache().

    - ``torch.accelerator.device_index(index)`` — used as a context manager in
      fla/ops/utils.py to pin operations to a specific device. The MUSA
      equivalent is ``torch_musa.device(index)``.
    """
    try:
        import torch
        import torch_musa

        if getattr(torch.accelerator, "_musa_attrs_patched", False):
            return

        if not hasattr(torch.accelerator, 'empty_cache'):
            torch.accelerator.empty_cache = torch_musa.empty_cache
            logger.info("Patched torch.accelerator.empty_cache for MUSA")

        if not hasattr(torch.accelerator, 'device_index'):
            torch.accelerator.device_index = torch_musa.device
            logger.info("Patched torch.accelerator.device_index for MUSA")

        torch.accelerator._musa_attrs_patched = True
    except Exception as e:
        logger.warning("Failed to patch torch.accelerator attrs for MUSA: %s", e)


def patch_cuda_stream_for_musa():
    """Patch torch.cuda stream APIs and vllm aux_stream for MUSA compatibility.

    On MUSA, ``torch.cuda.Stream`` is a dummy base class that cannot be
    instantiated (raises ``RuntimeError: Tried to instantiate dummy base class
    Stream``). Several vLLM modules create and use CUDA streams:

    - ``vllm.utils.torch_utils.aux_stream()`` creates a background stream for
      MoE shared-expert overlap. Patched to return a ``torch_musa.Stream()``.

    - ``torch.cuda.stream(s)`` context manager used in shared_experts.py.
      Patched to delegate to ``torch.musa.stream(s)`` on MUSA.

    - ``torch.cuda.set_stream(s)`` called by vllm's current_stream bookkeeping.
      Patched to delegate to ``torch.musa.set_stream(s)`` on MUSA.

    - ``torch.cuda.current_stream()`` called in some fallback paths.
      Patched to delegate to ``torch.musa.current_stream()`` on MUSA.
    """
    try:
        import torch
        import torch_musa
        import torch.cuda as torch_cuda

        if getattr(torch_cuda, "_musa_stream_patched", False):
            return

        # --- aux_stream: return torch_musa.Stream() instead of torch.cuda.Stream() ---
        try:
            import vllm.utils.torch_utils as tu_mod

            _orig_aux_stream = tu_mod.aux_stream

            def _aux_stream_musa():
                """Return a torch_musa.Stream for background stream usage on MUSA."""
                if tu_mod._aux_stream is None:
                    tu_mod._aux_stream = torch_musa.Stream()
                return tu_mod._aux_stream

            tu_mod.aux_stream = _aux_stream_musa
            # Patch the reference already imported into shared_experts module if loaded
            try:
                import vllm.model_executor.layers.fused_moe.runner.shared_experts as se_mod
                se_mod.aux_stream = _aux_stream_musa
            except Exception:
                pass
            logger.info("Patched vllm.utils.torch_utils.aux_stream for MUSA")
        except Exception as e:
            logger.warning("Failed to patch aux_stream for MUSA: %s", e)

        # --- torch.cuda.stream() context manager -> torch.musa.stream() ---
        _orig_cuda_stream_ctx = torch_cuda.stream

        def _cuda_stream_ctx_musa(stream):
            if stream is None:
                import contextlib
                return contextlib.nullcontext()
            if isinstance(stream, torch_musa.Stream):
                return torch.musa.stream(stream)
            return _orig_cuda_stream_ctx(stream)

        torch_cuda.stream = _cuda_stream_ctx_musa

        # --- torch.cuda.set_stream() -> torch.musa.set_stream() ---
        _orig_set_stream = torch_cuda.set_stream

        def _set_stream_musa(stream):
            if isinstance(stream, torch_musa.Stream):
                torch.musa.set_stream(stream)
                # Also update vllm's TLS bookkeeping
                try:
                    from vllm.utils.torch_utils import _current_stream_tls
                    _current_stream_tls.value = stream
                except Exception:
                    pass
            else:
                _orig_set_stream(stream)

        torch_cuda.set_stream = _set_stream_musa

        # --- torch.cuda.current_stream() -> torch.musa.current_stream() ---
        _orig_current_stream = torch_cuda.current_stream

        def _current_stream_musa(device=None):
            try:
                return torch.musa.current_stream(device)
            except Exception:
                return _orig_current_stream(device)

        torch_cuda.current_stream = _current_stream_musa

        torch_cuda._musa_stream_patched = True
        logger.info("Patched torch.cuda stream APIs for MUSA")
    except Exception as e:
        logger.warning("Failed to patch torch.cuda stream APIs for MUSA: %s", e)
