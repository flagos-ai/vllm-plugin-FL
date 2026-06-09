# Copyright (c) 2026 BAAI. All rights reserved.

"""
MUSA-specific patches for vLLM 0.20.2 compatibility.
"""

import contextlib
import importlib
import logging
import sys

import triton
import triton.language as tl

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
    patch_accelerator_api()
    patch_cuda_get_device_capability()
    patch_cuda_can_device_access_peer()
    patch_cuda_get_device_properties()
    patch_cuda_stream_for_musa()
    patch_model_runner_sync()
    patch_spec_decode_prepare_next_token()


def _get_musa_device_module():
    import torch
    from vllm.platforms import current_platform

    torch_device_fn = getattr(current_platform, "torch_device_fn", None)
    if torch_device_fn is not None:
        return torch_device_fn
    if hasattr(torch, "musa"):
        return torch.musa
    return None


def patch_mccl_backend():
    """Register the mccl distributed backend for MUSA via torch_musa."""
    try:
        import torch_musa.distributed as musa_dist

        musa_dist._apply_distributed_patch()
        logger.info("Registered mccl distributed backend for MUSA")
    except Exception as e:
        logger.warning("Failed to register mccl backend for MUSA: %s", e)


def patch_topk_topp_sampler():
    """Use PyTorch-native top-k/top-p on MUSA."""
    try:
        import vllm.v1.sample.ops.topk_topp_sampler as sampler_mod
        from vllm.v1.sample.ops.topk_topp_sampler import (
            apply_top_k_top_p_pytorch,
        )

        sampler_mod.apply_top_k_top_p = apply_top_k_top_p_pytorch
        logger.info("Patched apply_top_k_top_p to use PyTorch-native path for MUSA")
    except Exception as e:
        logger.debug("Failed to patch top-k/top-p sampler for MUSA: %s", e)


def patch_accelerator_api():
    """Patch torch.accelerator APIs used by vLLM on MUSA."""
    try:
        import torch

        accelerator = getattr(torch, "accelerator", None)
        musa_device = _get_musa_device_module()
        if accelerator is None or musa_device is None:
            return
        if getattr(accelerator, "_musa_accelerator_api_patched", False):
            return

        def _synchronize_musa(device=None):
            if hasattr(musa_device, "synchronize"):
                if device is None:
                    return musa_device.synchronize()
                return musa_device.synchronize(device)
            return None

        def _empty_cache_musa():
            if hasattr(musa_device, "empty_cache"):
                return musa_device.empty_cache()
            return None

        def _device_index_musa(index=None):
            if index is None:
                return contextlib.nullcontext()
            try:
                index = int(index)
            except Exception:
                return contextlib.nullcontext()
            if hasattr(musa_device, "device"):
                return musa_device.device(index)

            @contextlib.contextmanager
            def _device_context():
                old_index = None
                if hasattr(musa_device, "current_device"):
                    try:
                        old_index = musa_device.current_device()
                    except Exception:
                        old_index = None
                if hasattr(musa_device, "set_device"):
                    try:
                        musa_device.set_device(index)
                    except Exception:
                        pass
                try:
                    yield
                finally:
                    if old_index is not None and hasattr(musa_device, "set_device"):
                        try:
                            musa_device.set_device(old_index)
                        except Exception:
                            pass

            return _device_context()

        accelerator.synchronize = _synchronize_musa
        accelerator.empty_cache = _empty_cache_musa
        accelerator.device_index = _device_index_musa
        accelerator._musa_accelerator_api_patched = True
        logger.info(
            "Patched torch.accelerator synchronize/empty_cache/device_index for MUSA"
        )
    except Exception as e:
        logger.warning("Failed to patch torch.accelerator APIs for MUSA: %s", e)


def patch_cuda_get_device_capability():
    """Patch torch.cuda.get_device_capability for MUSA."""
    try:
        import torch.cuda as torch_cuda

        if getattr(torch_cuda, "_musa_get_device_capability_patched", False):
            return

        orig_get_device_capability = torch_cuda.get_device_capability

        def _get_device_capability_musa(device=None):
            try:
                return orig_get_device_capability(device)
            except (ValueError, RuntimeError):
                return (8, 0)

        torch_cuda.get_device_capability = _get_device_capability_musa
        torch_cuda._musa_get_device_capability_patched = True
        logger.info("Patched torch.cuda.get_device_capability for MUSA")
    except Exception as e:
        logger.warning("Failed to patch torch.cuda.get_device_capability for MUSA: %s", e)


def patch_cuda_can_device_access_peer():
    """Patch torch.cuda.can_device_access_peer for MUSA custom all-reduce init."""
    try:
        import torch.cuda as torch_cuda

        if getattr(torch_cuda, "_musa_can_device_access_peer_patched", False):
            return

        orig_can_device_access_peer = torch_cuda.can_device_access_peer

        def _can_device_access_peer_musa(device, peer_device):
            try:
                return orig_can_device_access_peer(device, peer_device)
            except (AssertionError, RuntimeError, ValueError):
                return False

        torch_cuda.can_device_access_peer = _can_device_access_peer_musa
        torch_cuda._musa_can_device_access_peer_patched = True
        logger.info("Patched torch.cuda.can_device_access_peer for MUSA")
    except Exception as e:
        logger.warning(
            "Failed to patch torch.cuda.can_device_access_peer for MUSA: %s", e
        )


def patch_cuda_get_device_properties():
    """Avoid subprocess-based CUDA property probing on MUSA."""
    try:
        import torch_musa
        import vllm.utils.platform_utils as platform_utils

        if getattr(platform_utils, "_musa_cuda_get_device_properties_patched", False):
            return

        def _cuda_get_device_properties_musa(device, names, init_cuda=False):
            props = torch_musa.get_device_properties(device)
            return tuple(getattr(props, name) for name in names)

        platform_utils.cuda_get_device_properties = _cuda_get_device_properties_musa
        try:
            import vllm.usage.usage_lib as usage_lib

            usage_lib.cuda_get_device_properties = _cuda_get_device_properties_musa
        except Exception:
            pass
        platform_utils._musa_cuda_get_device_properties_patched = True
        logger.info("Patched cuda_get_device_properties to use torch_musa for MUSA")
    except Exception as e:
        logger.warning("Failed to patch cuda_get_device_properties for MUSA: %s", e)


def patch_cuda_stream_for_musa():
    """Patch CUDA stream helpers to delegate MUSA streams to torch.musa."""
    try:
        import torch.cuda as torch_cuda

        musa_device = _get_musa_device_module()
        if musa_device is None:
            return
        if getattr(torch_cuda, "_musa_stream_patched", False):
            return

        try:
            import vllm.utils.torch_utils as torch_utils

            def _aux_stream_musa():
                if torch_utils._aux_stream is None:
                    torch_utils._aux_stream = musa_device.Stream()
                return torch_utils._aux_stream

            torch_utils.aux_stream = _aux_stream_musa
            torch_utils._musa_aux_stream_patched = True
            logger.info("Patched vllm.utils.torch_utils.aux_stream for MUSA")
        except Exception as e:
            logger.warning("Failed to patch aux_stream for MUSA: %s", e)

        orig_cuda_stream = torch_cuda.stream
        orig_set_stream = torch_cuda.set_stream
        orig_current_stream = torch_cuda.current_stream

        def _is_musa_stream(stream):
            stream_type = getattr(musa_device, "Stream", None)
            return stream_type is not None and isinstance(stream, stream_type)

        def _cuda_stream_musa(stream):
            if stream is None:
                return contextlib.nullcontext()
            if _is_musa_stream(stream) and hasattr(musa_device, "stream"):
                return musa_device.stream(stream)
            return orig_cuda_stream(stream)

        def _set_stream_musa(stream):
            if _is_musa_stream(stream) and hasattr(musa_device, "set_stream"):
                musa_device.set_stream(stream)
                try:
                    from vllm.utils.torch_utils import _current_stream_tls

                    _current_stream_tls.value = stream
                except Exception:
                    pass
                return None
            return orig_set_stream(stream)

        def _current_stream_musa(device=None):
            try:
                if hasattr(musa_device, "current_stream"):
                    return musa_device.current_stream(device)
            except Exception:
                pass
            return orig_current_stream(device)

        torch_cuda.stream = _cuda_stream_musa
        torch_cuda.set_stream = _set_stream_musa
        torch_cuda.current_stream = _current_stream_musa
        torch_cuda._musa_stream_patched = True
        logger.info("Patched torch.cuda stream APIs for MUSA")
    except Exception as e:
        logger.warning("Failed to patch torch.cuda stream APIs for MUSA: %s", e)


def patch_model_runner_sync():
    """Patch ModelRunnerFL._sync_device when the class is importable."""
    try:
        import vllm_fl.worker.model_runner as model_runner

        if getattr(model_runner.ModelRunnerFL, "_musa_sync_device_patched", False):
            return

        def _sync_device_musa(self):
            musa_device = _get_musa_device_module()
            if musa_device is not None and hasattr(musa_device, "synchronize"):
                musa_device.synchronize()

        model_runner.ModelRunnerFL._sync_device = _sync_device_musa
        model_runner.ModelRunnerFL._musa_sync_device_patched = True
        logger.info("Patched ModelRunnerFL._sync_device for MUSA")
    except Exception as e:
        logger.debug("Failed to patch ModelRunnerFL._sync_device for MUSA: %s", e)


@triton.jit
def _musa_eagle_prepare_next_token_padded_kernel(
    sampled_token_ids_ptr,
    discard_request_mask_ptr,
    backup_next_token_ids_ptr,
    next_token_ids_ptr,
    valid_sampled_tokens_count_ptr,
    vocab_size,
    num_sampled_tokens_per_req,
    num_reqs,
    stride_sampled_token_ids,
    BLOCK_SIZE_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    is_discarded = tl.load(discard_request_mask_ptr + req_idx)
    if is_discarded:
        backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
        valid_count = tl.full((), 0, dtype=tl.int32)
        tl.store(next_token_ids_ptr + req_idx, backup_token)
        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)
    else:
        token_offsets = tl.arange(0, BLOCK_SIZE_TOKENS)
        token_mask = token_offsets < num_sampled_tokens_per_req
        row_ptr = sampled_token_ids_ptr + req_idx * stride_sampled_token_ids
        token_ids = tl.load(row_ptr + token_offsets, mask=token_mask, other=-1)

        is_valid_mask = (token_ids != -1) & (token_ids < vocab_size) & token_mask
        valid_count = tl.sum(is_valid_mask.to(tl.int32))

        if valid_count > 0:
            last_valid_index = tl.max(tl.where(is_valid_mask, token_offsets, -1))
            last_valid_token = tl.sum(
                tl.where(token_offsets == last_valid_index, token_ids, 0)
            )
            tl.store(next_token_ids_ptr + req_idx, last_valid_token)
        else:
            backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
            tl.store(next_token_ids_ptr + req_idx, backup_token)

        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)


def patch_spec_decode_prepare_next_token():
    """Patch Eagle/MTP prepare-next-token Triton kernel for MUSA."""
    try:
        utils_mod = importlib.import_module("vllm.v1.spec_decode.utils")
        utils_mod.eagle_prepare_next_token_padded_kernel = (
            _musa_eagle_prepare_next_token_padded_kernel
        )

        for module_name in (
            "vllm.v1.spec_decode.eagle",
            "vllm.v1.spec_decode.llm_base_proposer",
        ):
            caller_mod = sys.modules.get(module_name)
            if caller_mod is not None and hasattr(
                caller_mod, "eagle_prepare_next_token_padded_kernel"
            ):
                caller_mod.eagle_prepare_next_token_padded_kernel = (
                    _musa_eagle_prepare_next_token_padded_kernel
                )

        logger.info("Patched Eagle/MTP prepare-next-token kernel for MUSA Triton")
    except Exception as e:
        logger.warning(
            "Failed to patch Eagle/MTP prepare-next-token kernel for MUSA: %s", e
        )
