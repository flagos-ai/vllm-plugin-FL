# Copyright (c) 2026 BAAI. All rights reserved.

"""Hygon (HCU/DCU) patches for vLLM 0.32 compatibility."""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def apply_hygon_patches():
    """Apply Hygon patches that must run before model construction.

    Cached so the patches run once per process. ``lru_cache`` does not cache
    exceptions, so a failed call can be retried.
    """
    force_v1_model_runner()
    patch_accelerator_missing_attrs()


def force_v1_model_runner() -> None:
    """Disable vLLM 0.32's default V2 model runner.

    ``ModelRunnerFL`` is a fork of vLLM's v1 runner. With the V2 runner enabled,
    0.32's attention metadata builders (e.g. GDN) take CUDA-only paths such as
    ``mamba_aligned_state_indices`` that only the V2 runner populates, so the
    FL runner cannot drive them. ``VllmConfig.use_v2_model_runner`` is a
    read-only property backed by this env var, hence it is set before any
    VllmConfig is created.

    Hygon cannot run the V2 runner at all, so an explicit opt-in is rejected
    instead of silently degrading: ``=0`` is honoured, anything else raises.
    """
    import os

    value = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    if value is None:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        logger.info("Set VLLM_USE_V2_MODEL_RUNNER=0 (vLLM-FL uses the v1 runner)")
    elif value.strip() not in ("", "0"):
        raise RuntimeError(
            f"VLLM_USE_V2_MODEL_RUNNER={value} is not supported on Hygon: vLLM-FL "
            "only implements the v1 model runner. Unset the variable (the "
            "plugin selects v1) or set it to 0."
        )


def patch_accelerator_missing_attrs() -> bool:
    """Fill in ``torch.accelerator`` APIs that DTK's Torch 2.9 does not ship.

    vLLM 0.32 targets Torch 2.13, whose ``torch.accelerator`` namespace exposes
    the generic device-management entry points. The FlagOS Hygon image pins
    ``torch 2.9.0+das`` (DTK 26.04), where only the allocator helpers exist:

    - ``get_memory_info`` — used by ``vllm.utils.mem_utils.MemorySnapshot``
      during worker startup. Delegated to ``torch.cuda.mem_get_info`` (HCU
      devices go through the CUDA-shaped API).
    - ``empty_host_cache`` — used when releasing pinned host memory.
      Torch 2.9 has no host cache concept on HCU, so this is a no-op.
    - ``current_device`` / ``set_device`` — the CUDA-shaped equivalents.

    Return True when the accelerator namespace is complete afterwards.
    """
    try:
        import torch

        if getattr(torch.accelerator, "_hcu_attrs_patched", False):
            return True

        # Only fill in attributes the installed Torch does not provide, so a
        # newer DTK that ships the real implementations keeps them instead of
        # being shadowed by the CUDA-shaped fallbacks below.
        candidates = {
            "get_memory_info": lambda device_index=None: torch.cuda.mem_get_info(
                device_index
            ),
            "current_device": torch.cuda.current_device,
            "set_device": torch.cuda.set_device,
            "empty_host_cache": lambda device_index=None: None,
        }
        bindings = {
            name: fn
            for name, fn in candidates.items()
            if not hasattr(torch.accelerator, name)
        }

        for name, fn in bindings.items():
            setattr(torch.accelerator, name, fn)

        if not hasattr(torch.cuda, "get_device_count"):
            torch.cuda.get_device_count = torch.cuda.device_count

        torch.accelerator._hcu_attrs_patched = True
        logger.info(
            "Patched torch.accelerator APIs for Hygon DTK Torch: %s",
            sorted(bindings),
        )
        return True
    except Exception as exc:  # pragma: no cover - never block registration
        logger.warning("Failed to patch torch.accelerator attrs for Hygon: %s", exc)
        return False
