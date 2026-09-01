# Copyright (c) 2026 BAAI. All rights reserved.
"""Runtime compatibility for vLLM 0.24's CPU GDN metadata."""

import logging
from functools import wraps

import torch

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_vllm_fl_arm_cpu_gdn_state_indices_patch"


def apply_arm_cpu_gdn_state_indices_patch(builder_cls=None) -> bool:
    """Materialize dense CPU state indices returned by the GDN builder.

    A singleton ``block_table[:, 0]`` view can report ``is_contiguous()`` while
    retaining a stride larger than one. vLLM's native CPU GDN kernel requires
    the last dimension to have stride one. The wrapper is a no-op for tensors
    that already satisfy that contract and is safe to install repeatedly.
    """
    if builder_cls is None:
        from vllm.v1.attention.backends.gdn_attn import (
            GDNAttentionMetadataBuilder,
        )

        builder_cls = GDNAttentionMetadataBuilder

    original = builder_cls.build
    if getattr(original, _PATCH_MARKER, False):
        return False

    @wraps(original)
    def build_with_dense_cpu_state_indices(self, *args, **kwargs):
        metadata = original(self, *args, **kwargs)
        state_indices = metadata.non_spec_state_indices_tensor
        if (
            state_indices is not None
            and state_indices.device.type == "cpu"
            and state_indices.ndim > 0
            and state_indices.stride(-1) != 1
        ):
            metadata.non_spec_state_indices_tensor = state_indices.clone(
                memory_format=torch.contiguous_format
            )
        return metadata

    setattr(build_with_dense_cpu_state_indices, _PATCH_MARKER, True)
    build_with_dense_cpu_state_indices._vllm_fl_original = original
    builder_cls.build = build_with_dense_cpu_state_indices
    logger.info("Installed ARM CPU GDN state-index compatibility for vLLM 0.24")
    return True


__all__ = ["apply_arm_cpu_gdn_state_indices_patch"]
