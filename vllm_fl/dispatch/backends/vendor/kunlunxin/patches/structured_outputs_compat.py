# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Route the vLLM V2 grammar kernel to the Kunlunxin operator provider."""

# Image-baked provider: https://github.com/flagos-ai/FlagGems-vllm/pull/1248.
# Native grammar baseline: https://github.com/flagos-ai/FlagTree/issues/1700.

import importlib
import logging

logger = logging.getLogger(__name__)


class _GrammarBitmaskLaunch:
    """Adapt the pinned vLLM kernel launch to the provider's tensor API."""

    def __init__(self, apply_bitmask):
        self._apply_bitmask = apply_bitmask

    def __getitem__(self, _grid):
        return self._launch

    def _launch(
        self,
        logits,
        _logits_stride,
        logits_indices,
        cu_num_logits,
        bitmask,
        _bitmask_stride,
        _vocab_size,
        *,
        MASK_STRIDE,
        BLOCK_SIZE,
    ) -> None:
        self._apply_bitmask(logits, bitmask, logits_indices, cu_num_logits, MASK_STRIDE)


def apply_structured_outputs_compat() -> None:
    """Keep vLLM's request mapping and copy streams around the provider launch."""
    try:
        structured_outputs = importlib.import_module(
            "vllm.v1.worker.gpu.structured_outputs"
        )
    except ImportError:
        return

    kernel = structured_outputs._apply_grammar_bitmask_kernel
    if isinstance(kernel, _GrammarBitmaskLaunch):
        return

    from flaggems_vllm import v2_apply_grammar_bitmask

    structured_outputs._apply_grammar_bitmask_kernel = _GrammarBitmaskLaunch(
        v2_apply_grammar_bitmask
    )
    logger.info("Bound Kunlunxin Runner V2 grammar masking to FlagGems-vLLM")
