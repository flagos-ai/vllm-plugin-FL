# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Verify the adapter's pinned V2 kernel launch contract."""

import sys
from types import ModuleType
from unittest.mock import Mock

import torch

from vllm_fl.dispatch.backends.vendor.kunlunxin.patches import (
    structured_outputs_compat,
)


def test_grammar_mask_delegates_the_kernel_launch(monkeypatch):
    module = ModuleType("vllm.v1.worker.gpu.structured_outputs")
    module._apply_grammar_bitmask_kernel = Mock()
    worker = Mock()
    module.StructuredOutputsWorker = worker
    monkeypatch.setitem(sys.modules, module.__name__, module)

    provider = ModuleType("flaggems_vllm")
    provider.v2_apply_grammar_bitmask = Mock()
    monkeypatch.setitem(sys.modules, provider.__name__, provider)

    structured_outputs_compat.apply_structured_outputs_compat()
    kernel = module._apply_grammar_bitmask_kernel
    structured_outputs_compat.apply_structured_outputs_compat()
    assert module._apply_grammar_bitmask_kernel is kernel
    assert module.StructuredOutputsWorker is worker

    logits = torch.empty(3, 34)
    mask = torch.empty(2, 2, dtype=torch.int32)
    indices = torch.tensor([1, 3], dtype=torch.int32)
    offsets = torch.tensor([0, 2, 3], dtype=torch.int32)
    kernel[(2, 1)](
        logits,
        logits.stride(0),
        indices,
        offsets,
        mask,
        mask.stride(0),
        logits.shape[-1],
        MASK_STRIDE=2,
        BLOCK_SIZE=8192,
    )
    provider.v2_apply_grammar_bitmask.assert_called_once_with(
        logits, mask, indices, offsets, 2
    )


def test_grammar_mask_tolerates_an_absent_v2_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.structured_outputs", None)
    monkeypatch.setitem(sys.modules, "flaggems_vllm", None)
    structured_outputs_compat.apply_structured_outputs_compat()
