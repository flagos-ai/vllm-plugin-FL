# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.
"""Tests for the kunlunxin structured_outputs compat shim.

The shim applies grammar bitmasks with torch instead of the V2 runner's
Triton kernel, which FlagTree cannot lower (the reshape) and miscompiles
(the bit unpacking). The numerical equivalence of the torch path is
verified on hardware; these tests cover the patch mechanics and the
masking semantics that are device independent.
"""

import sys
import types

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.kunlunxin.patches import (
    structured_outputs_compat,
)


@pytest.fixture
def fake_structured_outputs(monkeypatch):
    """Install a stand-in vllm.v1.worker.gpu.structured_outputs module."""

    class StructuredOutputsWorker:
        def apply_grammar_bitmask(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("the original method must not run")

    def _build_grammar_mapping(*args, **kwargs):  # pragma: no cover
        return []

    module = types.ModuleType("vllm.v1.worker.gpu.structured_outputs")
    module.StructuredOutputsWorker = StructuredOutputsWorker
    module._build_grammar_mapping = _build_grammar_mapping
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.structured_outputs", module)
    # ``from vllm.v1.worker.gpu import structured_outputs`` reads the attribute
    # off the parent package, so in a build that ships the real module the
    # sys.modules entry alone is not enough to redirect the shim.
    gpu_pkg = sys.modules.get("vllm.v1.worker.gpu")
    if gpu_pkg is None:
        gpu_pkg = types.ModuleType("vllm.v1.worker.gpu")
        monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu", gpu_pkg)
    monkeypatch.setattr(gpu_pkg, "structured_outputs", module, raising=False)

    torch_utils = types.ModuleType("vllm.utils.torch_utils")
    torch_utils.PIN_MEMORY = False
    torch_utils.async_tensor_h2d = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils)

    original = StructuredOutputsWorker.apply_grammar_bitmask
    return module, StructuredOutputsWorker, original


def test_exports_public_api():
    assert callable(structured_outputs_compat.apply_structured_outputs_compat)
    assert callable(structured_outputs_compat._apply_grammar_bitmask_torch)


def test_replaces_the_method(fake_structured_outputs):
    _, worker_cls, original = fake_structured_outputs

    structured_outputs_compat.apply_structured_outputs_compat()

    replaced = worker_cls.apply_grammar_bitmask
    assert replaced is not original
    assert getattr(replaced, structured_outputs_compat._PATCH_FLAG) is True


def test_is_idempotent(fake_structured_outputs):
    _, worker_cls, _ = fake_structured_outputs

    structured_outputs_compat.apply_structured_outputs_compat()
    first = worker_cls.apply_grammar_bitmask
    structured_outputs_compat.apply_structured_outputs_compat()

    assert worker_cls.apply_grammar_bitmask is first


def test_does_not_override_a_foreign_replacement(fake_structured_outputs):
    """A method someone else installed and flagged is left alone."""
    _, worker_cls, _ = fake_structured_outputs

    def foreign(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("unreachable")

    setattr(foreign, structured_outputs_compat._PATCH_FLAG, True)
    worker_cls.apply_grammar_bitmask = foreign

    structured_outputs_compat.apply_structured_outputs_compat()

    assert worker_cls.apply_grammar_bitmask is foreign


def test_tolerates_a_build_without_the_v2_path(monkeypatch):
    """Builds that cannot import the module are a no-op, not an error."""
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "vllm.v1.worker.gpu" or name.startswith(
            "vllm.v1.worker.gpu.structured_outputs"
        ):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    monkeypatch.delitem(
        sys.modules, "vllm.v1.worker.gpu.structured_outputs", raising=False
    )

    structured_outputs_compat.apply_structured_outputs_compat()


def _reference(logits, logits_indices, cu_num_logits, bitmask, mask_stride):
    """Straightforward CPU model of the upstream kernel's semantics."""
    out = logits.clone()
    vocab = logits.shape[-1]
    words = bitmask.shape[1]
    for i in range(bitmask.shape[0]):
        mapping_idx = int(logits_indices[i].item())
        req_idx = mapping_idx // mask_stride
        position_idx = mapping_idx % mask_stride
        start = int(cu_num_logits[req_idx].item())
        num_req_logits = int(cu_num_logits[req_idx + 1].item()) - start
        if position_idx >= num_req_logits:
            continue
        row = start + position_idx
        for v in range(vocab):
            word, bit = v // 32, v % 32
            if word >= words:
                continue
            if ((int(bitmask[i, word].item()) >> bit) & 1) == 0:
                out[row, v] = float("-inf")
    return out


def test_torch_path_matches_reference_on_cpu():
    torch.manual_seed(0)
    vocab, mask_stride = 96, 4
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    logits = torch.randn(int(cu[-1].item()) + 2, vocab)
    bitmask = torch.randint(
        -(2**31), 2**31 - 1, (3, (vocab + 31) // 32), dtype=torch.int32
    )
    # Second entry's position is past its request's logit count, so it is
    # skipped; the others are active.
    logits_indices = torch.tensor(
        [0, mask_stride * 0 + 3, mask_stride * 1], dtype=torch.int32
    )

    expected = _reference(logits, logits_indices, cu, bitmask, mask_stride)
    got = logits.clone()
    structured_outputs_compat._apply_grammar_bitmask_torch(
        got, bitmask, logits_indices, cu, mask_stride
    )

    assert torch.equal(got, expected)


def test_torch_path_keeps_both_masks_for_a_repeated_row():
    """Two masks can name one logit row; every blocked token must stick."""
    vocab, mask_stride = 64, 1
    cu = torch.tensor([0, 1], dtype=torch.int32)
    logits = torch.zeros(1, vocab)
    words = (vocab + 31) // 32
    # First mask blocks only token 0, second blocks only token 40.
    first = torch.full((words,), -1, dtype=torch.int32)
    first[0] = ~torch.tensor(1, dtype=torch.int32)
    second = torch.full((words,), -1, dtype=torch.int32)
    second[1] = ~torch.tensor(1 << 8, dtype=torch.int32)
    bitmask = torch.stack([first, second])
    logits_indices = torch.tensor([0, 0], dtype=torch.int32)

    structured_outputs_compat._apply_grammar_bitmask_torch(
        logits, bitmask, logits_indices, cu, mask_stride
    )

    assert torch.isneginf(logits[0, 0])
    assert torch.isneginf(logits[0, 40])
    assert not torch.isneginf(logits[0, 1])


def test_torch_path_ignores_an_empty_bitmask():
    logits = torch.zeros(2, 64)
    cu = torch.tensor([0, 2], dtype=torch.int32)
    empty = torch.zeros((0, 2), dtype=torch.int32)

    structured_outputs_compat._apply_grammar_bitmask_torch(
        logits, empty, torch.zeros(0, dtype=torch.int32), cu, 1
    )

    assert torch.equal(logits, torch.zeros(2, 64))
