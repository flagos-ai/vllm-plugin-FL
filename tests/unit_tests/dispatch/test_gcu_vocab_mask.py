"""The wrapper must stay importable without initializing Triton kernels."""

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.gcu.impl import vocab_mask


def test_vocab_import_does_not_load_kernel():
    name = "vllm_fl.dispatch.backends.vendor.gcu.impl.vocab_mask_kernel"
    sys.modules.pop(name, None)
    importlib.reload(vocab_mask)
    assert name not in sys.modules


def test_empty_vocab_mask_needs_no_compiler(monkeypatch):
    def forbidden():
        raise AssertionError("empty input should not load a compiler")

    monkeypatch.setattr(vocab_mask, "_get_kernel", forbidden)
    output, mask = vocab_mask.get_masked_input_and_mask(
        torch.empty(0, dtype=torch.int64), 1, 4, 2, 8, 10
    )
    assert output.dtype == torch.int64 and mask.dtype == torch.bool
    assert output.numel() == mask.numel() == 0


def test_vocab_compiler_capability_error(monkeypatch):
    from dataclasses import dataclass

    @dataclass
    class OptionsWithoutInt64:
        num_warps: int = 4

    monkeypatch.setitem(
        sys.modules,
        "triton.backends.enflame.compiler",
        SimpleNamespace(GCUOptions=OptionsWithoutInt64),
    )
    vocab_mask._get_kernel.cache_clear()
    with pytest.raises(RuntimeError, match="enable_i64"):
        vocab_mask._get_kernel()


def test_vocab_install_rejects_unknown_abi():
    with pytest.raises(RuntimeError, match="ABI"):
        vocab_mask._install_vocab_mask(
            SimpleNamespace(get_masked_input_and_mask=lambda x: x)
        )
