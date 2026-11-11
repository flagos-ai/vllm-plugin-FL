# Copyright (c) 2026 BAAI. All rights reserved.
import importlib.util
from pathlib import Path

import pytest
import torch

PATH = (
    Path(__file__).parents[2] / "vllm_fl/dispatch/backends/vendor/gcu/impl/penalties.py"
)
spec = importlib.util.spec_from_file_location("gcu_penalties_test_impl", PATH)
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)


@pytest.mark.parametrize("length", [0, 31, 1025, 2051])
@pytest.mark.parametrize("vocab", [33, 97])
def test_bincount_preserves_unselected_rows_and_signed_bits(length, vocab):
    device = "cpu"
    import os

    if os.environ.get("TEST_GCU_PENALTIES"):
        import torch_gcu  # noqa: F401

        device = "gcu"
    tokens = (torch.arange(3 * max(length, 1)).reshape(3, -1) * 31 % vocab).int()
    prompt_n = length // 2
    prompt_lens = torch.tensor([0, prompt_n, length], dtype=torch.int32)
    prefill_lens = torch.tensor([0, length, length], dtype=torch.int32)
    words = torch.full((3, (vocab + 31) // 32), 123, dtype=torch.int32)
    counts = torch.full((3, vocab), 456, dtype=torch.int32)
    expected_words, expected_counts = words.clone(), counts.clone()
    for req in [2, 1]:
        expected_words[req].zero_()
        expected_counts[req].zero_()
        for pos in range(length):
            token = int(tokens[req, pos])
            if pos < int(prompt_lens[req]):
                word = int(expected_words[req, token // 32]) & 0xFFFFFFFF
                word |= 1 << (token % 32)
                expected_words[req, token // 32] = (
                    word if word < 2**31 else word - 2**32
                )
            else:
                expected_counts[req, token] += 1
    args = [torch.tensor([2, 1]), tokens, prompt_lens, prefill_lens, words, counts]
    args = [t.to(device) for t in args]
    impl.bincount_gcu(*args, length)
    torch.testing.assert_close(args[4].cpu(), expected_words, rtol=0, atol=0)
    torch.testing.assert_close(args[5].cpu(), expected_counts, rtol=0, atol=0)


def test_patch_is_scoped_idempotent_and_forwards_keywords(monkeypatch):
    import sys
    import types

    calls = []
    original = lambda *args, **kwargs: calls.append(("original", args, kwargs))
    module = types.ModuleType("vllm.v1.worker.gpu.sample.penalties")
    module.bincount = original
    package = types.ModuleType("vllm.v1.worker.gpu.sample")
    package.penalties = module
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        impl, "bincount_gcu", lambda *a, **k: calls.append(("gcu", a, k))
    )
    assert impl.apply_penalties_gcu_patch()
    assert not impl.apply_penalties_gcu_patch()
    cpu = types.SimpleNamespace(device=types.SimpleNamespace(type="cpu"))
    gcu = types.SimpleNamespace(device=types.SimpleNamespace(type="gcu"))
    module.bincount(cpu, "sentinel")
    module.bincount(expanded_idx_mapping=gcu, max_prefill_len=31)
    assert calls[0] == ("original", (cpu, "sentinel"), {})
    assert calls[1] == ("gcu", (), {"expanded_idx_mapping": gcu, "max_prefill_len": 31})
