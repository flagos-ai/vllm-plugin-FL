"""TP8 fallback contracts remain valid after removing unsupported TP32 paths."""

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.attention.kimi_k3 import decode


@pytest.mark.parametrize("batch", [1, 2, 64])
def test_tp8_dense_fallback_keeps_layout_arguments(monkeypatch, batch):
    q = torch.empty((batch, 1, 12, 576), device="meta", dtype=torch.bfloat16)
    cache = torch.empty((2, 768, 1, 576), device="meta", dtype=q.dtype)
    table = torch.empty((batch, 2), device="meta", dtype=torch.int32)
    lengths = torch.empty((batch,), device="meta", dtype=torch.int32)
    out = torch.empty((batch, 1, 12, 512), device="meta", dtype=q.dtype)
    lse = torch.empty((batch, 12, 1), device="meta", dtype=torch.float32)
    calls = []
    monkeypatch.setattr(
        decode, "dense_decode_fallback", lambda *args: calls.append(args)
    )
    assert decode.try_dense_decode(
        q, cache, table, lengths, out, lse, softmax_scale=192**-0.5, causal=False
    )
    assert len(calls) == 1
    assert all(
        a is b for a, b in zip(calls[0][:6], (q, cache, table, lengths, out, lse))
    )
    assert calls[0][6:] == (batch, 1, 12, 576, 512, 768, 192**-0.5, False)


def test_dense_fallback_keeps_stride_guard(monkeypatch):
    calls = []
    monkeypatch.setattr(
        decode, "dense_decode_fallback", lambda *args: calls.append(args)
    )
    noncontiguous = SimpleNamespace(stride=lambda dim: 2)
    with pytest.raises(NotImplementedError, match="contiguous entry/table"):
        decode.try_dense_decode(
            None, noncontiguous, None, None, None, None, softmax_scale=1.0, causal=False
        )
    assert not calls


def test_fgv_entry_preserves_arguments_and_dense_callback(monkeypatch):
    args = tuple(object() for _ in range(17))
    result = object()
    calls = []

    def original(*positional, **keywords):
        calls.append((positional, keywords))
        return result

    monkeypatch.setattr(decode, "flagos_decode", original)
    assert decode.flash_mla_with_kvcache(*args) is result
    assert calls == [(args, {"dense_decode_backend": decode.try_dense_decode})]
