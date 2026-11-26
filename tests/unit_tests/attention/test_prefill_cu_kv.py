import pytest
import torch

from vllm_fl.attention import prefill_cu_kv as cu_kv


def test_cumulative_lengths_keep_int32():
    lengths = torch.tensor([0, 15, 16, 17, 131072], dtype=torch.int32)
    expected = torch.tensor([0, 0, 15, 31, 48, 131120], dtype=torch.int32)
    torch.testing.assert_close(cu_kv.cumulative_kv_lengths_once(lengths), expected)


def test_runtime_fallback_is_limited_to_allocation_and_unsupported_errors(monkeypatch):
    def oom(_):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(cu_kv, "cumulative_kv_lengths_once", oom)
    assert cu_kv.try_cumulative_kv_lengths_once(torch.tensor([1]))[0] is None

    def shape_error(_):
        raise RuntimeError("unrelated shape bug")

    monkeypatch.setattr(cu_kv, "cumulative_kv_lengths_once", shape_error)
    with pytest.raises(RuntimeError, match="unrelated shape bug"):
        cu_kv.try_cumulative_kv_lengths_once(torch.tensor([1]))
