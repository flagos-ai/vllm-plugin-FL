import torch

from vllm_fl.patches.sharded_state import _filter_subtensors_stride_safe


def test_filter_subtensors_drops_noncontiguous_alias():
    base = torch.arange(24).reshape(4, 6)
    tensors = {
        "base": base,
        "transpose": base.t(),
        "standalone": torch.arange(3),
    }

    filtered = _filter_subtensors_stride_safe(tensors)

    assert set(filtered) == {"base", "standalone"}
    assert all(tensor.is_contiguous() for tensor in filtered.values())


def test_filter_subtensors_copies_uncovered_noncontiguous_tensor():
    tensor = torch.arange(24).reshape(4, 6).t()

    filtered = _filter_subtensors_stride_safe({"transpose": tensor})

    assert filtered["transpose"].is_contiguous()
    torch.testing.assert_close(filtered["transpose"], tensor)
