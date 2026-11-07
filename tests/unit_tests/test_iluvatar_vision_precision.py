from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision import (
    _wrap_vision_forward,
    in_vision_forward,
)
from vllm_fl.distributed import communicator as module


def test_context_nested_exception_and_idempotence():
    def fail():
        assert in_vision_forward()
        raise ValueError("test")

    wrapped = _wrap_vision_forward(fail)
    assert _wrap_vision_forward(wrapped) is wrapped
    with pytest.raises(ValueError, match="test"):
        wrapped()
    assert not in_vision_forward()

    @_wrap_vision_forward
    def outer():
        assert _wrap_vision_forward(lambda: in_vision_forward())()
        assert in_vision_forward()

    outer()
    assert not in_vision_forward()


@pytest.mark.parametrize("vendor", ["iluvatar", "nvidia"])
@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
def test_precision_scoped_to_iluvatar_vision(monkeypatch, vendor, vision, fallback):
    monkeypatch.setattr(module, "current_platform", SimpleNamespace(vendor_name=vendor))
    seen = []

    def reduce(x):
        seen.append(x.dtype)
        return None if fallback else x * 2

    def native(x, group):
        x.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", native)
    comm = object.__new__(module.CommunicatorFL)
    comm.pyflagcx_comm = SimpleNamespace(all_reduce=reduce)
    comm.device_group = object()
    x = torch.tensor([1.25, -2.5], dtype=torch.bfloat16)
    call = lambda: comm.all_reduce(x)
    y = _wrap_vision_forward(call)() if vision else call()
    assert seen == [torch.float32 if vendor == "iluvatar" and vision else x.dtype]
    assert y.dtype == x.dtype
    assert torch.equal(y, x * 2)
    assert not in_vision_forward()
