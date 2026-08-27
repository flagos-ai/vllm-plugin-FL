import sys
from types import ModuleType

from vllm_fl.dispatch.backends.vendor.cuda.impl.fused_moe import moe_sum_cuda


def test_cuda_moe_sum_unwraps_dispatch_adapter(monkeypatch):
    calls = []
    custom_ops = ModuleType("vllm._custom_ops")

    def native_moe_sum(inp, out):
        calls.append(("native", inp, out))

    def dispatch_adapter(inp, out):
        calls.append(("dispatch", inp, out))

    dispatch_adapter._vllm_fl_original = native_moe_sum
    custom_ops.moe_sum = dispatch_adapter
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", custom_ops)

    inp, out = object(), object()
    moe_sum_cuda(inp, out)

    assert calls == [("native", inp, out)]
