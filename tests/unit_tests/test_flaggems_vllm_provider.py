# Copyright 2026 FlagOS Contributors

import sys
from types import ModuleType


def test_fused_experts_uses_flaggems_vllm_without_changing_arguments(monkeypatch):
    from vllm_fl.quantization.w8a8 import moe_experts

    calls = []
    module = ModuleType("flaggems_vllm")

    def invoke(**kwargs):
        calls.append(kwargs)
        return kwargs["hidden_states"]

    module.fused_experts_impl = invoke
    monkeypatch.setitem(sys.modules, "flaggems_vllm", module)
    value = object()
    assert (
        moe_experts._flaggems_fused_experts_impl(hidden_states=value, inplace=False)
        is value
    )
    assert calls == [{"hidden_states": value, "inplace": False}]
