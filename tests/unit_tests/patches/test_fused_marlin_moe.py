from types import ModuleType

from vllm_fl.patches import fused_marlin_moe


def test_fused_marlin_moe_patch_routes_and_is_idempotent(monkeypatch):
    calls = []
    module = ModuleType("fake_marlin_moe")
    module.fused_marlin_moe = lambda *args, **kwargs: calls.append(
        ("original", args, kwargs)
    )

    monkeypatch.setattr(fused_marlin_moe, "use_flaggems_op", lambda name: True)
    monkeypatch.setattr(
        fused_marlin_moe,
        "_dispatch_fused_marlin_moe",
        lambda *args, **kwargs: calls.append(("dispatch", args, kwargs)),
    )

    assert fused_marlin_moe.patch_vllm_fused_marlin_moe(module) is True
    assert fused_marlin_moe.patch_vllm_fused_marlin_moe(module) is False
    module.fused_marlin_moe("hidden", output="out")

    assert calls == [("dispatch", ("hidden",), {"output": "out"})]
    assert module.fused_marlin_moe._vllm_fl_original is not module.fused_marlin_moe


def test_fused_marlin_moe_patch_respects_flaggems_disable(monkeypatch):
    module = ModuleType("fake_marlin_moe")
    original = lambda *args, **kwargs: None
    module.fused_marlin_moe = original

    monkeypatch.setattr(fused_marlin_moe, "use_flaggems_op", lambda name: False)

    assert fused_marlin_moe.patch_vllm_fused_marlin_moe(module) is False
    assert module.fused_marlin_moe is original
