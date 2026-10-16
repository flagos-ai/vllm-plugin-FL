# Copyright 2026 FlagOS Contributors

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def registered_clamp(monkeypatch):
    import vllm.platforms
    from vllm.model_executor.custom_op import op_registry_oot

    from vllm_fl.dispatch import reset_default_manager, reset_global_policy
    from vllm_fl.ops import custom_ops
    from vllm_fl.ops.activation import SiluAndMulWithClampFL

    monkeypatch.setenv("USE_FLAGGEMS", "1")
    monkeypatch.setenv("VLLM_FL_FLAGOS_WHITELIST", "silu_and_mul_with_clamp")
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)
    monkeypatch.delenv("VLLM_FL_CONFIG", raising=False)
    monkeypatch.delenv("VLLM_FL_PER_OP", raising=False)
    monkeypatch.delitem(op_registry_oot, "SiluAndMulWithClamp", raising=False)
    monkeypatch.setattr(SiluAndMulWithClampFL, "name", SiluAndMulWithClampFL.name)
    monkeypatch.setattr(custom_ops, "_patch_unquantized_moe_oracle", lambda: None)
    monkeypatch.setattr(custom_ops, "_patch_fused_moe_factory", lambda: None)
    with monkeypatch.context() as registration:
        registration.setattr(
            vllm.platforms,
            "current_platform",
            SimpleNamespace(device_type="cuda", vendor_name="nvidia"),
        )
        registration.setattr(
            "vllm_fl.dispatch.config.utils.get_platform_name", lambda: "cuda"
        )
        custom_ops.register_oot_ops(whitelist=["silu_and_mul_with_clamp"])

    calls = []
    package = ModuleType("flaggems_vllm")

    def fused(gate, up, limit):
        calls.append("fused")
        gate = gate.clamp(max=limit)
        return gate * torch.sigmoid(gate) * up.clamp(-limit, limit)

    package.silu_and_mul_with_clamp = fused
    monkeypatch.setitem(sys.modules, "flaggems_vllm", package)
    reset_global_policy()
    reset_default_manager()
    yield calls
    reset_global_policy()
    reset_default_manager()


def construct(controls, alpha=1.0, beta=0.0):
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    config = VllmConfig()
    config.compilation_config.custom_ops = controls
    return config, lambda: SiluAndMulWithClamp(2.0, alpha, beta, compile_native=False)


@pytest.mark.parametrize("switch", ["0", "1"])
@pytest.mark.parametrize(
    "controls,enabled",
    [
        (["all"], True),
        (["none"], False),
        (["all", "-silu_and_mul_with_clamp"], False),
        (["none", "+silu_and_mul_with_clamp"], True),
        (["all", "-SiluAndMulWithClamp"], False),
        (["none", "+SiluAndMulWithClamp"], True),
    ],
)
def test_upstream_clamp_controls(
    registered_clamp, monkeypatch, switch, controls, enabled
):
    from vllm.config import set_current_vllm_config

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", switch)
    config, make = construct(controls)
    with set_current_vllm_config(config):
        obj = make()
        assert obj.name == "silu_and_mul_with_clamp"
        assert obj.enabled() is enabled
        x = torch.randn(3, 16)
        torch.testing.assert_close(obj(x), obj.forward_native(x))
    assert bool(registered_clamp) == (switch == "1" and enabled)


@pytest.mark.parametrize("source", ["environment", "yaml", "context"])
def test_clamp_reference_policy(registered_clamp, monkeypatch, tmp_path, source):
    from vllm.config import set_current_vllm_config

    from vllm_fl.dispatch import SelectionPolicy, policy_context, reset_global_policy

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    config, make = construct(["all"])
    with set_current_vllm_config(config):
        obj = make()
        x = torch.randn(3, 16)
        obj(x)
        assert registered_clamp
        registered_clamp.clear()
        if source == "environment":
            monkeypatch.setenv("VLLM_FL_PER_OP", "silu_and_mul_with_clamp=reference")
            reset_global_policy()
            actual = obj(x)
        elif source == "yaml":
            path = tmp_path / "reference.yml"
            path.write_text("op_backends:\n  silu_and_mul_with_clamp: [reference]\n")
            monkeypatch.setenv("VLLM_FL_CONFIG", str(path))
            reset_global_policy()
            actual = obj(x)
        else:
            policy = SelectionPolicy.from_dict(
                per_op_order={"silu_and_mul_with_clamp": ["reference"]}
            )
            with policy_context(policy):
                actual = obj(x)
            assert not registered_clamp
            obj(x)
            assert registered_clamp
            registered_clamp.clear()
        torch.testing.assert_close(actual, obj.forward_native(x))
    assert not registered_clamp


@pytest.mark.parametrize("alpha,beta", [(1.702, 0.0), (1.0, 1.0)])
def test_unsupported_clamp_parameters_stay_native(
    registered_clamp, monkeypatch, alpha, beta
):
    from vllm.config import set_current_vllm_config

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    config, make = construct(["all"], alpha, beta)
    with set_current_vllm_config(config):
        obj = make()
        x = torch.randn(3, 16)
        torch.testing.assert_close(obj(x), obj.forward_native(x))
    assert not registered_clamp


def test_compile_selection_preserves_controls(monkeypatch):
    from vllm_fl.ops.activation import enable_fused_activations_for_compile

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    monkeypatch.setenv(
        "VLLM_FL_OOT_WHITELIST", "silu_and_mul,gelu_and_mul,silu_and_mul_with_clamp"
    )
    config = SimpleNamespace(custom_ops=["none", "-GeluAndMul"])
    enable_fused_activations_for_compile(config)
    enable_fused_activations_for_compile(config)
    assert config.custom_ops == [
        "none",
        "-GeluAndMul",
        "+SiluAndMul",
        "+silu_and_mul_with_clamp",
    ]
    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "0")
    untouched = SimpleNamespace(custom_ops=["all", "-silu_and_mul_with_clamp"])
    enable_fused_activations_for_compile(untouched)
    assert untouched.custom_ops == ["all", "-silu_and_mul_with_clamp"]
