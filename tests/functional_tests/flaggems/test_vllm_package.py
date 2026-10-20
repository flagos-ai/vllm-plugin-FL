# Copyright 2026 FlagOS Contributors
"""Exercise migrated fused operators through the plugin boundary."""

from types import SimpleNamespace

import pytest
import torch

OPS = (
    "silu_and_mul",
    "silu_and_mul_with_clamp",
    "gelu_and_mul",
    "apply_rotary_pos_emb",
    "flash_attn_varlen_func",
    "reshape_and_cache_flash",
    "flash_mla",
    "moe_align_block_size_triton",
    "topk_softmax",
    "invoke_fused_moe_triton_kernel",
    "grouped_topk",
    "moe_sum",
    "fused_experts_impl",
)

pytestmark = [pytest.mark.gpu, pytest.mark.flaggems]


def test_release_exports():
    import flaggems_vllm

    for name in OPS:
        assert callable(getattr(flaggems_vllm, name)), name


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("activation", ["silu", "gelu"])
def test_activation_dispatch(monkeypatch, graph, activation):
    import flaggems_vllm

    from vllm_fl.dispatch.backends.flaggems.impl.activation import (
        gelu_and_mul_flaggems,
        silu_and_mul_flaggems,
    )

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    op_name = f"{activation}_and_mul"
    calls = []
    real_op = getattr(flaggems_vllm, op_name)

    def observed(*args, **kwargs):
        calls.append(real_op.__module__)
        return real_op(*args, **kwargs)

    monkeypatch.setattr(flaggems_vllm, op_name, observed)
    x = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)
    obj = SimpleNamespace(approximate="none")
    wrapper = silu_and_mul_flaggems if activation == "silu" else gelu_and_mul_flaggems
    expected = getattr(torch.nn.functional, activation)(x[:, :128]) * x[:, 128:]
    for _ in range(3):
        actual = wrapper(obj, x)
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            actual = wrapper(obj, x)
        x.copy_(torch.randn_like(x))
        capture.replay()
        expected = getattr(torch.nn.functional, activation)(x[:, :128]) * x[:, 128:]
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    assert calls and all(module.startswith("flaggems_vllm.") for module in calls)


@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_inplace_dispatch(monkeypatch, renormalize):
    from vllm_fl.dispatch.backends.flaggems.impl.fused_moe import topk_softmax_flaggems

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    logits = torch.randn(16, 64, device="cuda")
    weights = torch.empty(16, 4, device="cuda")
    ids = torch.empty(16, 4, device="cuda", dtype=torch.int32)
    token_ids = torch.empty_like(ids)
    expected_weights, expected_ids = logits.softmax(-1).topk(4)
    if renormalize:
        expected_weights /= expected_weights.sum(-1, keepdim=True)
    result_weights, result_ids = topk_softmax_flaggems(
        weights, ids, token_ids, logits, renormalize
    )
    assert result_weights.data_ptr() == weights.data_ptr()
    assert result_ids.data_ptr() == ids.data_ptr()
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(ids.long(), expected_ids)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (1.702, 1.0)])
def test_clamped_activation(monkeypatch, graph, alpha, beta):
    import flaggems_vllm

    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm_fl.ops.activation import SiluAndMulWithClampFL

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    calls = []
    real_op = flaggems_vllm.silu_and_mul_with_clamp

    def observed(*args, **kwargs):
        calls.append(real_op.__module__)
        return real_op(*args, **kwargs)

    monkeypatch.setattr(flaggems_vllm, "silu_and_mul_with_clamp", observed)
    with set_current_vllm_config(VllmConfig()):
        obj = SiluAndMulWithClampFL(2.0, alpha, beta, compile_native=False)
    x = torch.randn(16, 128, device="cuda") * 4
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=2.0)
    expected = gate * torch.sigmoid(alpha * gate) * (up.clamp(-2.0, 2.0) + beta)
    for _ in range(3):
        actual = obj.forward_oot(x)
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            actual = obj.forward_oot(x)
        x.copy_(torch.randn_like(x) * 4)
        capture.replay()
        gate, up = x.chunk(2, dim=-1)
        gate = gate.clamp(max=2.0)
        expected = gate * torch.sigmoid(alpha * gate) * (up.clamp(-2.0, 2.0) + beta)
    torch.testing.assert_close(actual, expected)
    assert bool(calls) == (alpha == 1.0 and beta == 0.0)


@pytest.fixture
def registered_activations(monkeypatch):
    # Each service configuration starts in a fresh process; reset compiler and
    # provider caches between tests that construct different startup modes.
    torch._dynamo.reset()
    from vllm_fl.dispatch import reset_default_manager

    reset_default_manager()
    from vllm.model_executor.custom_op import op_registry_oot

    from vllm_fl.ops import custom_ops
    from vllm_fl.ops.activation import GeluAndMulFL, SiluAndMulFL, SiluAndMulWithClampFL

    for cls, name in (
        (SiluAndMulFL, "SiluAndMul"),
        (GeluAndMulFL, "GeluAndMul"),
        (SiluAndMulWithClampFL, "SiluAndMulWithClamp"),
    ):
        monkeypatch.delitem(op_registry_oot, name, raising=False)
        monkeypatch.setattr(cls, "name", cls.name)
    monkeypatch.setattr(custom_ops, "_patch_unquantized_moe_oracle", lambda: None)
    monkeypatch.setattr(custom_ops, "_patch_fused_moe_factory", lambda: None)
    custom_ops.register_oot_ops(
        whitelist=["silu_and_mul", "gelu_and_mul", "silu_and_mul_with_clamp"]
    )


@pytest.mark.parametrize("activation", ["silu", "gelu", "clamp"])
def test_compiled_activation_uses_fused_package(
    monkeypatch, registered_activations, activation
):
    import flaggems_vllm

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import (
        GeluAndMul,
        SiluAndMul,
        SiluAndMulWithClamp,
    )

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", "1")
    name = (
        "silu_and_mul_with_clamp" if activation == "clamp" else f"{activation}_and_mul"
    )
    real = getattr(flaggems_vllm, name)
    calls = []

    def observed(*args, **kwargs):
        calls.append(real.__module__)
        return real(*args, **kwargs)

    monkeypatch.setattr(flaggems_vllm, name, observed)
    with set_current_vllm_config(VllmConfig()):
        obj = {
            "silu": SiluAndMul,
            "gelu": GeluAndMul,
            "clamp": lambda: SiluAndMulWithClamp(2.0, 1.0, 0.0, compile_native=False),
        }[activation]()
    x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
    eager = obj(x)
    calls.clear()
    compiled = torch.compile(obj, fullgraph=True)
    torch.testing.assert_close(compiled(x), eager)
    calls.clear()
    updated = x * 0.5
    actual = compiled(updated)
    expected = obj.forward_native(updated)
    torch.testing.assert_close(actual, expected)
    assert calls and all(module.startswith("flaggems_vllm.") for module in calls)


@pytest.mark.parametrize(
    "switch,controls,fused",
    [
        ("0", ["all"], False),
        ("1", ["all", "-silu_and_mul_with_clamp"], False),
        ("1", ["none", "+silu_and_mul_with_clamp"], True),
    ],
)
def test_compiled_upstream_clamp_controls(
    monkeypatch, registered_activations, switch, controls, fused
):
    import flaggems_vllm

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    monkeypatch.setenv("VLLM_FL_USE_FLAGGEMS_VLLM", switch)
    calls = []
    real = flaggems_vllm.silu_and_mul_with_clamp

    def observed(*args, **kwargs):
        calls.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(flaggems_vllm, "silu_and_mul_with_clamp", observed)
    config = VllmConfig()
    config.compilation_config.custom_ops = controls
    with set_current_vllm_config(config):
        obj = SiluAndMulWithClamp(2.0, 1.0, 0.0, compile_native=False)
        assert obj.enabled() is (controls == ["all"] or controls[0] == "none"), (
            obj.name,
            controls,
            obj._forward_method,
        )
        assert obj._forward_method.__name__ == (
            "forward_oot" if obj.enabled() else "forward_native"
        )
        compiled = torch.compile(obj, fullgraph=True)
        x = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
        compiled(x)
        calls.clear()
        updated = x * 0.5
        actual = compiled(updated)
        torch.testing.assert_close(actual, obj.forward_native(updated))
    assert bool(calls) is fused
