# Copyright (c) 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import __future__

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.kunlunxin.register_ops import register_builtins
from vllm_fl.dispatch.registry import OpRegistry

ROOT = Path(__file__).resolve().parents[3]


def _load_body(filename, function, namespace, *, owner=None):
    source = ROOT / "vllm_fl" / "ops" / "fused_moe" / filename
    nodes = ast.parse(source.read_text(encoding="utf-8")).body
    if owner:
        nodes = next(
            node.body
            for node in nodes
            if isinstance(node, ast.ClassDef) and node.name == owner
        )
    method = next(
        node
        for node in nodes
        if isinstance(node, ast.FunctionDef) and node.name == function
    )
    module = ast.Module(body=[method], type_ignores=[])
    exec(
        compile(
            module, str(source), "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )
    return namespace[function]


def test_kunlunxin_registers_fused_experts(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.kunlunxin.kunlunxin import KunlunxinBackend

    execute = Mock(return_value=object())
    monkeypatch.setattr(KunlunxinBackend, "fused_experts", execute)
    monkeypatch.setattr(KunlunxinBackend, "is_available", lambda self: True)
    registry = OpRegistry()
    register_builtins(registry)
    impl = registry.get_implementation("fused_experts", "vendor.kunlunxin")

    assert impl is not None and impl.is_available()
    assert impl.fn(activation="silu") is execute.return_value
    execute.assert_called_once_with(activation="silu")


@pytest.mark.parametrize("vendor", ["kunlunxin", "other"])
def test_legacy_experts_preserves_provider_arguments(vendor):
    quant = SimpleNamespace(
        **dict.fromkeys(
            ("use_fp8_w8a8", "use_int8_w8a8", "use_int8_w8a16", "use_int4_w4a16"),
            False,
        ),
        **dict.fromkeys(
            (
                "w1_scale",
                "w2_scale",
                "w1_zp",
                "w2_zp",
                "a1_scale",
                "a2_scale",
                "block_shape",
                "w1_bias",
                "w2_bias",
            ),
            None,
        ),
        per_act_token_quant=False,
    )
    legacy = Mock(return_value=object())
    vendor_impl = Mock(return_value=object())
    resolve = Mock(return_value=vendor_impl)
    fused_experts = _load_body(
        "fused_moe.py",
        "fused_experts",
        {
            "FUSED_MOE_UNQUANTIZED_CONFIG": quant,
            "current_platform": SimpleNamespace(vendor_name=vendor),
            "fused_experts_impl": legacy,
            "resolve_op": resolve,
        },
    )
    hidden_states = torch.zeros(2, 3)
    result = fused_experts(hidden_states, None, None, None, None, activation="gelu")

    if vendor == "kunlunxin":
        resolve.assert_called_once_with("fused_experts")
        legacy.assert_not_called()
        assert result is vendor_impl.return_value
        assert vendor_impl.call_args.kwargs["hidden_states"] is hidden_states
        assert vendor_impl.call_args.kwargs["activation"] == "gelu"
    else:
        resolve.assert_not_called()
        vendor_impl.assert_not_called()
        assert result is legacy.return_value


def test_modular_experts_keeps_workspace_and_provider_errors():
    execute = Mock(return_value=torch.ones(2, 3))
    resolve = Mock(return_value=execute)
    apply = _load_body(
        "fused_moe_utils.py",
        "apply",
        {
            "current_platform": SimpleNamespace(vendor_name="kunlunxin"),
            "resolve_op": resolve,
        },
        owner="TritonExpertsFL",
    )
    experts = SimpleNamespace(
        _lora_context=None,
        quant_config=SimpleNamespace(
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
        ),
        **dict.fromkeys(("w1_scale", "w2_scale", "block_shape", "w1_bias", "w2_bias")),
        per_act_token_quant=False,
    )
    output = torch.zeros(2, 3)
    hidden_states = torch.zeros_like(output)
    args = (
        experts,
        output,
        hidden_states,
        None,
        None,
        None,
        None,
        SimpleNamespace(value="silu"),
        8,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
    )

    assert apply(*args) is None
    torch.testing.assert_close(output, execute.return_value)
    assert execute.call_args.args[0] is hidden_states
    assert execute.call_args.kwargs["inplace"] is False
    assert execute.call_args.kwargs["global_num_experts"] == 8

    error = NotImplementedError("unsupported activation")
    execute.side_effect = error
    with pytest.raises(NotImplementedError) as raised:
        apply(*args)
    assert raised.value is error
    execute.side_effect = None
    assert apply(*args) is None
    assert resolve.call_args.args == ("fused_experts",)
