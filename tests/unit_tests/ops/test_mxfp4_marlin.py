# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests. FL_GEMS_MARLIN_SOURCE selects the frozen Gems source."""

import ast
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

provider_dir = Path(__file__).resolve().parents[3] / "vllm_fl/ops/fused_moe"
package = types.ModuleType("fl_mxfp4_contract_package")
package.__path__ = [str(provider_dir)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location(
    package.__name__ + ".mxfp4_marlin", provider_dir / "mxfp4_marlin.py"
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


@pytest.fixture
def raw():
    g = torch.Generator().manual_seed(17)
    return (
        torch.randint(0, 256, (3, 256, 64), dtype=torch.uint8, generator=g),
        torch.randint(0, 256, (3, 128, 64), dtype=torch.uint8, generator=g),
        torch.randint(118, 123, (3, 256, 4), dtype=torch.uint8, generator=g),
        torch.randint(118, 123, (3, 128, 4), dtype=torch.uint8, generator=g),
    )


def _frozen_packers():
    source = os.environ.get("FL_GEMS_MARLIN_SOURCE")
    if not source:
        pytest.skip("set FL_GEMS_MARLIN_SOURCE to audit the exact dependency packers")
    tree = ast.parse(Path(source).read_text())
    wanted = {"_pack_w_interleave", "_pack_scale_e8m0"}
    selected = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted
    ]
    assert len(selected) == 2
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), source, "exec"), namespace)
    return namespace["_pack_w_interleave"], namespace["_pack_scale_e8m0"]


def _fp4(codes):
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    return values[(codes & 7).long()] * torch.where(codes & 8 != 0, -1.0, 1.0)


def _decode_raw(weight, scale):
    codes = torch.stack((weight & 15, weight >> 4), -1).flatten(-2)
    return _fp4(codes) * torch.exp2(scale.float() - 127).repeat_interleave(32, -1)


def _decode_packed(weight, scale):
    e, k8, n = weight.shape
    shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], dtype=torch.int64)
    words = weight.transpose(1, 2).long().reshape(e, n, k8 // 16, 16)
    codes = ((words[..., None] >> shifts) & 15).transpose(-1, -2).reshape(e, n, k8 * 8)
    return _fp4(codes) * scale.transpose(1, 2).float().repeat_interleave(32, -1)


def _activation(out, inp, clamp):
    gate, up = inp.float().chunk(2, -1)
    if clamp is not None:
        gate, up = gate.clamp(max=clamp), up.clamp(-clamp, clamp)
    out.copy_(torch.nn.functional.silu(gate) * up)


@pytest.mark.parametrize("m", [1, 4])
@pytest.mark.parametrize("clamp", [None, 0.1])
@pytest.mark.parametrize("weight_input", [False, True])
def test_split_gemms_match_raw_mxfp4_reference(raw, m, clamp, weight_input):
    pack_w, pack_s = _frozen_packers()
    w1, w2, s1, s2 = raw
    packed = (
        pack_w(w1, 128),
        pack_w(w2, 128),
        pack_s(s1, torch.bfloat16),
        pack_s(s2, torch.bfloat16),
    )
    dense1, dense2 = _decode_raw(w1, s1), _decode_raw(w2, s2)
    torch.testing.assert_close(
        _decode_packed(packed[0], packed[2]), dense1, rtol=0, atol=0
    )
    torch.testing.assert_close(
        _decode_packed(packed[1], packed[3]), dense2, rtol=0, atol=0
    )
    x = torch.randn(m, 128).to(torch.bfloat16)
    ids = torch.tensor([[0, 2], [2, 1], [1, 0], [2, 0]][:m])
    weights = torch.tensor([[0.3, 0.7]] * m)
    calls = []

    def gemm(**kwargs):
        calls.append(kwargs)
        dense = _decode_packed(kwargs["B"], kwargs["B_scale"])
        for token in range(m):
            for route in range(2):
                row = token if kwargs["top_k"] == 2 else token * 2 + route
                y = dense[ids[token, route]] @ kwargs["A"][row].float()
                if kwargs["mul_routed_weight"]:
                    y *= kwargs["topk_weights"][token, route]
                kwargs["C"].reshape(m * 2, -1)[token * 2 + route].copy_(y)

    fake_gems = SimpleNamespace(
        _select_w4a16_mxfp4_kernel_policy=lambda *a: SimpleNamespace(
            block_m=8, align_mode=0
        ),
        _align_mxfp4_tokens=lambda *a: (None, None, None),
        _invoke_w4a16_mxfp4_moe_gemm=gemm,
        moe_sum=lambda inp, out: out.copy_(inp.sum(1)),
    )
    output = torch.empty_like(x)
    adapter.execute_split_gemms(
        fake_gems,
        x,
        packed,
        weights,
        ids,
        output,
        lambda out, inp: _activation(out, inp, clamp),
        num_experts=3,
        apply_router_weight_on_input=weight_input,
        fold_scale=False,
        compute_type=None,
    )
    expected = torch.empty_like(x)
    for token in range(m):
        routes = []
        for route in range(2):
            expert = ids[token, route]
            first = dense1[expert] @ x[token].float()
            if weight_input:
                first *= weights[token, route]
            middle = torch.empty(128, dtype=x.dtype)
            _activation(middle, first.to(x.dtype), clamp)
            second = dense2[expert] @ middle.float()
            if not weight_input:
                second *= weights[token, route]
            routes.append(second.to(x.dtype))
        expected[token] = torch.stack(routes).sum(0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert len(calls) == 2
    assert calls[0]["mul_routed_weight"] is weight_input
    assert calls[1]["mul_routed_weight"] is not weight_input


def test_raw_layout_and_scale_contract(raw):
    assert adapter.validate_raw_weights(*raw, hidden_size=128, num_experts=3) == 128
    for bad in (
        (raw[0].transpose(1, 2).contiguous(), *raw[1:]),
        (*raw[:2], raw[2].float(), raw[3]),
    ):
        with pytest.raises(ValueError):
            adapter.validate_raw_weights(*bad, hidden_size=128, num_experts=3)
    with pytest.raises(ValueError, match="128"):
        adapter.validate_raw_weights(*raw, hidden_size=96, num_experts=3)
    with pytest.raises(ValueError):
        adapter.validate_raw_weights(*raw, hidden_size=128, num_experts=6)


def _moe():
    return SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            use_ep=False,
            ep_size=1,
            enable_eplb=False,
            use_batched_activation_format=False,
        ),
        num_local_experts=3,
        num_experts=3,
        w13_num_shards=2,
        is_act_and_mul=True,
        activation="silu",
        has_bias=False,
        is_lora_enabled=False,
        in_dtype=torch.bfloat16,
    )


def test_deployment_rejection_preserves_precision_and_parallelism():
    moe = _moe()
    assert adapter.deployment_rejection(moe) is None
    moe.moe_parallel_config.use_ep = True
    assert adapter.deployment_rejection(moe) is None
    moe.moe_parallel_config.enable_eplb = True
    assert "EPLB" in adapter.deployment_rejection(moe)
    moe = _moe()
    assert "quantization" in adapter.deployment_rejection(moe, activation_key=object())
    moe.num_local_experts = 1
    assert adapter.deployment_rejection(moe) is None
    moe.num_local_experts = 0
    assert "global and local" in adapter.deployment_rejection(moe)
    moe = _moe()
    moe.activation = "situ"
    assert adapter.deployment_rejection(moe) is None


def test_empty_tokens_skip_kernel():
    output = torch.empty((0, 128), dtype=torch.bfloat16)
    adapter.execute_split_gemms(
        None,
        output,
        (torch.empty((3, 16, 256)), None, None, None),
        torch.empty((0, 2)),
        torch.empty((0, 2)),
        output,
        None,
        num_experts=3,
        apply_router_weight_on_input=False,
        fold_scale=False,
        compute_type=None,
    )


def test_nonfinite_scale_is_rejected(raw):
    raw[2][0, 0, 0] = 255
    with pytest.raises(ValueError, match="NaN"):
        adapter.validate_raw_weights(*raw, hidden_size=128, num_experts=3)


def test_installer_keeps_original_loader_and_clamp(monkeypatch, raw):
    import sys
    from types import ModuleType

    def fake_module(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    class UpstreamMethod:
        pass

    method = UpstreamMethod()
    method.moe = _moe()
    method.activation_key = None
    method.moe_kernel = None
    method.hidden_size = 128
    method.num_experts = 3
    original_loader = object()
    method.create_weights = original_loader
    layer = SimpleNamespace(
        quant_method=method,
        w13_weight=raw[0],
        w2_weight=raw[1],
        w13_weight_scale=raw[2],
        w2_weight_scale=raw[3],
        swiglu_limit=10.0,
        expert_map=None,
        _expert_routing_tables=lambda: None,
    )
    runner = SimpleNamespace(routed_experts=layer)
    processed = []
    fake_experts = SimpleNamespace(
        process_weights_after_loading=lambda layer: processed.append(layer)
    )
    fake_module(
        "vllm.model_executor.layers.quantization.mxfp4", Mxfp4MoEMethod=UpstreamMethod
    )
    fake_module(
        "vllm.model_executor.layers.fused_moe.config",
        mxfp4_w4a16_moe_quant_config=lambda **kw: SimpleNamespace(**kw),
    )
    fake_module(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4",
        Mxfp4MoeBackend=SimpleNamespace(MARLIN="marlin"),
        make_mxfp4_moe_kernel=lambda **kw: SimpleNamespace(fused_experts=fake_experts),
    )
    fake_module("vllm.model_executor.utils", is_weights_pre_processed=lambda: False)
    monkeypatch.setattr(
        adapter,
        "_make_experts_class",
        lambda: SimpleNamespace(_supports_current_device=lambda: True),
    )
    monkeypatch.setenv("VLLM_FL_MXFP4_MARLIN", "1")
    assert adapter.maybe_install_mxfp4_marlin(runner)
    assert adapter.maybe_install_mxfp4_marlin(runner)  # Idempotent.
    assert method.create_weights is original_loader
    assert layer.w13_weight is raw[0]
    method.process_weights_after_loading(layer)
    assert processed == [layer]
    assert method.moe_quant_config.gemm1_clamp_limit == 10.0
    assert method.moe_quant_config.w1_scale is raw[2]
    layer.expert_map = torch.tensor([0, -1, -1])
    method.process_weights_after_loading(layer)
    assert processed == [layer, layer]


def test_standard_ep_global_to_local_routing():
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    ids = torch.tensor([[4, 0, 2, -1]])
    expert_map = torch.tensor([-1, 1, -1, -1, 0, -1])
    local_weights, local_ids = adapter.localize_routing(
        weights, ids, expert_map, local_experts=2, global_experts=6
    )
    torch.testing.assert_close(local_ids, torch.tensor([[0, 0, 0, 0]]))
    torch.testing.assert_close(local_weights, torch.tensor([[0.1, 0.0, 0.0, 0.0]]))
    with pytest.raises(RuntimeError, match="expert_map"):
        adapter.localize_routing(weights, ids, None, local_experts=2, global_experts=6)
    with pytest.raises(RuntimeError, match="shape"):
        adapter.localize_routing(
            weights, ids, expert_map[:2], local_experts=2, global_experts=6
        )


def test_standard_ep_partial_sums_equal_global_moe():
    # Include a non-contiguous local expert order and one missing route.
    ids = torch.tensor([[0, 3], [2, 1], [3, -1]])
    weights = torch.tensor([[0.3, 0.7], [0.4, 0.6], [0.8, 0.2]])
    global_values = torch.tensor([11.0, 23.0, 37.0, 41.0])
    maps = [torch.tensor([1, -1, 0, -1]), torch.tensor([-1, 0, -1, 1])]
    local_values = [
        global_values[torch.tensor([2, 0])],
        global_values[torch.tensor([1, 3])],
    ]
    partials = []
    for mapping, values in zip(maps, local_values):
        ws, local_ids = adapter.localize_routing(
            weights, ids, mapping, local_experts=2, global_experts=4
        )
        partials.append((ws * values[local_ids]).sum(1))
    expected = (
        weights * torch.where(ids >= 0, global_values[ids.clamp(min=0)], 0)
    ).sum(1)
    torch.testing.assert_close(sum(partials), expected)


def test_invalid_ep_routes_are_explicitly_zeroed_after_gemm():
    def gemm(**kw):
        kw["C"].fill_(7.0)
        if kw["C"].ndim == 3:
            kw["C"][:, 0] = 5.0
            kw["C"][:, 1] = float("nan")

    gems = SimpleNamespace(
        _select_w4a16_mxfp4_kernel_policy=lambda *a: SimpleNamespace(
            block_m=8, align_mode=0
        ),
        _align_mxfp4_tokens=lambda *a: (None, None, None),
        _invoke_w4a16_mxfp4_moe_gemm=gemm,
        moe_sum=lambda inp, out: out.copy_(inp.sum(1)),
    )
    x = torch.zeros(1, 128, dtype=torch.bfloat16)
    output = torch.empty_like(x)
    adapter.execute_split_gemms(
        gems,
        x,
        (torch.empty(2, 16, 256), None, None, None),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0, 0]]),
        output,
        lambda out, inp: out.zero_(),
        num_experts=2,
        apply_router_weight_on_input=False,
        fold_scale=False,
        compute_type=None,
        route_mask=torch.tensor([[True, False]]),
    )
    torch.testing.assert_close(output, torch.full_like(output, 5.0))
    _, ids, mask = adapter.localize_routing(
        torch.tensor([[0.2, 0.8]]),
        torch.tensor([[-1, 1]]),
        None,
        local_experts=2,
        global_experts=2,
        return_valid_mask=True,
    )
    torch.testing.assert_close(ids, torch.tensor([[0, 1]]))
    torch.testing.assert_close(mask, torch.tensor([[False, True]]))
