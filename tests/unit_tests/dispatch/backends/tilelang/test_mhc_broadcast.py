# SPDX-License-Identifier: Apache-2.0
"""Mhc broadcast contracts and regression coverage."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.manager import OpManager
from vllm_fl.dispatch.policy import SelectionPolicy, policy_context
from vllm_fl.dispatch.types import BackendImplKind
from vllm_fl.models.glm5 import indexer_backend, provider

OP = "mhc_pre_broadcast_tilelang"

ADAPTER = "vllm_fl.dispatch.backends.tilelang.mhc_broadcast"

UPSTREAM = "vllm.model_executor.kernels.mhc.tilelang"

BACKPORT = "vllm_fl.dispatch.backends.tilelang.impl.mhc_broadcast"


def _upstream_api(calls, *, error=None):
    def native(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        *,
        norm_weight,
        norm_eps,
        fn_broadcast,
    ):
        calls.append(
            (
                (
                    residual,
                    fn,
                    hc_scale,
                    hc_base,
                    rms_eps,
                    hc_pre_eps,
                    hc_sinkhorn_eps,
                    hc_post_mult_value,
                    sinkhorn_repeat,
                ),
                {
                    "norm_weight": norm_weight,
                    "norm_eps": norm_eps,
                    "fn_broadcast": fn_broadcast,
                },
            )
        )
        if error is not None:
            raise error
        return "upstream"

    return native


@pytest.fixture
def load_adapter(monkeypatch):
    monkeypatch.setattr(indexer_backend, "use_nvidia_reference", lambda: True)
    monkeypatch.setattr(provider, "use_nvidia_reference", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST", raising=False)
    monkeypatch.delenv("VLLM_FL_FLAGOS_BLACKLIST", raising=False)
    # These are policy/forwarding tests with object sentinels. The separate
    # GPU suite exercises the real typed custom-op and its compiler boundary.
    monkeypatch.setattr(
        "vllm.utils.torch_utils.direct_register_custom_op", lambda **kwargs: None
    )
    original = importlib.import_module
    imported = []
    backport_calls = []

    def backport(*args, **kwargs):
        backport_calls.append((args, kwargs))
        return "backport"

    def load(candidate=None, *, upstream_missing=False, deep_gemm=True, tilelang=True):
        # Only optional capability and numerical boundaries are mocked.
        # OpManager, policy contexts, registry and OperatorBinding are real.
        modules = {
            "vllm.utils.deep_gemm": SimpleNamespace(
                is_deep_gemm_supported=lambda: deep_gemm,
                tf32_hc_prenorm_gemm=lambda *a, **k: None,
            ),
            "flashinfer.comm": SimpleNamespace(),
            "tilelang": SimpleNamespace(),
            UPSTREAM: SimpleNamespace(mhc_pre_broadcast_tilelang=candidate),
            BACKPORT: SimpleNamespace(mhc_pre_broadcast_tilelang=backport),
        }

        def import_module(name, *args, **kwargs):
            imported.append(name)
            if name == UPSTREAM and upstream_missing:
                raise ModuleNotFoundError(name=name)
            if name == "tilelang" and not tilelang:
                raise ModuleNotFoundError(name=name)
            if name in modules:
                return modules[name]
            return original(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, ADAPTER, raising=False)
        monkeypatch.setattr(importlib, "import_module", import_module)
        adapter = original(ADAPTER)
        backend = indexer_backend.Glm5NextIndexerBackend()
        backend._manager = OpManager(register_builtins=False)
        monkeypatch.setattr(adapter, "INDEXER_BACKEND", backend)
        return adapter, backend

    yield load, imported, backport_calls
    sys.modules.pop(ADAPTER, None)


def _invoke(adapter, *, preflight=False):
    args = tuple(object() for _ in range(4)) + (1e-5, 1e-6, 1e-6, 2.5, 20)
    kwargs = dict(norm_weight=object(), norm_eps=1e-5, fn_broadcast=object())
    if preflight:
        return adapter.preflight_mhc_broadcast(), args, kwargs
    return adapter._mhc_pre_broadcast(*args, **kwargs), args, kwargs


def test_upstream_api_has_priority_and_preserves_forwarding(load_adapter):
    load, imported, backport_calls = load_adapter
    calls = []
    native = _upstream_api(calls)
    adapter, backend = load(native)
    with policy_context(SelectionPolicy()):
        result, args, kwargs = _invoke(adapter)
        info, _, _ = _invoke(adapter, preflight=True)
    assert result == "upstream"
    assert calls == [(args, kwargs)]
    assert BACKPORT not in imported
    assert backport_calls == []
    assert not hasattr(native, "_is_available")
    assert info["selected"] == "glm5.cuda"
    implementations = backend._manager.registry.get_implementations(OP)
    assert len(implementations) == 1
    assert implementations[0].kind == BackendImplKind.VENDOR
    assert implementations[0].vendor == "cuda"


@pytest.mark.parametrize("invalid", ["missing", "not_callable", "signature", "module"])
def test_absent_or_incompatible_upstream_api_loads_backport(load_adapter, invalid):
    load, imported, backport_calls = load_adapter
    candidate = None
    if invalid == "not_callable":
        candidate = object()
    elif invalid == "signature":
        candidate = lambda residual, fn: pytest.fail("incompatible API was called")
    adapter, _ = load(candidate, upstream_missing=invalid == "module")
    with policy_context(SelectionPolicy()):
        result, args, kwargs = _invoke(adapter)
    assert result == "backport"
    assert BACKPORT in imported
    assert backport_calls == [(args, kwargs)]


@pytest.mark.parametrize("prefer", ["flagos", "vendor", "reference"])
def test_global_preference_keeps_backport_classified_as_vendor(load_adapter, prefer):
    load, _, _ = load_adapter
    adapter, _ = load()
    with policy_context(SelectionPolicy(prefer=prefer)):
        info, _, _ = _invoke(adapter, preflight=True)
        result, _, _ = _invoke(adapter)
    assert result == "backport"
    assert info["candidates"] == ["glm5.cuda"]


@pytest.mark.parametrize(
    "policy,permitted",
    [
        (SelectionPolicy.from_dict(allow_vendors={"cuda"}), True),
        (SelectionPolicy.from_dict(allow_vendors={"ascend"}), False),
        (SelectionPolicy.from_dict(deny_vendors={"cuda"}), False),
        (SelectionPolicy.from_dict(per_op_order={OP: ["vendor:cuda"]}), True),
        (SelectionPolicy.from_dict(per_op_order={OP: ["flagos"]}), False),
        (SelectionPolicy.from_dict(per_op_order={OP: ["reference"]}), False),
    ],
)
def test_vendor_filters_and_explicit_order_apply(load_adapter, policy, permitted):
    load, _, backport_calls = load_adapter
    adapter, _ = load()
    with policy_context(policy):
        if permitted:
            result, _, _ = _invoke(adapter)
            assert result == "backport"
        else:
            with pytest.raises(RuntimeError, match=OP):
                _invoke(adapter)
    assert len(backport_calls) == int(permitted)


@pytest.mark.parametrize("strict", [False, True])
def test_execution_failure_never_loads_or_retries_backport(load_adapter, strict):
    load, imported, backport_calls = load_adapter
    calls = []
    error = RuntimeError("CUDA execution failure")
    adapter, backend = load(_upstream_api(calls, error=error))
    with (
        policy_context(SelectionPolicy(strict=strict)),
        pytest.raises(RuntimeError) as caught,
    ):
        _invoke(adapter)
    assert caught.value is error
    assert len(calls) == 1
    assert BACKPORT not in imported
    assert backport_calls == []
    assert backend._manager.get_failed_impls(OP) == {}


@pytest.mark.parametrize("missing", ["nvidia", "cuda", "deep_gemm", "tilelang"])
def test_unavailable_capability_does_not_load_backport(
    load_adapter, monkeypatch, missing
):
    load, imported, backport_calls = load_adapter
    if missing == "nvidia":
        monkeypatch.setattr(provider, "use_nvidia_reference", lambda: False)
    elif missing == "cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ImportError):
        load(deep_gemm=missing != "deep_gemm", tilelang=missing != "tilelang")
    assert BACKPORT not in imported
    assert UPSTREAM not in imported
    assert backport_calls == []
    if missing != "tilelang":
        assert "tilelang" not in imported


def _require_cuda_broadcast_kernel():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from vllm.utils.deep_gemm import is_deep_gemm_supported

    if not is_deep_gemm_supported():
        pytest.skip("requires the DeepGEMM TF32 mHC projection")

    from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_tilelang

    from vllm_fl.dispatch.backends.tilelang.mhc_broadcast import (
        mhc_pre_broadcast_tilelang,
    )

    return mhc_pre_tilelang, mhc_pre_broadcast_tilelang


def _make_inputs(tokens: int):
    torch.manual_seed(20260827 + tokens)
    hidden_size, streams = 4096, 4
    mix_size = streams * (2 + streams)
    fn = (
        torch.randn(
            mix_size,
            streams * hidden_size,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.01
    ).contiguous()
    return {
        "residual": torch.randn(
            tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        ),
        "fn": fn,
        "fn_broadcast": fn.view(mix_size, streams, hidden_size).sum(dim=1).contiguous(),
        "scale": torch.tensor([0.8, 0.7, 0.6], device="cuda", dtype=torch.float32),
        "base": (
            torch.randn(mix_size, device="cuda", dtype=torch.float32) * 0.1
        ).contiguous(),
        "norm_weight": (
            1 + torch.randn(hidden_size, device="cuda", dtype=torch.float32) * 0.05
        )
        .to(torch.bfloat16)
        .contiguous(),
        "streams": streams,
    }


def _generic_outputs(mhc_pre_tilelang, values):
    expanded = (
        values["residual"].unsqueeze(1).expand(-1, values["streams"], -1).contiguous()
    )
    post, comb, layer_input = mhc_pre_tilelang(
        expanded,
        values["fn"],
        values["scale"],
        values["base"],
        1e-5,
        1e-6,
        1e-6,
        2.5,
        20,
        norm_weight=values["norm_weight"],
        norm_eps=1e-5,
    )
    return expanded, post, comb, layer_input


def _broadcast_outputs(mhc_pre_broadcast_tilelang, values):
    return mhc_pre_broadcast_tilelang(
        values["residual"],
        values["fn"],
        values["scale"],
        values["base"],
        1e-5,
        1e-6,
        1e-6,
        2.5,
        20,
        norm_weight=values["norm_weight"],
        norm_eps=1e-5,
        fn_broadcast=values["fn_broadcast"],
    )


def _assert_outputs_close(reference, actual) -> None:
    assert torch.equal(reference[0], actual[0])
    torch.testing.assert_close(reference[1], actual[1], rtol=2e-3, atol=1e-3)
    torch.testing.assert_close(reference[2], actual[2], rtol=2e-3, atol=1e-3)
    torch.testing.assert_close(reference[3], actual[3], rtol=2e-2, atol=4e-2)


@pytest.mark.parametrize("tokens", [1, 24, 257, 1024])
def test_mhc_broadcast_matches_materialized_reference(tokens: int) -> None:
    generic, broadcast = _require_cuda_broadcast_kernel()
    values = _make_inputs(tokens)

    reference = _generic_outputs(generic, values)
    actual = _broadcast_outputs(broadcast, values)
    repeated = _broadcast_outputs(broadcast, values)
    torch.cuda.synchronize()

    _assert_outputs_close(reference, actual)
    for lhs, rhs in zip(actual, repeated):
        assert torch.equal(lhs, rhs)


def test_mhc_broadcast_cuda_graph_replay() -> None:
    generic, broadcast = _require_cuda_broadcast_kernel()
    values = _make_inputs(24)

    # Compile both DeepGEMM and TileLang before stream capture.
    _broadcast_outputs(broadcast, values)
    torch.cuda.synchronize()

    static_residual = values["residual"].clone()
    graph_values = dict(values, residual=static_residual)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = _broadcast_outputs(broadcast, graph_values)

    next_residual = torch.randn_like(static_residual)
    static_residual.copy_(next_residual)
    for output in graph_outputs:
        output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    replayed = tuple(value.clone() for value in graph_outputs)

    reference_values = dict(values, residual=next_residual)
    reference = _generic_outputs(generic, reference_values)
    torch.cuda.synchronize()
    _assert_outputs_close(reference, replayed)


@pytest.mark.parametrize("strided", [False, True])
def test_mhc_broadcast_converts_norm_weight_like_upstream(strided: bool) -> None:
    generic, broadcast = _require_cuda_broadcast_kernel()
    values = _make_inputs(24)
    reference = _generic_outputs(generic, values)
    norm_weight = values["norm_weight"].to(torch.float32)
    if strided:
        storage = torch.empty(norm_weight.numel(), 2, device="cuda")
        storage[:, 0].copy_(norm_weight)
        norm_weight = storage[:, 0]
        assert not norm_weight.is_contiguous()
    actual = _broadcast_outputs(broadcast, dict(values, norm_weight=norm_weight))
    torch.cuda.synchronize()
    _assert_outputs_close(reference, actual)


def test_mhc_broadcast_empty_tokens() -> None:
    _, broadcast = _require_cuda_broadcast_kernel()
    values = _make_inputs(0)
    actual = _broadcast_outputs(broadcast, values)
    torch.cuda.synchronize()
    expected_shapes = [(0, 4, 4096), (0, 4, 1), (0, 4, 4), (0, 4096)]
    expected_dtypes = [torch.bfloat16, torch.float32, torch.float32, torch.bfloat16]
    for output, shape, dtype in zip(actual, expected_shapes, expected_dtypes):
        assert output.shape == shape
        assert output.dtype == dtype
        assert output.device == values["residual"].device


def test_mhc_broadcast_fullgraph_compile_and_replay() -> None:
    generic, broadcast = _require_cuda_broadcast_kernel()
    values = _make_inputs(24)

    def forward(residual):
        return _broadcast_outputs(broadcast, dict(values, residual=residual))

    compiled = torch.compile(forward, fullgraph=True, dynamic=True)
    actual = compiled(values["residual"])
    reference = _generic_outputs(generic, values)
    torch.cuda.synchronize()
    _assert_outputs_close(reference, actual)

    # Warm compilation before capture, then change the input and poison all
    # outputs so an empty graph or cached result cannot satisfy the check.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = compiled(values["residual"])
    values["residual"].copy_(torch.randn_like(values["residual"]))
    for output in graph_outputs:
        output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    replayed = tuple(output.clone() for output in graph_outputs)
    reference = _generic_outputs(generic, values)
    torch.cuda.synchronize()
    _assert_outputs_close(reference, replayed)

    # The same dynamic compiled entry must preserve fake metadata at zero
    # and at a different nonempty token count.
    for tokens in (0, 1, 257):
        next_values = dict(
            values,
            residual=torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16),
        )
        result = compiled(next_values["residual"])
        if tokens == 0:
            # The generic materialized vLLM entry divides by zero at T=0;
            # the broadcast entry explicitly defines these empty outputs.
            for output, shape, dtype in zip(
                result,
                [(0, 4, 4096), (0, 4, 1), (0, 4, 4), (0, 4096)],
                [torch.bfloat16, torch.float32, torch.float32, torch.bfloat16],
            ):
                assert output.shape == shape
                assert output.dtype == dtype
                assert output.device == next_values["residual"].device
            continue
        reference = _generic_outputs(generic, next_values)
        torch.cuda.synchronize()
        _assert_outputs_close(reference, result)


@pytest.mark.parametrize(
    "tokens,strided", [(0, False), (0, True), (1, False), (24, False)]
)
def test_mhc_broadcast_fake_matches_actual_metadata(tokens, strided) -> None:
    _, broadcast = _require_cuda_broadcast_kernel()
    from torch._subclasses.fake_tensor import FakeTensorMode

    values = _make_inputs(tokens)
    if strided:
        storage = torch.empty(tokens, 8192, device="cuda", dtype=torch.bfloat16)
        storage[:, ::2].copy_(values["residual"])
        values["residual"] = storage[:, ::2]
    actual = _broadcast_outputs(broadcast, values)
    torch.cuda.synchronize()
    with FakeTensorMode() as mode:
        fake_values = {
            key: mode.from_tensor(value) if isinstance(value, torch.Tensor) else value
            for key, value in values.items()
        }
        fake = _broadcast_outputs(broadcast, fake_values)
    for real, meta in zip(actual, fake):
        assert (real.shape, real.dtype, real.device, real.stride()) == (
            meta.shape,
            meta.dtype,
            meta.device,
            meta.stride(),
        )
