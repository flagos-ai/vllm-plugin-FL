# SPDX-License-Identifier: Apache-2.0
"""Real common-policy behavior with explicit compiler/numerical boundary stubs."""

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.manager import OpManager
from vllm_fl.dispatch.policy import SelectionPolicy, policy_context
from vllm_fl.dispatch.types import BackendImplKind
from vllm_fl.kernels.glm5_next import indexer_backend, provider

OP = "mhc_pre_broadcast_tilelang"
ADAPTER = "vllm_fl.kernels.glm5_next.mhc_broadcast_tilelang"
UPSTREAM = "vllm.model_executor.kernels.mhc.tilelang"
BACKPORT = "vllm_fl._vendor.mhc_broadcast_tilelang"


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
        kwargs["_preflight"] = True
    return adapter.mhc_pre_broadcast_tilelang(*args, **kwargs), args, kwargs


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
