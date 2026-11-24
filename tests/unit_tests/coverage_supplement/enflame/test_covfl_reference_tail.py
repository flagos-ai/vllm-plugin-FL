# Copyright (c) 2026 BAAI. All rights reserved.

"""Tail-line coverage for the reference backend, GCU registration and base ABC.

The tests only use pytest + mock + stdlib and a real ``torch``; device/kernel
work is always mocked or replaced with a sentinel callable so nothing is
launched on hardware and no subprocess is spawned.
"""

from __future__ import annotations

import builtins
import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_fl.dispatch.backends.base import Backend
from vllm_fl.dispatch.backends.reference.reference import ReferenceBackend
from vllm_fl.dispatch.registry import OpRegistry


@pytest.fixture(autouse=True)
def _reset_reference_availability():
    """Keep the class-level availability cache from leaking between tests."""
    yield
    ReferenceBackend._available = None


# ---------------------------------------------------------------------------
# reference/reference.py
# ---------------------------------------------------------------------------


def test_reference_name_vendor_and_cached_availability(monkeypatch):
    monkeypatch.setattr(ReferenceBackend, "_available", None, raising=False)
    backend = ReferenceBackend()

    assert backend.name == "reference"
    assert backend.vendor is None
    # Real torch is importable in the test environment.
    assert backend.is_available() is True
    # The second call must hit the cached class attribute.
    monkeypatch.setattr(ReferenceBackend, "_available", True, raising=False)
    assert backend.is_available() is True


def test_reference_is_available_handles_missing_torch(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(ReferenceBackend, "_available", None, raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocking_import)

    assert ReferenceBackend().is_available() is False


def test_reference_dynamic_per_token_quant_delegates(monkeypatch):
    sentinel_out = object()
    sentinel_scale = object()
    calls = []
    fake_module = types.ModuleType("vllm_fl.quantization.w8a8.reference")

    def _fake_quant(x):
        calls.append(x)
        return sentinel_out, sentinel_scale

    fake_module.dynamic_per_token_quant_int8 = _fake_quant
    monkeypatch.setitem(
        sys.modules, "vllm_fl.quantization.w8a8.reference", fake_module
    )

    x = object()
    out, scale = ReferenceBackend().dynamic_per_token_quant_int8(x)
    assert (out, scale) == (sentinel_out, sentinel_scale)
    assert calls == [x]


def test_reference_activation_wrappers_delegate():
    backend = ReferenceBackend()
    x = torch.randn(2, 4, 8)
    d = x.shape[-1] // 2

    silu = backend.silu_and_mul(None, x)
    expected_silu = torch.nn.functional.silu(x[..., :d]) * x[..., d:]
    torch.testing.assert_close(silu, expected_silu)

    gelu = backend.gelu_and_mul(None, x)
    expected_gelu = torch.nn.functional.gelu(x[..., :d]) * x[..., d:]
    torch.testing.assert_close(gelu, expected_gelu)


def test_reference_gelu_and_mul_uses_obj_approximation():
    from vllm_fl.dispatch.backends.reference.impl.activation import (
        gelu_and_mul_torch,
    )

    x = torch.randn(3, 10)
    d = x.shape[-1] // 2
    obj = SimpleNamespace(approximate="tanh")
    result = gelu_and_mul_torch(obj, x)
    expected = torch.nn.functional.gelu(x[..., :d], approximate="tanh") * x[..., d:]
    torch.testing.assert_close(result, expected)


def test_reference_rms_norm_wrapper_with_and_without_residual():
    backend = ReferenceBackend()
    x = torch.randn(2, 6, dtype=torch.float32)
    weight = torch.ones(6)
    obj = SimpleNamespace(weight=weight, variance_epsilon=1e-6, pass_weight=True)

    out = backend.rms_norm(obj, x)
    assert out.shape == x.shape

    residual = torch.ones_like(x)
    out2, res2 = backend.rms_norm(obj, x, residual)
    assert out2.shape == x.shape
    assert res2.shape == x.shape


def test_reference_rotary_embedding_wrapper_delegates():
    backend = ReferenceBackend()
    tokens, heads, head_dim = 3, 2, 4
    q = torch.randn(tokens, heads, head_dim)
    k = torch.randn(tokens, heads, head_dim)
    cos = torch.ones(8, head_dim)
    sin = torch.zeros(8, head_dim)
    pos = torch.arange(tokens, dtype=torch.long)

    q_out, k_out = backend.rotary_embedding(
        None, q, k, cos, sin, pos, rotary_interleaved=False, inplace=True
    )
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_reference_attention_backend_paths(monkeypatch):
    flash = SimpleNamespace(get_path=lambda: "flash.path")
    mla = SimpleNamespace(get_path=lambda: "mla.path")
    sparse = SimpleNamespace(get_path=lambda: "sparse.path")

    registry_mod = types.ModuleType("vllm.v1.attention.backends.registry")
    registry_mod.AttentionBackendEnum = SimpleNamespace(
        FLASH_ATTN=flash,
        FLASHMLA=mla,
        FLASHMLA_SPARSE=sparse,
    )
    monkeypatch.setitem(
        sys.modules, "vllm.v1.attention.backends.registry", registry_mod
    )

    backend = ReferenceBackend()
    assert backend.attention_backend() == "flash.path"
    assert backend.attention_backend(use_mla=True) == "mla.path"
    assert backend.attention_backend(use_mla=True, use_sparse=True) == "sparse.path"


def test_reference_topk_softmax_wrapper_delegates():
    backend = ReferenceBackend()
    tokens, experts, topk = 2, 4, 2
    gating = torch.randn(tokens, experts)
    weights = torch.empty(tokens, topk)
    indices = torch.empty(tokens, topk, dtype=torch.int32)
    token_expert = torch.empty(tokens, topk, dtype=torch.int32)
    bias = torch.zeros(experts)

    out_w, out_i = backend.topk_softmax(
        weights, indices, token_expert, gating, True, bias
    )
    assert out_w is weights
    assert out_i is indices
    # Renormalized weights sum to one per token.
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(tokens))


def test_reference_moe_sum():
    backend = ReferenceBackend()
    inp = torch.randn(3, 2, 4)
    out = torch.empty(3, 4)
    backend.moe_sum(inp, out)
    torch.testing.assert_close(out, inp.sum(dim=1))


def test_reference_invoke_fused_moe_triton_kernel_delegates(monkeypatch):
    recorder = MagicMock()
    fake_mod = types.ModuleType(
        "vllm.model_executor.layers.fused_moe.fused_moe"
    )
    fake_mod.invoke_fused_moe_triton_kernel = recorder
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.fused_moe",
        fake_mod,
    )

    args = [MagicMock(name=f"arg{i}") for i in range(18)]
    ReferenceBackend().invoke_fused_moe_triton_kernel(
        *args, block_shape="bs", B_bias="bias"
    )

    recorder.assert_called_once_with(*args, block_shape="bs", B_bias="bias")


def test_reference_registry_get_implementation_delegation():
    """Registered REFERENCE impls are retrievable and callable wrappers work."""
    from vllm_fl.dispatch.backends.reference.register_ops import (
        register_builtins as register_reference,
    )

    registry = OpRegistry()
    register_reference(registry)
    impl = registry.get_implementation("silu_and_mul", "reference.torch")
    assert impl is not None
    assert impl.is_available() is True

    x = torch.randn(2, 8)
    d = x.shape[-1] // 2
    out = impl.fn(None, x)
    expected = torch.nn.functional.silu(x[..., :d]) * x[..., d:]
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# reference/impl/*.py
# ---------------------------------------------------------------------------


def test_impl_quantization_delegates(monkeypatch):
    from vllm_fl.dispatch.backends.reference.impl import quantization

    fake_module = types.ModuleType("vllm_fl.quantization.w8a8.reference")
    fake_module.dynamic_per_token_quant_int8 = lambda x: ("q", "s")
    monkeypatch.setitem(
        sys.modules, "vllm_fl.quantization.w8a8.reference", fake_module
    )
    assert quantization.dynamic_per_token_quant_int8_torch(object()) == ("q", "s")


def test_impl_activation_gelu_modes():
    from vllm_fl.dispatch.backends.reference.impl.activation import (
        gelu_and_mul_torch,
        silu_and_mul_torch,
    )

    x = torch.randn(2, 6)
    d = x.shape[-1] // 2
    assert silu_and_mul_torch(None, x).shape == (2, d)

    none_result = gelu_and_mul_torch(None, x)
    expected_none = torch.nn.functional.gelu(x[..., :d], approximate="none") * x[..., d:]
    torch.testing.assert_close(none_result, expected_none)

    tanh_result = gelu_and_mul_torch(SimpleNamespace(approximate="tanh"), x)
    expected_tanh = torch.nn.functional.gelu(x[..., :d], approximate="tanh") * x[..., d:]
    torch.testing.assert_close(tanh_result, expected_tanh)


def test_impl_fused_moe_bias_and_renormalize():
    from vllm_fl.dispatch.backends.reference.impl.fused_moe import topk_softmax

    tokens, experts, topk = 3, 5, 2
    gating = torch.randn(tokens, experts)
    weights = torch.empty(tokens, topk)
    indices = torch.empty(tokens, topk, dtype=torch.int32)
    token_expert = torch.empty(tokens, topk, dtype=torch.int32)
    bias = torch.tensor([0.0, 10.0, 0.0, 0.0, 0.0])

    topk_softmax(
        weights,
        indices,
        token_expert,
        gating,
        renormalize=True,
        e_score_correction_bias=bias,
    )
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(tokens))
    # The biased expert is always selected, but weights come from raw scores.
    assert (indices == 1).any(dim=-1).all()
    # Slot-major token index layout: token_idx + slot * num_tokens.
    rows = torch.arange(tokens).view(-1, 1)
    slots = torch.arange(topk).view(1, -1)
    assert torch.equal(token_expert, (rows + slots * tokens).to(torch.int32))


def test_impl_fused_moe_without_bias_or_renormalize():
    from vllm_fl.dispatch.backends.reference.impl.fused_moe import topk_softmax

    tokens, experts, topk = 2, 4, 2
    gating = torch.randn(tokens, experts)
    weights = torch.empty(tokens, topk)
    indices = torch.empty(tokens, topk, dtype=torch.int32)
    token_expert = torch.empty(tokens, topk, dtype=torch.int32)

    topk_softmax(weights, indices, token_expert, gating)
    assert weights.shape == (tokens, topk)
    assert indices.shape == (tokens, topk)


def _rotary_inputs(dim, tokens=3, heads=2, batch=None, half_cache=False, positions_1d=True):
    cache_dim = dim // 2 if half_cache else dim
    max_pos = 8
    cos = torch.randn(max_pos, cache_dim)
    sin = torch.randn(max_pos, cache_dim)
    if batch is None:
        q = torch.randn(tokens, heads, dim)
        k = torch.randn(tokens, heads, dim)
    else:
        q = torch.randn(batch, heads, tokens, dim)
        k = torch.randn(batch, heads, tokens, dim)
    if positions_1d:
        pos = torch.arange(tokens, dtype=torch.long)
    else:
        pos = torch.arange(batch * tokens, dtype=torch.long).view(batch, tokens)
    return q, k, cos, sin, pos


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(batch=None, half_cache=False, positions_1d=True, interleaved=False),
        dict(batch=2, half_cache=False, positions_1d=True, interleaved=False),
        dict(batch=None, half_cache=True, positions_1d=True, interleaved=True),
        dict(batch=None, half_cache=True, positions_1d=True, interleaved=False),
    ],
)
def test_impl_rotary_shape_and_cache_variants(kwargs):
    from vllm_fl.dispatch.backends.reference.impl import rotary

    head_dim = 4
    q, k, cos, sin, pos = _rotary_inputs(
        head_dim,
        batch=kwargs["batch"],
        half_cache=kwargs["half_cache"],
        positions_1d=kwargs["positions_1d"],
    )
    q_out, k_out = rotary.rotary_embedding_torch(
        None,
        q,
        k,
        cos,
        sin,
        pos,
        rotary_interleaved=kwargs["interleaved"],
        inplace=True,
    )
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


# ---------------------------------------------------------------------------
# vendor/gcu/gcu.py
# ---------------------------------------------------------------------------


def test_gcu_backend_name_vendor():
    from vllm_fl.dispatch.backends.vendor.gcu.gcu import GCUBackend

    backend = GCUBackend()
    assert backend.name == "gcu"
    assert backend.vendor == "gcu"


def test_gcu_is_available_without_runtime(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.gcu.gcu import GCUBackend

    monkeypatch.setattr(GCUBackend, "_available", None, raising=False)
    monkeypatch.delattr(torch, "gcu", raising=False)
    assert GCUBackend().is_available() is False


def test_gcu_is_available_unavailable_and_no_devices(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.gcu.gcu import GCUBackend

    monkeypatch.setattr(GCUBackend, "_available", None, raising=False)
    monkeypatch.setattr(
        torch,
        "gcu",
        SimpleNamespace(is_available=lambda: False, device_count=lambda: 1),
        raising=False,
    )
    assert GCUBackend().is_available() is False

    monkeypatch.setattr(GCUBackend, "_available", None, raising=False)
    monkeypatch.setattr(
        torch,
        "gcu",
        SimpleNamespace(is_available=lambda: True, device_count=lambda: 0),
        raising=False,
    )
    assert GCUBackend().is_available() is False


def test_gcu_is_available_caches_positive_result(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.gcu.gcu import GCUBackend

    calls = []

    def _is_available():
        calls.append("is_available")
        return True

    def _device_count():
        calls.append("device_count")
        return 1

    monkeypatch.setattr(GCUBackend, "_available", None, raising=False)
    monkeypatch.setattr(
        torch,
        "gcu",
        SimpleNamespace(is_available=_is_available, device_count=_device_count),
        raising=False,
    )

    backend = GCUBackend()
    assert backend.is_available() is True
    assert backend.is_available() is True
    assert calls == ["is_available", "device_count"]


def test_gcu_delegates_ops_to_impl(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.gcu import gcu as gcu_module

    activation = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.gcu.impl.activation"
    )
    normalization = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.gcu.impl.normalization"
    )
    rotary = importlib.import_module(
        "vllm_fl.dispatch.backends.vendor.gcu.impl.rotary"
    )

    monkeypatch.setattr(
        activation, "silu_and_mul_gcu", lambda obj, x: ("silu", obj, x)
    )
    monkeypatch.setattr(
        normalization, "rms_norm_gcu", lambda obj, x, residual: ("rms", obj, x, residual)
    )
    monkeypatch.setattr(
        rotary,
        "rotary_embedding_gcu",
        lambda obj, q, k, c, s, p, rotary_interleaved, inplace: (
            "rot",
            obj,
            q,
            k,
            c,
            s,
            p,
            rotary_interleaved,
            inplace,
        ),
    )

    backend = gcu_module.GCUBackend()
    obj = object()
    x = object()
    assert backend.silu_and_mul(obj, x) == ("silu", obj, x)
    assert backend.rms_norm(obj, x) == ("rms", obj, x, None)
    assert backend.rms_norm(obj, x, "res") == ("rms", obj, x, "res")

    q, k, c, s, p = (object() for _ in range(5))
    result = backend.rotary_embedding(
        obj, q, k, c, s, p, rotary_interleaved=True, inplace=False
    )
    assert result == ("rot", obj, q, k, c, s, p, True, False)


def test_gcu_attention_backend_paths():
    from vllm_fl.dispatch.backends.vendor.gcu.gcu import GCUBackend

    backend = GCUBackend()
    path = backend.attention_backend()
    assert path.endswith("AttentionGCUBackend")

    with pytest.raises(NotImplementedError, match="MLA"):
        backend.attention_backend(use_mla=True)
    with pytest.raises(NotImplementedError, match="sparse"):
        backend.attention_backend(use_mla=True, use_sparse=True)


# ---------------------------------------------------------------------------
# vendor/gcu/register_ops.py
# ---------------------------------------------------------------------------


def test_gcu_bind_is_available_wraps_function():
    from vllm_fl.dispatch.backends.vendor.gcu.register_ops import _bind_is_available

    def fn(a, b=1):
        return a + b

    availability = MagicMock(return_value=True)
    wrapped = _bind_is_available(fn, availability)

    assert wrapped(2, b=3) == 5
    assert wrapped._is_available is availability
    assert wrapped._is_available() is True
    assert wrapped.__wrapped__ is fn


def _block_vllm_c(monkeypatch):
    """Make the bare ``import vllm._C`` in the fallback path fail."""
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "vllm._C":
            raise ImportError("vllm._C blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def _allow_vllm_c(monkeypatch):
    """Make the bare ``import vllm._C`` succeed without a real extension."""
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "vllm._C":
            return types.ModuleType("vllm._C")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def test_gcu_register_fallbacks_skips_when_C_present(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.gcu import register_ops

    _allow_vllm_c(monkeypatch)
    library = MagicMock(side_effect=AssertionError("Library must not be opened"))
    monkeypatch.setattr(torch.library, "Library", library)

    register_ops._register_c_ops_privateuse1_fallbacks()
    library.assert_not_called()


def test_gcu_register_fallbacks_handles_library_error(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.gcu import register_ops

    _block_vllm_c(monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("cannot open fragment")

    monkeypatch.setattr(torch.library, "Library", _boom)
    # Should swallow the error and log it instead of propagating.
    register_ops._register_c_ops_privateuse1_fallbacks()


def test_gcu_register_fallbacks_installs_privateuse1(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.gcu import register_ops

    _block_vllm_c(monkeypatch)

    instances = []

    class _FakeLibrary:
        def __init__(self, *args):
            self.args = args
            self.impl = MagicMock()
            instances.append(self)

    monkeypatch.setattr(torch.library, "Library", _FakeLibrary)

    applied = lambda *a, **k: None
    fake_ops = types.ModuleType("vllm_fl.ops._C_ops_registry")
    fake_ops._apply_repetition_penalties_impl = applied
    monkeypatch.setitem(
        sys.modules, "vllm_fl.ops._C_ops_registry", fake_ops
    )

    had_lib = hasattr(register_ops._register_c_ops_privateuse1_fallbacks, "_lib")
    try:
        register_ops._register_c_ops_privateuse1_fallbacks()
        lib = instances[-1]
        assert lib.args == ("_C", "FRAGMENT")
        lib.impl.assert_called_once_with(
            "apply_repetition_penalties_", applied, "PrivateUse1"
        )
        assert register_ops._register_c_ops_privateuse1_fallbacks._lib is lib
    finally:
        if not had_lib and hasattr(
            register_ops._register_c_ops_privateuse1_fallbacks, "_lib"
        ):
            del register_ops._register_c_ops_privateuse1_fallbacks._lib


def test_gcu_register_builtins_records_impls(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.gcu import register_ops
    from vllm_fl.dispatch.backends.vendor.gcu.gcu import GCUBackend
    from vllm_fl.dispatch.types import BackendImplKind, BackendPriority

    monkeypatch.setattr(
        register_ops,
        "_register_c_ops_privateuse1_fallbacks",
        lambda: None,
    )
    monkeypatch.setattr(GCUBackend, "_available", True, raising=False)

    class _RecordingRegistry:
        def __init__(self):
            self.impls = None

        def register_many(self, impls):
            self.impls = list(impls)

    registry = _RecordingRegistry()
    register_ops.register_builtins(registry)

    assert registry.impls is not None
    by_name = {impl.op_name: impl for impl in registry.impls}
    assert set(by_name) == {
        "silu_and_mul",
        "rms_norm",
        "rotary_embedding",
        "attention_backend",
    }
    for impl in registry.impls:
        assert impl.impl_id == "vendor.gcu"
        assert impl.kind == BackendImplKind.VENDOR
        assert impl.vendor == "gcu"
        assert impl.priority == BackendPriority.VENDOR
        assert impl.is_available() is True
        assert impl.fn._is_available is not None


# ---------------------------------------------------------------------------
# base.py
# ---------------------------------------------------------------------------


class _ConcreteBackend(Backend):
    def is_available(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "concrete"


def test_base_vendor_defaults_to_none():
    backend = _ConcreteBackend()
    assert backend.vendor is None
    assert backend.name == "concrete"
    assert backend.is_available() is True


def test_base_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Backend()  # type: ignore[abstract]
