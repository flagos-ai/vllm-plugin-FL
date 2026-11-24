# Coverage tests: vllm_fl/ops/activation.py and rotary_embedding.py (CPU only).

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm_fl.ops.activation as activation
import vllm_fl.ops.rotary_embedding as rotary


def _stub_package(monkeypatch, names, **attrs):
    created = []
    for name in names:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        created.append(module)
    for key, value in attrs.items():
        setattr(created[-1], key, value)
    return created


# ---------------------------------------------------------------------------
# activation.py
# ---------------------------------------------------------------------------


def _make_silu(monkeypatch, vendor="mthreads"):
    monkeypatch.setattr(activation.SiluAndMul, "__init__", lambda self, **kw: None)
    monkeypatch.setattr(
        activation, "current_platform", SimpleNamespace(vendor_name=vendor)
    )
    return activation.SiluAndMulFL()


def _make_gelu(monkeypatch, vendor="mthreads", approximate="none"):
    monkeypatch.setattr(activation.GeluAndMul, "__init__", lambda self, **kw: None)
    monkeypatch.setattr(
        activation, "current_platform", SimpleNamespace(vendor_name=vendor)
    )
    layer = activation.GeluAndMulFL(approximate=approximate)
    layer.approximate = approximate
    return layer


def test_silu_forward_oot_dispatches(monkeypatch):
    layer = _make_silu(monkeypatch)
    sentinel = torch.ones(2, 4)
    calls = []
    monkeypatch.setattr(
        activation,
        "_silu_and_mul",
        lambda self, x: calls.append((self, x)) or sentinel,
    )
    x = torch.zeros(2, 8)
    assert layer.forward_oot(x) is sentinel
    assert calls[0][0] is layer
    assert calls[0][1] is x


def test_gelu_forward_oot_dispatches(monkeypatch):
    layer = _make_gelu(monkeypatch, approximate="tanh")
    sentinel = torch.ones(2, 4)
    calls = []
    monkeypatch.setattr(
        activation,
        "_gelu_and_mul",
        lambda self, x: calls.append((self, x)) or sentinel,
    )
    x = torch.zeros(2, 8)
    assert layer.forward_oot(x) is sentinel
    assert calls[0][0] is layer


def test_silu_amd_compiling_path(monkeypatch):
    layer = _make_silu(monkeypatch, vendor="amd")
    assert layer._use_amd_graph_ops is True
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    sentinel = torch.ones(2, 4)
    calls = []
    _stub_package(
        monkeypatch,
        ["vllm_fl.dispatch.backends.vendor.amd"],
        **{},
    )
    graph_ops = ModuleType("vllm_fl.dispatch.backends.vendor.amd.graph_ops")
    graph_ops.silu_and_mul_custom_op = lambda x: calls.append(x) or sentinel
    graph_ops.gelu_and_mul_custom_op = lambda x, approximate: sentinel
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.amd.graph_ops", graph_ops
    )

    x = torch.zeros(2, 8)
    assert layer.forward_oot(x) is sentinel
    assert calls == [x]


def test_gelu_amd_compiling_path(monkeypatch):
    layer = _make_gelu(monkeypatch, vendor="amd", approximate="tanh")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    sentinel = torch.ones(2, 4)
    calls = []
    _stub_package(monkeypatch, ["vllm_fl.dispatch.backends.vendor.amd"])
    graph_ops = ModuleType("vllm_fl.dispatch.backends.vendor.amd.graph_ops")
    graph_ops.gelu_and_mul_custom_op = lambda x, approximate: calls.append(
        (x, approximate)
    ) or sentinel
    graph_ops.silu_and_mul_custom_op = lambda x: sentinel
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.amd.graph_ops", graph_ops
    )

    x = torch.zeros(2, 8)
    assert layer.forward_oot(x) is sentinel
    assert calls[0][1] == "tanh"


def test_activation_forward_cuda_delegates(monkeypatch):
    layer = _make_silu(monkeypatch)
    sentinel = torch.ones(2, 4)
    monkeypatch.setattr(
        activation.SiluAndMulFL, "forward_oot", lambda self, x: sentinel
    )
    assert layer.forward_cuda(torch.zeros(2, 8)) is sentinel

    gelu = _make_gelu(monkeypatch)
    monkeypatch.setattr(
        activation.GeluAndMulFL, "forward_oot", lambda self, x: sentinel
    )
    assert gelu.forward_cuda(torch.zeros(2, 8)) is sentinel


# ---------------------------------------------------------------------------
# rotary_embedding.py
# ---------------------------------------------------------------------------


def _rotary_layer(monkeypatch, vendor="mthreads", device_type="cuda"):
    monkeypatch.setattr(
        rotary.RotaryEmbedding, "__init__", lambda self, *a, **k: None
    )
    monkeypatch.setattr(
        rotary,
        "current_platform",
        SimpleNamespace(vendor_name=vendor, device_type=device_type),
    )
    return rotary.RotaryEmbeddingFL(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=16,
        base=10000.0,
        is_neox_style=True,
        dtype=torch.float32,
    )


def _forward_instance(head_size, rotary_dim, amd=False):
    instance = object.__new__(rotary.RotaryEmbeddingFL)
    instance.head_size = head_size
    instance.rotary_dim = rotary_dim
    instance.is_neox_style = True
    instance._use_amd_graph_ops = amd
    return instance


def test_rotary_init_sets_amd_flag(monkeypatch):
    layer = _rotary_layer(monkeypatch)
    assert layer._use_amd_graph_ops is False

    amd = _rotary_layer(monkeypatch, vendor="amd")
    assert amd._use_amd_graph_ops is True


def test_rotary_init_initializes_ascend(monkeypatch):
    called = []
    _stub_package(
        monkeypatch,
        [
            "vllm_fl.dispatch.backends.vendor.ascend",
            "vllm_fl.dispatch.backends.vendor.ascend.impl",
        ],
    )
    module = ModuleType("vllm_fl.dispatch.backends.vendor.ascend.impl.rotary")
    module.initialize_rotary_embedding_ascend = lambda: called.append(True)
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.ascend.impl.rotary", module
    )
    _rotary_layer(monkeypatch, vendor="ascend", device_type="npu")
    assert called == [True]


def test_rotary_forward_oot_key_none_uses_native(monkeypatch):
    layer = _forward_instance(8, 8)
    sentinel = (torch.zeros(1, 1, 8), None)
    monkeypatch.setattr(
        rotary.RotaryEmbeddingFL, "forward_native", lambda self, *a, **k: sentinel
    )
    assert layer.forward_oot(torch.tensor([0]), torch.zeros(1, 1, 8), None) is sentinel


def test_rotary_forward_oot_full_head(monkeypatch):
    layer = _forward_instance(8, 8)
    cache = torch.zeros(16, 8)
    monkeypatch.setattr(
        rotary.RotaryEmbeddingFL,
        "_match_cos_sin_cache_dtype",
        lambda self, query: cache,
    )
    calls = []
    monkeypatch.setattr(
        rotary,
        "_rotary_embedding",
        lambda self, q, k, cos, sin, pos, interleaved, inplace: calls.append(
            (q.shape, k.shape, cos.shape, sin.shape, interleaved, inplace)
        )
        or (torch.ones_like(q), torch.ones_like(k)),
    )
    positions = torch.tensor([0, 1, 2, 3])
    query = torch.zeros(4, 2, 8)
    key = torch.zeros(4, 2, 8)

    out_query, out_key = layer.forward_oot(positions, query, key)

    assert out_query.shape == (4, 2, 8)
    assert out_key.shape == (4, 2, 8)
    assert calls[0][4] is False  # not rotary_interleaved for neox
    assert calls[0][5] is True  # inplace


def test_rotary_forward_oot_partial_rotary_dim(monkeypatch):
    layer = _forward_instance(8, 4)
    cache = torch.zeros(16, 4)
    monkeypatch.setattr(
        rotary.RotaryEmbeddingFL,
        "_match_cos_sin_cache_dtype",
        lambda self, query: cache,
    )
    monkeypatch.setattr(
        rotary,
        "_rotary_embedding",
        lambda self, q, k, cos, sin, pos, interleaved, inplace: (
            torch.ones_like(q),
            torch.ones_like(k),
        ),
    )
    query = torch.zeros(4, 2, 8)
    key = torch.zeros(4, 2, 8)
    out_query, out_key = layer.forward_oot(torch.tensor([0, 1, 2, 3]), query, key)
    assert out_query.shape == (4, 2, 8)
    assert out_key.shape == (4, 2, 8)


def test_rotary_forward_oot_amd_compiling(monkeypatch):
    layer = _forward_instance(8, 8, amd=True)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    _stub_package(monkeypatch, ["vllm_fl.dispatch.backends.vendor.amd"])
    graph_ops = ModuleType("vllm_fl.dispatch.backends.vendor.amd.graph_ops")
    graph_ops.rotary_embedding_custom_op = (
        lambda q, k, cos, sin, pos, rotary_interleaved: (
            torch.zeros_like(q),
            torch.zeros_like(k),
        )
    )
    monkeypatch.setitem(
        sys.modules, "vllm_fl.dispatch.backends.vendor.amd.graph_ops", graph_ops
    )
    monkeypatch.setattr(
        rotary.RotaryEmbeddingFL,
        "_match_cos_sin_cache_dtype",
        lambda self, query: torch.zeros(16, 8),
    )
    query = torch.zeros(4, 2, 8)
    key = torch.zeros(4, 2, 8)
    out_query, out_key = layer.forward_oot(torch.tensor([0, 1, 2, 3]), query, key)
    assert out_query.shape == (4, 2, 8)
    assert out_key.shape == (4, 2, 8)


def test_rotary_forward_cuda_delegates(monkeypatch):
    layer = _forward_instance(8, 8)
    sentinel = (torch.zeros(1, 1, 8), torch.zeros(1, 1, 8))
    monkeypatch.setattr(
        rotary.RotaryEmbeddingFL, "forward_oot", lambda self, *a, **k: sentinel
    )
    assert (
        layer.forward_cuda(torch.tensor([0]), torch.zeros(1, 1, 8), torch.zeros(1, 1, 8))
        is sentinel
    )
