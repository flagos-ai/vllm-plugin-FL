# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU checks for AMD compile routing and the rotary custom-op contract."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
AMD_ROTARY_MODULE = "vllm_fl.dispatch.backends.vendor.amd.graph_ops"
FLAGGEMS_ROTARY_MODULE = "vllm_fl.dispatch.backends.flaggems.impl.rotary"


def _load_class(path, name, **scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    scope.update(torch=torch)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), path, "exec"), scope)
    return scope[name]


def test_amd_rejects_sparse_mla_before_selecting_a_dense_backend(monkeypatch):
    registry = ModuleType("vllm.v1.attention.backends.registry")
    registry.AttentionBackendEnum = object()
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    backend = _load_class(
        "vllm_fl/dispatch/backends/vendor/amd/amd.py", "AMDBackend", Backend=object
    )

    with pytest.raises(NotImplementedError, match="Sparse MLA"):
        backend().attention_backend(use_mla=True, use_sparse=True)


@pytest.mark.parametrize(
    "source, name, base, dispatcher, op",
    [
        ("activation", "SiluAndMulFL", "SiluAndMul", "_silu_and_mul", "silu_and_mul"),
        ("activation", "GeluAndMulFL", "GeluAndMul", "_gelu_and_mul", "gelu_and_mul"),
        ("layernorm", "RMSNormFL", "RMSNorm", "_rms_norm", "rms_norm"),
        (
            "rotary_embedding",
            "RotaryEmbeddingFL",
            "RotaryEmbedding",
            "_rotary_embedding",
            "rotary_embedding",
        ),
    ],
)
def test_non_amd_compiled_ops_keep_normal_dispatch(
    monkeypatch, source, name, base, dispatcher, op
):
    import typing

    def forbidden(*args, **kwargs):
        raise AssertionError("Non-AMD graph must not call the AMD operator")

    custom_ops = ModuleType(AMD_ROTARY_MODULE)
    setattr(custom_ops, op + "_custom_op", forbidden)
    monkeypatch.setitem(sys.modules, AMD_ROTARY_MODULE, custom_ops)

    def normal_dispatch(layer, x, *args):
        if source == "rotary_embedding":
            return x + 1, args[0] + 1
        return x + 1

    layer_type = _load_class(
        f"vllm_fl/ops/{source}.py",
        name,
        **{
            base: object,
            dispatcher: normal_dispatch,
            "Optional": typing.Optional,
            "Union": typing.Union,
        },
    )
    layer = layer_type.__new__(layer_type)
    layer._use_amd_graph_ops = False
    if source == "rotary_embedding":
        layer.head_size = layer.rotary_dim = 4
        layer.is_neox_style = True
        layer._match_cos_sin_cache_dtype = lambda query: torch.zeros(2, 4)
        positions = torch.arange(2)
        query = torch.ones(2, 4)
        key = torch.ones(2, 4)
        fn = torch.compile(layer.forward_oot, backend="eager", fullgraph=True)
        q_out, k_out = fn(positions, query, key)
        torch.testing.assert_close(q_out, query + 1)
        torch.testing.assert_close(k_out, key + 1)
    else:
        fn = torch.compile(layer.forward_oot, backend="eager", fullgraph=True)
        sample = torch.ones(2, 4)
        torch.testing.assert_close(fn(sample), sample + 1)


def test_rotary_custom_op_schema_and_compiled_packed_qkv(monkeypatch):
    # Substitute only the ROCm leaf. Its in-place writes and aliased return
    # values reproduce the pinned kernel's interface on CPU.
    def rotary_leaf(
        obj, query, key, cos, sin, positions, *, rotary_interleaved, inplace
    ):
        assert inplace
        query.add_(1)
        key.add_(2)
        return query.view_as(query), key.view_as(key)

    rotary = ModuleType(FLAGGEMS_ROTARY_MODULE)
    rotary.rotary_embedding_flaggems = rotary_leaf
    monkeypatch.setitem(sys.modules, FLAGGEMS_ROTARY_MODULE, rotary)

    path = ROOT / "vllm_fl/dispatch/backends/vendor/amd/graph_ops.py"
    spec = importlib.util.spec_from_file_location("_amd_graph_ops_test", path)
    graph_ops = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, graph_ops)
    spec.loader.exec_module(graph_ops)
    try:
        graph_ops._rot_lib.impl("rotary_embedding", graph_ops._rot_cuda, "CPU")
        monkeypatch.setitem(sys.modules, AMD_ROTARY_MODULE, graph_ops)
        packed = torch.zeros(3, 3, 8)
        query, key = packed[:, 0:1, :4], packed[:, 1:2, :4]
        positions = torch.arange(3)
        cos = sin = torch.ones(3, 2)
        op = torch.ops.vllm_fl_amd_rot.rotary_embedding.default
        torch.library.opcheck(op, (query, key, cos, sin, positions, False))
        assert not packed.count_nonzero()

        def dispatch_rotary(layer, q, k, cos, sin, position_ids, interleaved, inplace):
            return rotary_leaf(
                layer,
                q,
                k,
                cos,
                sin,
                position_ids,
                rotary_interleaved=interleaved,
                inplace=inplace,
            )

        layer_type = _load_class(
            "vllm_fl/ops/rotary_embedding.py",
            "RotaryEmbeddingFL",
            RotaryEmbedding=object,
            _rotary_embedding=dispatch_rotary,
        )
        layer = layer_type.__new__(layer_type)
        layer.head_size = 8
        layer.rotary_dim = 4
        layer.is_neox_style = True
        layer._use_amd_graph_ops = True
        layer._match_cos_sin_cache_dtype = lambda query: torch.zeros(3, 4)

        def run(packed_qkv):
            q_out, k_out = layer.forward_oot(
                positions, packed_qkv[:, 0], packed_qkv[:, 1]
            )
            return q_out, k_out, packed_qkv.clone()

        expected = run(packed.clone())
        compiled = torch.compile(run, backend="aot_eager", fullgraph=True)
        actual = compiled(packed)
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference)
        torch.testing.assert_close(packed, expected[2])
    finally:
        torch._dynamo.reset()
        for library in (
            graph_ops._rot_lib,
            graph_ops._act_lib,
            graph_ops._rms_norm_lib,
        ):
            library._destroy()
