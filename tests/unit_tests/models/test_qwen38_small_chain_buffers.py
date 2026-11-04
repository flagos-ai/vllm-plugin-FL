# SPDX-License-Identifier: Apache-2.0
"""Model output-initialization and HC callback contracts, without a GPU."""

import __future__

import ast
import sys
from functools import lru_cache
from types import ModuleType, SimpleNamespace

import pytest
import torch
from _qwen38_cpu import ROOT, declarations


def fn(path, name, ns):
    node = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    scope = dict(ns)
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        scope,
    )
    return scope[name]


@pytest.mark.parametrize(
    "active,padded,capability",
    [(3, 3, True), (3, 5, True), (0, 5, True), (3, 5, False)],
)
def test_qsa_zero_elision_requires_provider_and_preserves_padding(
    monkeypatch, active, padded, capability
):
    module = ModuleType("vllm_fl.models.qwen3_8_flash_next.gpu.ops.qsa")
    module.qsa_prepare_split_workspace = lambda *a, **k: (1, None)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cache = torch.empty(1, 16, 1, 256, dtype=torch.bfloat16)
    indices = torch.zeros(active, 2051, dtype=torch.int32)

    class Indexer:
        raw_key_cache = SimpleNamespace(prefix="raw")
        compressed_key_cache = SimpleNamespace(prefix="compressed")
        output_width = 2051

        def __call__(self, *args):
            return indices

    main = SimpleNamespace(
        num_actual_tokens=active,
        slot_mapping=torch.full((active,), -1),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
    )
    raw = SimpleNamespace(
        num_actual_tokens=active, token_to_req=torch.zeros(active, dtype=torch.int32)
    )
    context = SimpleNamespace(
        attn_metadata={
            "main": main,
            "raw": raw,
            "compressed": SimpleNamespace(score_columns=2048),
        }
    )
    calls = []

    def attention(q, k, v, idx, table, mapping, scale, out, **kw):
        assert torch.isnan(out).all() if capability else torch.all(out == 0)
        out.fill_(3)
        calls.append(active)

    if capability:
        attention._qwen38_full_output = True
    obj = SimpleNamespace(
        layer_name="main",
        kv_cache=torch.empty(1),
        indexer=Indexer(),
        _metax=True,
        num_kv_heads=1,
        head_dim=256,
        _qsa_store_cache_rows=lambda *a: None,
        _qsa_split_workspaces={},
        _qsa_sparse_split_count=lambda *a: 1,
        _qsa_sparse_paged_attention=attention,
        _qsa_packed_workspaces={},
        _qsa_long_context=True,
        topk_indices_buffer=indices,
        scaling=0.0625,
    )
    method = fn(
        ROOT / "vllm_fl/models/qwen3_8_flash_next/gpu/qsa.py",
        "_run_qsa",
        {
            "torch": torch,
            "cast": lambda t, x: x,
            "QSAForwardMetadata": object,
            "get_forward_context": lambda: context,
            "_unpack_qsa_kv_cache": lambda *a: (cache, cache),
            "__package__": "vllm_fl.models.qwen3_8_flash_next.gpu",
        },
    )
    q = torch.ones(padded, 3, 256, dtype=torch.bfloat16)
    kv = torch.ones(padded, 1, 256, dtype=torch.bfloat16)
    out = torch.full_like(q, float("nan"))
    method(obj, torch.empty(padded, 2560), torch.arange(padded), q, kv, kv, q, out)
    assert torch.all(out[:active] == 3) and torch.all(out[active:] == 0)
    assert calls == ([active] if active else [])


@pytest.mark.parametrize("packed", [False, True])
def test_vendor_fallback_initializes_output_before_call(monkeypatch, packed):
    package = "vllm_fl.dispatch.backends.vendor.metax.impl.qwen3_8_flash_next"
    module = ModuleType(package + ".qsa_packing")

    def packed_call(*args, **kw):
        if packed:
            args[7].fill_(5)
        return packed

    module.packed_attention = packed_call
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def portable_call(*args, **kw):
        assert torch.all(args[7] == 0)
        args[7].fill_(7)

    method = fn(
        ROOT
        / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next/qsa_policy.py",
        "qsa_sparse_paged_attention",
        {
            "torch": torch,
            "portable": SimpleNamespace(qsa_sparse_paged_attention=portable_call),
            "__package__": package,
        },
    )
    x = torch.ones(2, 3, 256)
    out = torch.full_like(x, float("nan"))
    method(x, x, x, None, None, None, 0.0625, out, packed_workspace_cache={})
    assert torch.all(out == (5 if packed else 7))


def test_hc_callback_and_none_fallback_preserve_checkpoint_and_values():
    path = ROOT / "vllm_fl/models/qwen3_8_flash_next/common/hyperconnection.py"
    hc = declarations(
        path,
        "_hc_small_callback_cpu",
        {"F": torch.nn.functional, "can_use_hc_combine_norm_triton": None},
        names={
            "HyperConnectionConfig",
            "GroupedGemmaRMSNorm",
            "HyperConnectionBase",
            "GatedResidualSimple",
        },
    )
    cfg = hc.HyperConnectionConfig(
        hc_count=4,
        hidden_size=8,
        hc_lowrank=3,
        hc_per_branch_norm=True,
        use_fused_ops=False,
        params_dtype=torch.float32,
    )
    calls = []

    def activation(x, groups):
        calls.append((x.shape, groups))
        return torch.nn.functional.silu(x / groups)

    base = hc.GatedResidualSimple(cfg)
    new = hc.GatedResidualSimple(cfg, down_activation=activation)
    new.load_state_dict(base.state_dict())
    assert list(base.state_dict()) == list(new.state_dict())
    x = torch.randn(5, 32)
    block = torch.randn(5, 8)

    def run(mod):
        y, state = mod.mix(x)
        return y, mod.combine(block, state)

    with torch.no_grad():
        expected = run(base)
        actual = run(new)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
        new._down_activation = lambda *args: None
        fallback = run(new)
        assert all(torch.equal(a, b) for a, b in zip(expected, fallback))
    assert calls == [(torch.Size([5, 3]), 4)]


def test_hc_row_settings_cache_is_bounded_and_reuses_same_shape():
    path = (
        ROOT
        / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next/hc_policy.py"
    )
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_row_settings"
    )
    scope = {
        "lru_cache": lru_cache,
        "_settings": lambda: {"gate": [{"max_rows": 8192, "block": 256}]},
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    get = scope["_row_settings"]
    assert get("gate", 8192) is get("gate", 8192)
    for rows in range(1, 80):
        get("gate", rows)
    assert get.cache_info().maxsize == 64 and get.cache_info().currsize == 64
