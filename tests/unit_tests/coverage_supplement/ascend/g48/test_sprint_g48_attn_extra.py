# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""G48: residual CPU coverage for the two attention impl modules.

Targets the regions left by g6/g20/g33/g45 and the repo's NPU-only tests:

  * ``vendor/ascend/impl/attention.py`` -- ``AscendAttentionBackendImpl``
    construction, FIA/PA eager forwards, reshape-and-cache branches, encoder
    forward, ``forward_impl`` routing, ``forward`` guards, the FULL-graph
    capture/update task path, ``supported_kv_cache_layouts`` and the MLA
    placeholder.
  * ``flaggems/impl/attention.py`` -- backend capability matrix, hybrid block
    sizes, sliding-window config discovery, cascade metadata build, KV-cache
    split validation, encoder KV no-op and the encoder/none forward guards.

Everything is CPU-only: ``torch_npu`` kernels, NPU streams/events and the
graph-parameter store are recording fakes; the FlagGems wheel is stubbed when
absent.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

_TARGET_REL = "vllm_fl/dispatch/backends/vendor/ascend/impl/attention.py"
_FLAGGEMS_REL = "vllm_fl/dispatch/backends/flaggems/impl/attention.py"


# ---------------------------------------------------------------------------
# module loading
# ---------------------------------------------------------------------------


def _find_repo():
    env = os.environ.get("FL_REPO_ROOT") or os.environ.get("FL_WORKSPACE")
    candidates = []
    if env:
        candidates.append(pathlib.Path(env))
    here = pathlib.Path(__file__).resolve()
    candidates.extend([here.parent, *here.parents])
    candidates.append(pathlib.Path.cwd())
    for entry in sys.path:
        if entry:
            candidates.append(pathlib.Path(entry))
    for base in candidates:
        if (base / "vllm_fl" / "dispatch" / "backends").is_dir():
            return base
    return None


def _install_module(monkeypatch, name, attrs):
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent in sys.modules:
            continue
        try:
            importlib.import_module(parent)
        except Exception:
            pkg = types.ModuleType(parent)
            pkg.__path__ = []
            monkeypatch.setitem(sys.modules, parent, pkg)
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    if len(parts) > 1:
        parent_mod = sys.modules.get(".".join(parts[:-1]))
        if parent_mod is not None:
            monkeypatch.setattr(parent_mod, parts[-1], mod, raising=False)
    return mod


def _exec_attention(name, path):
    impl_dir = path.parent
    package = types.ModuleType(name.rsplit(".", 1)[0])
    package.__path__ = [str(impl_dir)]
    package.__package__ = name.rsplit(".", 1)[0]
    sys.modules[package.__name__] = package

    import vllm.v1.attention.backends.registry as registry

    original = registry.register_backend
    registry.register_backend = lambda *_a, **_k: (lambda cls: cls)
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        registry.register_backend = original
    return module


def _load_flaggems(repo):
    try:
        from flaggems_vllm import (  # noqa: F401
            flash_attn_varlen_func,
            flash_mla,
            reshape_and_cache_flash,
        )
    except Exception:
        if "flaggems_vllm" not in sys.modules:
            stub = types.ModuleType("flaggems_vllm")
            stub.flash_attn_varlen_func = lambda *a, **k: None
            stub.flash_mla = lambda *a, **k: None
            stub.reshape_and_cache_flash = lambda *a, **k: None
            sys.modules["flaggems_vllm"] = stub
    path = repo / _FLAGGEMS_REL
    spec = importlib.util.spec_from_file_location("g48_flaggems_attention", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["g48_flaggems_attention"] = module
    spec.loader.exec_module(module)
    return module


_REPO = _find_repo()
if _REPO is None:
    pytest.skip("cannot locate vllm_fl tree on this host", allow_module_level=True)
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_TARGET = _REPO / _TARGET_REL
if not _TARGET.is_file():
    pytest.skip(f"target module missing: {_TARGET}", allow_module_level=True)

try:
    attn = _exec_attention("g48_asc_impl.attention", _TARGET)
except Exception as exc:  # torch_npu / vllm not importable here
    attn = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    fg = _load_flaggems(_REPO)
except Exception as exc:  # pragma: no cover - env specific
    fg = None
    _FG_ERROR = exc
else:
    _FG_ERROR = None


@pytest.fixture()
def mod():
    if attn is None:
        pytest.skip(f"attention module unavailable: {_IMPORT_ERROR}")
    return attn


@pytest.fixture()
def gems():
    if fg is None:
        pytest.skip(f"flaggems attention unavailable: {_FG_ERROR}")
    return fg


# ---------------------------------------------------------------------------
# shared fakes
# ---------------------------------------------------------------------------


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Event:
    def wait(self, *a, **k):
        pass

    def reset(self, *a, **k):
        pass

    def record(self, *a, **k):
        pass


def _fake_torch_npu_state():
    return SimpleNamespace(
        stream=lambda *a, **k: _Ctx(),
        current_stream=lambda *a, **k: _Ctx(),
        ExternalEvent=_Event,
        graph_task_update_begin=lambda *a, **k: None,
        graph_task_update_end=lambda *a, **k: None,
        graph_task_group_begin=lambda *a, **k: None,
        graph_task_group_end=lambda *a, **k: "handle",
        is_available=lambda: True,
    )


def _fake_ops(rec=None):
    rec = rec if rec is not None else {}

    def paged(**kw):
        rec.setdefault("paged", []).append(kw)

    def paged_ws(**kw):
        rec.setdefault("paged_ws", []).append(kw)
        return "PWS"

    def fia(**kw):
        rec.setdefault("fia", []).append(kw)
        q = kw["query"]
        return torch.zeros(q.shape[0], q.shape[1] * q.shape[2]), None

    def fia_out(**kw):
        rec.setdefault("fia_out", []).append(kw)

    fia.out = fia_out

    def fia_ws(**kw):
        rec.setdefault("fia_ws", []).append(kw)
        return "FWS"

    def rac(**kw):
        rec.setdefault("rac", []).append(kw)

    def fusion(**kw):
        rec.setdefault("fusion", []).append(kw)
        return torch.zeros(kw["query"].shape), None

    return SimpleNamespace(
        _npu_paged_attention=paged,
        _npu_paged_attention_get_workspace=paged_ws,
        npu_fused_infer_attention_score=fia,
        _npu_fused_infer_attention_score_get_max_workspace=fia_ws,
        _npu_reshape_and_cache=rac,
        npu_fusion_attention=fusion,
    )


class _GraphParams:
    def __init__(self):
        self.attn_params = {}
        self.handles = {}
        self.events = {}
        self.workspaces = {}

    def require_capture(self, descriptor):
        self.attn_params.setdefault(descriptor, [])
        self.handles.setdefault(descriptor, [])
        self.events.setdefault(descriptor, [])
        self.workspaces.setdefault(descriptor, None)


def _impl(mod, **attrs):
    impl = object.__new__(mod.AscendAttentionBackendImpl)
    defaults = dict(
        num_heads=4,
        head_size=8,
        num_kv_heads=2,
        scale=0.5,
        sliding_window=None,
        sinks=None,
        key_cache=None,
        value_cache=None,
        attn_type=mod.AttentionType.DECODER,
        vllm_config=SimpleNamespace(
            speculative_config=None,
            quant_config=None,
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace()),
            use_v2_model_runner=False,
        ),
    )
    defaults.update(attrs)
    for key, value in defaults.items():
        setattr(impl, key, value)
    return impl


def _meta(mod, state=None, **kw):
    base = dict(
        attn_state=state or mod.AscendAttentionState.PrefillNoCache,
        num_actual_tokens=2,
        seq_lens=torch.tensor([5, 6], dtype=torch.int32),
        seq_lens_list=[5, 6],
        actual_seq_lengths_q=[1, 2],
        block_tables=torch.zeros(2, 2, dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        causal=True,
        attn_mask=None,
        model_runner_type="generation",
    )
    base.update(kw)
    return mod.AscendMetadata(**base)


def _layer(scale=1.0, layer_name="L"):
    return SimpleNamespace(
        _k_scale_float=scale,
        _v_scale_float=scale,
        layer_name=layer_name,
    )


# ===========================================================================
# Ascend impl
# ===========================================================================


def test_ascend_impl_init_and_weak_ref(mod, monkeypatch):
    monkeypatch.setattr(
        mod, "get_current_vllm_config", lambda: SimpleNamespace()
    )
    impl = mod.AscendAttentionBackendImpl(
        num_heads=4,
        head_size=8,
        scale=0.5,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=64,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=mod.AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
        sinks="sink",
    )
    assert impl.num_kv_heads == 2
    assert impl.hidden_size == 32
    assert impl.num_queries_per_kv == 2
    assert impl.sliding_window == 64
    assert impl.sinks == "sink"
    assert impl.key_cache is None and impl.value_cache is None

    plain = mod.AscendAttentionBackendImpl(
        4, 8, 0.5, None, None, None, "auto", None,
        mod.AttentionType.DECODER, None,
    )
    assert plain.num_kv_heads == 4

    _install_module(
        monkeypatch,
        "vllm_fl.compilation.graph",
        {"weak_ref_tensors": lambda t: ("weak", t)},
    )
    assert impl._weak_ref_tensor(None) is None
    assert impl._weak_ref_tensor("x") == ("weak", "x")


def test_ascend_supported_kv_cache_layouts(mod, monkeypatch):
    layouts = SimpleNamespace(LBNHC="LBNHC", LHBNC="LHBNC")
    _install_module(
        monkeypatch, "vllm.v1.kv_cache_layout", {"KVCacheLayout": layouts}
    )
    monkeypatch.setattr(
        mod, "get_current_vllm_config", lambda: SimpleNamespace(model_config=object())
    )
    _install_module(
        monkeypatch,
        "vllm_fl.kv_cache.ascend.hybrid",
        {"uses_dense_hybrid_cache": lambda mc: True},
    )
    assert mod.AscendAttentionBackend.supported_kv_cache_layouts() == ("LBNHC",)

    _install_module(
        monkeypatch,
        "vllm_fl.kv_cache.ascend.hybrid",
        {"uses_dense_hybrid_cache": lambda mc: False},
    )
    assert mod.AscendAttentionBackend.supported_kv_cache_layouts() == ("LHBNC",)




def test_ascend_reshape_and_cache_contiguous(mod, monkeypatch):
    rec = {}
    monkeypatch.setattr(mod, "torch_npu", _fake_ops(rec))
    impl = _impl(mod)
    key_cache = torch.zeros(2, 4, 2, 8)
    value_cache = torch.zeros(2, 4, 2, 8)
    key = torch.zeros(3, 2, 8)
    value = torch.ones(3, 2, 8)
    metadata = _meta(
        mod, num_actual_tokens=3, slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int64)
    )
    out_key, out_value = impl.reshape_and_cache(
        key, value, (key_cache, value_cache), metadata
    )
    assert out_key is key and out_value is value
    assert impl.key_cache is key_cache
    assert rec["rac"][0]["slot_indices"].dtype == torch.int32

    single_key, single_value = impl.reshape_and_cache(key, value, (key_cache,), metadata)
    assert single_key is key and single_value is value
    assert len(rec["rac"]) == 1


def test_ascend_reshape_and_cache_strided(mod, monkeypatch):
    rec = {}

    def flash(*args):
        rec.setdefault("flash", []).append(args)

    _install_module(
        monkeypatch,
        "flag_gems.fused.reshape_and_cache_flash",
        {"reshape_and_cache_flash": flash},
    )
    monkeypatch.setattr(mod, "torch_npu", _fake_ops({}))
    impl = _impl(mod)
    key_cache = torch.zeros(2, 4, 2, 8).transpose(0, 1)
    value_cache = torch.zeros(2, 4, 2, 8).transpose(0, 1)
    metadata = _meta(
        mod, num_actual_tokens=3, slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int64)
    )
    key = torch.zeros(3, 2, 8)
    value = torch.ones(3, 2, 8)
    out_key, out_value = impl.reshape_and_cache(
        key, value, (key_cache, value_cache), metadata
    )
    assert out_key is key and out_value is value
    assert len(rec["flash"]) == 1
    assert rec["flash"][0][5] == "auto"


def test_ascend_forward_fused_matrix(mod, monkeypatch):
    rec = {}
    monkeypatch.setattr(mod, "torch_npu", _fake_ops(rec))
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: SimpleNamespace(capturing=False)
    )
    impl = _impl(mod)
    query = torch.zeros(2, 4, 8)
    key = torch.zeros(2, 2, 8)
    output = torch.zeros(2, 4, 8)

    returned = impl.forward_fused_infer_attention(
        query, key, key, _meta(mod, mod.AscendAttentionState.PrefillNoCache), output
    )
    assert returned is output
    assert rec["fia"][-1]["sparse_mode"] == 3
    assert "atten_mask" in rec["fia"][-1]

    impl.sliding_window = 8
    impl.forward_fused_infer_attention(
        query, key, key, _meta(mod, mod.AscendAttentionState.PrefillNoCache), output
    )
    assert rec["fia"][-1]["sparse_mode"] == 4
    assert rec["fia"][-1]["pre_tokens"] == 8
    assert rec["fia"][-1]["next_tokens"] == 0

    impl.sliding_window = None
    impl.forward_fused_infer_attention(
        query,
        key,
        key,
        _meta(mod, mod.AscendAttentionState.PrefillNoCache, causal=False),
        output,
    )
    assert rec["fia"][-1]["sparse_mode"] == 0
    assert "atten_mask" not in rec["fia"][-1]


def test_ascend_forward_fused_capture_delegates(mod, monkeypatch):
    impl = _impl(mod)
    impl._capture_fused_infer_attention = lambda *a: "captured"
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: SimpleNamespace(capturing=True)
    )
    assert (
        impl.forward_fused_infer_attention(
            torch.zeros(2, 4, 8),
            torch.zeros(2, 2, 8),
            torch.zeros(2, 2, 8),
            _meta(mod),
            torch.zeros(2, 4, 8),
        )
        == "captured"
    )


def test_ascend_forward_paged_and_capture_delegates(mod, monkeypatch):
    rec = {}
    monkeypatch.setattr(mod, "torch_npu", _fake_ops(rec))
    impl = _impl(
        mod,
        key_cache=torch.zeros(2, 4, 2, 8),
        value_cache=torch.zeros(2, 4, 2, 8),
    )
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: SimpleNamespace(capturing=False)
    )
    query = torch.zeros(2, 4, 8)
    output = torch.zeros(2, 4, 8)
    returned = impl.forward_paged_attention(
        query, _meta(mod, mod.AscendAttentionState.DecodeOnly), output
    )
    assert returned is output
    assert len(rec["paged"]) == 1

    impl._capture_paged_attention = lambda *a: "captured"
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: SimpleNamespace(capturing=True)
    )
    assert (
        impl.forward_paged_attention(query, _meta(mod), output) == "captured"
    )


def test_ascend_forward_encoder_attention(mod, monkeypatch):
    rec = {}
    monkeypatch.setattr(mod, "torch_npu", _fake_ops(rec))
    impl = _impl(mod)
    query = torch.zeros(2, 4, 8)
    output = torch.zeros(2, 4, 8)
    impl._forward_encoder_attention(
        query,
        torch.zeros(2, 2, 8),
        torch.zeros(2, 2, 8),
        _meta(mod, attn_mask=torch.zeros(1)),
        output,
    )
    assert rec["fusion"][-1]["sparse_mode"] == 3
    assert "atten_mask" in rec["fusion"][-1]

    impl._forward_encoder_attention(
        query,
        torch.zeros(2, 2, 8),
        torch.zeros(2, 2, 8),
        _meta(mod, causal=False),
        output,
    )
    assert "atten_mask" not in rec["fusion"][-1]


def test_ascend_forward_impl_routing(mod, monkeypatch):
    impl = _impl(mod)
    calls = []
    impl.forward_paged_attention = lambda *a: calls.append("pa") or a[2]
    impl.forward_fused_infer_attention = lambda *a: calls.append("fia") or a[4]
    query = torch.zeros(2, 4, 8)
    output = torch.zeros(2, 4, 8)
    cache = (torch.zeros(2, 4, 2, 8), torch.zeros(2, 4, 2, 8))

    monkeypatch.setattr(mod, "using_paged_attention", lambda *a: True)
    impl.forward_impl(
        query, None, None, cache,
        _meta(mod, mod.AscendAttentionState.DecodeOnly), output,
    )
    assert calls[-1] == "pa"

    impl.sliding_window = 4
    impl.forward_impl(
        query, None, None, cache,
        _meta(mod, mod.AscendAttentionState.DecodeOnly), output,
    )
    assert calls[-1] == "fia"

    impl.sliding_window = None
    monkeypatch.setattr(mod, "using_paged_attention", lambda *a: False)
    impl.forward_impl(
        query, None, None, cache,
        _meta(mod, mod.AscendAttentionState.PrefillNoCache), output,
    )
    assert calls[-1] == "fia"


def test_ascend_forward_guards_and_paths(mod, monkeypatch):
    impl = _impl(mod)
    layer = _layer()
    query = torch.zeros(2, 4, 8)

    output = torch.ones(2, 4, 8)
    assert torch.equal(
        impl.forward(layer, query, None, None, None, None, output),
        torch.zeros_like(output),
    )

    with pytest.raises(NotImplementedError):
        impl.forward(
            layer, query, None, None, None, None,
            torch.zeros(2, 4, 8), output_scale=torch.ones(1),
        )

    impl.attn_type = "invalid"
    with pytest.raises(NotImplementedError):
        impl.forward(layer, query, None, None, None, None, torch.zeros(2, 4, 8))

    impl.attn_type = mod.AttentionType.DECODER
    impl._forward_encoder_attention = lambda *a: torch.ones_like(a[0])
    pooling = _meta(mod, model_runner_type="pooling")
    output = torch.zeros(2, 4, 8)
    assert impl.forward(layer, query, None, None, None, pooling, output) is output

    impl.forward_impl = lambda *a: a[5]
    impl.reshape_and_cache = lambda k, v, c, m: (k, v)
    output = torch.zeros(2, 4, 8)
    returned = impl.forward(
        layer,
        query,
        torch.zeros(2, 2, 8),
        torch.zeros(2, 2, 8),
        (torch.zeros(2, 4, 2, 8), torch.zeros(2, 4, 2, 8)),
        _meta(mod),
        output,
    )
    assert returned is output


def test_ascend_require_regular_graph_capture_guards(mod):
    meta = _meta(mod, mod.AscendAttentionState.DecodeOnly)
    layer = _layer()

    impl = _impl(
        mod,
        vllm_config=SimpleNamespace(
            speculative_config=object(),
            quant_config=None,
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace()),
        ),
    )
    with pytest.raises(NotImplementedError):
        impl._require_regular_graph_capture(meta, layer)

    impl = _impl(
        mod,
        vllm_config=SimpleNamespace(
            speculative_config=None,
            quant_config=object(),
            model_config=SimpleNamespace(hf_text_config=SimpleNamespace()),
        ),
    )
    with pytest.raises(NotImplementedError):
        impl._require_regular_graph_capture(meta, layer)

    impl = _impl(mod, sinks=object())
    with pytest.raises(NotImplementedError):
        impl._require_regular_graph_capture(meta, layer)

    impl = _impl(mod)
    with pytest.raises(NotImplementedError):
        impl._require_regular_graph_capture(
            _meta(mod, mod.AscendAttentionState.ChunkedPrefill), layer
        )
    with pytest.raises(RuntimeError):
        impl._require_regular_graph_capture(meta, SimpleNamespace(layer_name=None))
    assert impl._require_regular_graph_capture(meta, layer) == "L"


def test_ascend_capture_fused_infer_attention(mod, monkeypatch):
    rec = {}
    gp = _GraphParams()
    _install_module(
        monkeypatch,
        "vllm_fl.compilation.graph_params",
        {"get_graph_params": lambda: gp},
    )
    monkeypatch.setattr(mod, "torch_npu", _fake_ops(rec))
    monkeypatch.setattr(torch, "npu", _fake_torch_npu_state(), raising=False)
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: SimpleNamespace(capturing=True, batch_descriptor="d"),
    )
    impl = _impl(
        mod,
        key_cache=torch.zeros(2, 4, 2, 8),
        value_cache=torch.zeros(2, 4, 2, 8),
    )
    impl._weak_ref_tensor = lambda t: t
    output = torch.zeros(2, 4, 8)
    returned = impl._capture_fused_infer_attention(
        torch.zeros(2, 4, 8),
        torch.zeros(2, 2, 8),
        torch.zeros(2, 2, 8),
        _meta(mod, mod.AscendAttentionState.DecodeOnly, attn_mask=torch.zeros(1)),
        output,
        _layer(),
    )
    assert returned is output
    assert gp.workspaces["d"] == "FWS"
    assert len(gp.attn_params["d"]) == 1
    assert isinstance(gp.attn_params["d"][0], mod.FusedInferAttentionGraphParam)

    gp.workspaces["d"] = "PRE"
    impl._capture_fused_infer_attention(
        torch.zeros(2, 4, 8),
        torch.zeros(2, 2, 8),
        torch.zeros(2, 2, 8),
        _meta(mod, mod.AscendAttentionState.DecodeOnly, attn_mask=torch.zeros(1)),
        output,
        _layer(),
    )
    assert len(rec["fia_ws"]) == 1


def test_ascend_capture_paged_attention(mod, monkeypatch):
    rec = {}
    gp = _GraphParams()
    _install_module(
        monkeypatch,
        "vllm_fl.compilation.graph_params",
        {"get_graph_params": lambda: gp},
    )
    monkeypatch.setattr(mod, "torch_npu", _fake_ops(rec))
    monkeypatch.setattr(torch, "npu", _fake_torch_npu_state(), raising=False)
    monkeypatch.setattr(
        mod,
        "get_forward_context",
        lambda: SimpleNamespace(capturing=True, batch_descriptor="d"),
    )
    impl = _impl(
        mod,
        key_cache=torch.zeros(2, 4, 2, 8),
        value_cache=torch.zeros(2, 4, 2, 8),
    )
    impl._weak_ref_tensor = lambda t: t
    output = torch.zeros(2, 4, 8)
    returned = impl._capture_paged_attention(
        torch.zeros(2, 4, 8),
        _meta(mod, mod.AscendAttentionState.DecodeOnly),
        output,
        _layer(),
    )
    assert returned is output
    assert len(gp.attn_params["d"]) == 1
    assert isinstance(gp.attn_params["d"][0], mod.PagedAttentionGraphParam)
    assert len(rec["paged"]) == 1


def _preseed_pa_gp(mod, descriptor):
    gp = _GraphParams()
    param = mod.PagedAttentionGraphParam(
        "L",
        torch.zeros(2, 4, 8),
        torch.zeros(2, 4, 2, 8),
        torch.zeros(2, 4, 2, 8),
        2,
        4,
        0.5,
        torch.zeros(2, 2, dtype=torch.int32),
        torch.tensor([5, 6], dtype=torch.int32),
        torch.zeros(2, 4, 8),
    )
    gp.attn_params[descriptor] = [param]
    gp.handles[descriptor] = ["h"]
    gp.events[descriptor] = [_Event()]
    gp.workspaces[descriptor] = "WS"
    return gp


def _fia_param(mod, name="L", table=None):
    return mod.FusedInferAttentionGraphParam(
        name,
        torch.zeros(2, 4, 8),
        torch.zeros(2, 4, 16),
        torch.zeros(2, 4, 16),
        table if table is not None else torch.zeros(2, 2, dtype=torch.int32),
        torch.zeros(1),
        128,
        [5, 6],
        [1, 2],
        2,
        4,
        0.5,
        torch.zeros(2, 4, 8),
        torch.zeros(1),
        3,
        2147483647,
        2147483647,
    )






def test_ascend_mla_backend_raises(mod):
    with pytest.raises(NotImplementedError):
        mod.AscendMLABackend()


# ===========================================================================
# FlagGems impl
# ===========================================================================


def test_fg_static_support_matrix(gems):
    backend = gems.AttentionFLBackend
    assert backend.supports_attn_type(gems.AttentionType.DECODER) is True
    assert backend.supports_attn_type(gems.AttentionType.ENCODER_DECODER) is True
    assert backend.supports_attn_type("nope") is False
    assert backend.supports_head_size(8) is True
    assert backend.supports_head_size(256) is True
    assert backend.supports_head_size(7) is False
    assert backend.supports_head_size(264) is False
    assert (
        backend.supports_combination(
            64, torch.float16, "auto", 16, False, True, False, False, None
        )
        == "not support sink"
    )
    assert (
        backend.supports_combination(
            64, torch.float16, "auto", 16, False, False, False, True, None
        )
        == "FlagGems FA2 does not support multimodal prefix attention"
    )
    assert (
        backend.supports_combination(
            64, torch.float16, "auto", 16, False, False, False, False, None
        )
        is None
    )


def test_fg_block_sizes_hybrid(gems, monkeypatch):
    monkeypatch.setattr(
        gems,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=None,
            cache_config=SimpleNamespace(
                mamba_ssm_cache_dtype="auto", mamba_cache_dtype="auto"
            ),
        ),
    )
    sizes = gems.AttentionFLBackend.get_supported_kernel_block_sizes()
    assert len(sizes) == 1
    assert getattr(sizes[0], "base", sizes[0]) == 16

    monkeypatch.setattr(
        gems,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            model_config=SimpleNamespace(is_hybrid=True),
            cache_config=SimpleNamespace(
                mamba_ssm_cache_dtype="auto", mamba_cache_dtype="float32"
            ),
        ),
    )
    assert gems.AttentionFLBackend.get_supported_kernel_block_sizes() == [16, 32, 64]


def test_fg_sliding_window_configs(gems, monkeypatch):
    plain = gems.AttentionFLImpl(4, 64, 0.5, 2, None, None, "auto")
    windowed = gems.AttentionFLImpl(4, 64, 0.5, 2, None, 5, "auto")
    layers = {
        "a": SimpleNamespace(impl=plain),
        "b": SimpleNamespace(impl=windowed),
    }
    monkeypatch.setattr(
        gems, "get_layers_from_vllm_config", lambda *a, **k: layers
    )
    assert gems._get_sliding_window_configs(object()) == {(-1, -1), (4, 0)}


def test_fg_builder_build_cascade(gems):
    builder = object.__new__(gems.AttentionFLMetadataBuilder)
    for key, value in dict(
        aot_schedule=False,
        aot_sliding_window=None,
        use_full_cuda_graph=False,
        max_cudagraph_size=0,
        max_num_splits=0,
        dcp_world_size=1,
        dcp_rank=0,
        cp_kv_cache_interleave_size=1,
        device=torch.device("cpu"),
    ).items():
        setattr(builder, key, value)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=4,
        max_query_len=2,
        max_seq_len=10,
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([6, 8], dtype=torch.int32),
        block_table_tensor=torch.zeros(2, 2, dtype=torch.int32),
        slot_mapping=torch.arange(4),
        causal=True,
    )
    metadata = builder.build(4, common)
    assert metadata.use_cascade is True
    assert metadata.cu_prefix_query_lens.tolist() == [0, 4]
    assert metadata.prefix_kv_lens.tolist() == [4]
    assert metadata.suffix_kv_lens.tolist() == [2, 4]
    assert metadata.scheduler_metadata is None


def test_fg_split_kv_cache(gems):
    cache = torch.zeros(2, 2, 16, 128)
    key_cache, value_cache = gems._split_kv_cache(cache, 64)
    assert key_cache.shape == (2, 16, 2, 64)
    assert value_cache.shape == (2, 16, 2, 64)
    with pytest.raises(ValueError):
        gems._split_kv_cache(torch.zeros(2, 3), 8)
    with pytest.raises(ValueError):
        gems._split_kv_cache(torch.zeros(2, 2, 16, 64), 64)


def test_fg_do_kv_cache_update_encoder(gems):
    impl = gems.AttentionFLImpl(
        4, 64, 0.5, 2, None, None, "auto", attn_type=gems.AttentionType.ENCODER
    )
    assert (
        impl.do_kv_cache_update(
            None,
            torch.zeros(3, 2, 64),
            torch.zeros(3, 2, 64),
            torch.zeros(2, 2, 16, 128),
            torch.zeros(3, dtype=torch.int64),
        )
        is None
    )


def test_fg_forward_guards_and_encoder(gems, monkeypatch):
    impl = gems.AttentionFLImpl(4, 64, 0.5, 2, None, None, "auto")
    output = torch.ones(3, 4, 64)
    with pytest.raises(NotImplementedError):
        impl.forward(
            None, None, None, None, None, None, output,
            output_block_scale=torch.ones(1),
        )
    assert torch.equal(
        impl.forward(None, None, None, None, None, None, output),
        torch.zeros_like(output),
    )

    encoder = gems.AttentionFLImpl(
        4, 64, 0.5, 2, None, None, "auto",
        attn_type=gems.AttentionType.ENCODER_ONLY,
    )
    monkeypatch.setattr(
        encoder, "_forward_encoder_attention", lambda *a: "ENCODER"
    )
    meta = SimpleNamespace(num_actual_tokens=3)
    assert (
        encoder.forward(
            None,
            torch.zeros(3, 4, 64),
            torch.zeros(3, 2, 64),
            torch.zeros(3, 2, 64),
            None,
            meta,
            torch.empty(3, 4, 64),
        )
        == "ENCODER"
    )


def test_g48_source_ast_parses():
    here = pathlib.Path(__file__)
    ast.parse(here.read_text(encoding="utf-8"))
