# Copyright (c) 2026 BAAI. All rights reserved.

"""G51 dispatch extra coverage.

Targets (814b line):
- vllm_fl/dispatch/manager.py          (missing ~65 lines)
- vllm_fl/dispatch/discovery.py        (missing 38 lines)
- vllm_fl/dispatch/backends/reference/reference.py (missing 40 lines)

Pure-local tests: fake ``vllm.*`` modules are injected through ``sys.modules``
so the tests do not depend on a real vLLM install.  Torch is required only by
the reference-backend cases (torch2.10+cpu in the target env).
"""

from __future__ import annotations

import logging
import os
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

from vllm_fl.dispatch import discovery as D
from vllm_fl.dispatch import manager as M
from vllm_fl.dispatch.backends.reference.reference import ReferenceBackend
from vllm_fl.dispatch.manager import (
    OpManager,
    _can_retry_failed_op,
    _record_default_flagos_op,
)
from vllm_fl.dispatch.policy import (
    SelectionPolicy,
    reset_global_policy,
    set_global_policy,
)
from vllm_fl.dispatch.registry import OpRegistry
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_policy():
    reset_global_policy()
    yield
    reset_global_policy()


def _mk_manager(registry=None, initialized=True):
    mgr = OpManager(registry=registry if registry is not None else OpRegistry())
    if initialized:
        mgr._state.initialized = True
        mgr._state.init_pid = os.getpid()
    return mgr


def _impl(
    op_name,
    impl_id="reference.torch",
    kind=BackendImplKind.REFERENCE,
    fn=None,
    priority=BackendPriority.REFERENCE,
    vendor=None,
    available=None,
):
    fn = fn if fn is not None else (lambda *a, **k: None)
    if available is not None:
        fn._is_available = lambda: available
    return OpImpl(
        op_name=op_name,
        impl_id=impl_id,
        kind=kind,
        fn=fn,
        vendor=vendor,
        priority=priority,
    )


def _fake_module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


# ===========================================================================
# manager.py
# ===========================================================================


def test_can_retry_failed_op_import_error(monkeypatch):
    # Force the ``from vllm.platforms import current_platform`` to raise.
    monkeypatch.setitem(sys.modules, "vllm.platforms", None)
    assert _can_retry_failed_op("rms_norm") is True
    assert _can_retry_failed_op("rotary_embedding") is True


def test_register_at_fork_attribute_error(monkeypatch):
    def _boom(*args, **kwargs):
        raise AttributeError("register_at_fork unavailable")

    monkeypatch.setattr(M.os, "register_at_fork", _boom, raising=False)
    mgr = OpManager()
    assert mgr.registry is not None


def test_reset_after_fork_clears_state():
    mgr = _mk_manager()
    mgr._dispatch_cache[("a", None, 0)] = object()
    mgr._candidates_cache[("a", None, 0)] = [object()]
    mgr._called_ops["a"] = "reference.torch"
    mgr._failed_impls["a"] = {"reference.torch"}
    epoch = mgr._state.policy_epoch

    mgr._reset_after_fork()

    assert mgr._state.initialized is False
    assert mgr._state.init_pid == -1
    assert mgr._state.policy_epoch == epoch + 1
    assert not mgr._dispatch_cache
    assert not mgr._candidates_cache
    assert not mgr._called_ops
    assert not mgr._failed_impls


def test_get_failed_impls_unknown_op_returns_empty():
    mgr = _mk_manager()
    mgr._failed_impls["known"] = {"x"}
    assert mgr.get_failed_impls("missing") == {}
    assert mgr.get_failed_impls("known") == {"known": {"x"}}


def test_record_default_flagos_op_writes_and_swallows(tmp_path):
    fg = logging.getLogger("flag_gems")
    handler = logging.FileHandler(tmp_path / "flag_gems.log", mode="w", encoding="utf-8")
    handler._flaggems_owned = True
    fg.addHandler(handler)
    try:
        _record_default_flagos_op("g51_flagos_write", _impl("g51_flagos_write"))
        handler.flush()
        content = (tmp_path / "flag_gems.log").read_text(encoding="utf-8")
        assert "vllm_fl.dispatch.ops.g51_flagos_write" in content

        class _Boom:
            def write(self, *args, **kwargs):
                raise OSError("boom")

            def flush(self):
                raise OSError("boom")

        original_stream = handler.stream
        handler.stream = _Boom()
        try:
            # Must be swallowed by the diagnostics guard.
            _record_default_flagos_op(
                "g51_flagos_error", _impl("g51_flagos_error", impl_id="reference.other")
            )
        finally:
            handler.stream = original_stream
    finally:
        fg.removeHandler(handler)
        handler.close()


def test_ensure_initialized_debug_prints(monkeypatch):
    import vllm_fl.dispatch.builtin_ops as builtin_ops

    monkeypatch.setattr(builtin_ops, "register_builtins", lambda registry: None)
    monkeypatch.setattr(M, "_DISPATCH_DEBUG", True)

    registry = OpRegistry()
    registry.register_impl(_impl("dbg_op", impl_id="reference.ok", available=True))
    registry.register_impl(
        _impl(
            "dbg_op",
            impl_id="vendor.cuda",
            kind=BackendImplKind.VENDOR,
            vendor="cuda",
            priority=BackendPriority.VENDOR,
            available=True,
        )
    )
    registry.register_impl(_impl("dbg_op2", impl_id="reference.off", available=False))

    mgr = OpManager(registry=registry)
    mgr.ensure_initialized()
    assert mgr._state.initialized is True


def test_matches_vendor_filters_vendor_without_name():
    mgr = _mk_manager()
    fake = types.SimpleNamespace(kind=BackendImplKind.VENDOR, vendor=None)
    assert mgr._matches_vendor_filters(fake, SelectionPolicy()) is False

    ref = _impl("v")
    assert mgr._matches_vendor_filters(ref, SelectionPolicy()) is True


class _FlakyGetDict(dict):
    def __init__(self, final):
        super().__init__()
        self._final = final
        self._calls = 0

    def get(self, key, default=None):
        self._calls += 1
        if self._calls == 1:
            return None
        return self._final


def test_record_first_use_double_check_returns():
    mgr = _mk_manager()
    impl = _impl("dc_op")
    mgr._called_ops = _FlakyGetDict("reference.torch")
    mgr._record_first_use("dc_op", impl)
    assert mgr._called_ops._calls == 2
    assert mgr._called_ops == {}


def test_resolve_no_token_match_strict_and_non_strict():
    registry = OpRegistry()
    registry.register_impl(_impl("nm_op"))
    mgr = _mk_manager(registry)
    per_op = (("nm_op", ("impl:missing",)),)

    set_global_policy(SelectionPolicy(per_op_order=per_op, strict=True))
    with pytest.raises(RuntimeError, match="strict policy"):
        mgr.resolve("nm_op")

    set_global_policy(SelectionPolicy(per_op_order=per_op, strict=False))
    with pytest.raises(RuntimeError, match="No implementation selected"):
        mgr.resolve("nm_op")


def test_resolve_debug_branch(monkeypatch):
    registry = OpRegistry()
    registry.register_impl(_impl("dbg_res_op"))
    mgr = _mk_manager(registry)
    monkeypatch.setattr(M, "_DISPATCH_DEBUG", True)
    assert mgr.get_selected_impl_id("dbg_res_op") == "reference.torch"


class _RaisingImpl:
    op_name = "raise_op"
    impl_id = "raise.impl"
    kind = BackendImplKind.DEFAULT
    vendor = None

    def is_available(self):
        raise RuntimeError("availability boom")


def test_compute_candidates_swallows_availability_error():
    registry = OpRegistry()
    registry.register_impl(_RaisingImpl())
    registry.register_impl(_impl("raise_op", impl_id="reference.ok", available=True))
    mgr = _mk_manager(registry)

    cands = mgr._compute_candidates("raise_op", SelectionPolicy())
    assert [c.impl_id for c in cands] == ["reference.ok"]


def test_resolve_candidates_no_token_match_raises():
    registry = OpRegistry()
    registry.register_impl(_impl("rc_op"))
    mgr = _mk_manager(registry)
    set_global_policy(
        SelectionPolicy(per_op_order=(("rc_op", ("impl:missing",)),))
    )
    with pytest.raises(RuntimeError, match="No implementation selected"):
        mgr.resolve_candidates("rc_op")


def test_call_with_hooks_dump_path(monkeypatch):
    calls = []
    monkeypatch.setattr(M, "is_dump_enabled", lambda: True)
    monkeypatch.setattr(
        M, "dump_before", lambda *a, **k: calls.append(("before", a, k))
    )
    monkeypatch.setattr(M, "dump_after", lambda *a, **k: calls.append(("after", a)))
    monkeypatch.setattr(M, "dump_cleanup", lambda *a, **k: calls.append(("cleanup",)))

    mgr = _mk_manager()
    result = mgr._call_with_hooks("hook_ok", lambda x: x + 1, (1,), {})
    assert result == 2
    assert [c[0] for c in calls] == ["before", "after"]


def test_call_with_hooks_fn_failure_cleans_up(monkeypatch):
    calls = []

    def _cleanup_boom(*args, **kwargs):
        calls.append("cleanup")
        raise OSError("cleanup failed")

    monkeypatch.setattr(M, "is_dump_enabled", lambda: True)
    monkeypatch.setattr(M, "dump_before", lambda *a, **k: calls.append("before"))
    monkeypatch.setattr(M, "dump_after", lambda *a, **k: calls.append("after"))
    monkeypatch.setattr(M, "dump_cleanup", _cleanup_boom)

    def _fn(x):
        raise ValueError("op failed")

    mgr = _mk_manager()
    with pytest.raises(ValueError, match="op failed"):
        mgr._call_with_hooks("hook_fail", _fn, (1,), {})
    assert calls == ["before", "cleanup"]


def test_call_with_hooks_hook_failures_swallowed(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("hook boom")

    monkeypatch.setattr(M, "is_dump_enabled", lambda: True)
    monkeypatch.setattr(M, "dump_before", _boom)
    monkeypatch.setattr(M, "dump_after", _boom)
    monkeypatch.setattr(M, "dump_cleanup", lambda *a, **k: None)

    mgr = _mk_manager()
    assert mgr._call_with_hooks("hook_swallow", lambda x: x * 2, (3,), {}) == 6


def test_call_all_candidates_failed_previously():
    registry = OpRegistry()
    registry.register_impl(_impl("prefail_op", impl_id="reference.torch"))
    mgr = _mk_manager(registry)
    mgr._failed_impls["prefail_op"] = {"reference.torch"}
    with pytest.raises(RuntimeError, match="have failed previously"):
        mgr.call("prefail_op")


# ===========================================================================
# discovery.py
# ===========================================================================


def test_get_entry_points_import_styles(monkeypatch):
    import importlib.metadata as md

    # 1) both importlib.metadata and importlib_metadata unavailable
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "importlib.metadata", None)
        mp.setitem(sys.modules, "importlib_metadata", None)
        assert D._get_entry_points() == []

    # 2) Python 3.9 dict-like entry_points()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(md, "entry_points", lambda: {D.PLUGIN_GROUP: ["ep-dict"]})
        assert D._get_entry_points() == ["ep-dict"]

    # 3) object exposing only .get()
    class _GetOnly:
        def get(self, group, default=None):
            return ["ep-get"]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(md, "entry_points", lambda: _GetOnly())
        assert D._get_entry_points() == ["ep-get"]

    # 4) unsupported object -> empty fallback
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(md, "entry_points", lambda: object())
        assert D._get_entry_points() == []

    # 5) entry_points() itself raises -> guarded
    def _raise():
        raise RuntimeError("entry point failure")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(md, "entry_points", _raise)
        assert D._get_entry_points() == []


def test_call_register_function_register_raises():
    def _boom(registry):
        raise RuntimeError("register failed")

    obj = types.SimpleNamespace(vllm_fl_register=_boom)
    assert D._call_register_function(obj, MagicMock(), "src") is False


def test_discover_from_entry_points_branches(monkeypatch):
    D.clear_discovered_plugins()

    def _ok():
        return lambda registry: None

    def _no_register():
        return types.SimpleNamespace()

    def _load_boom():
        raise RuntimeError("load failed")

    eps = [
        types.SimpleNamespace(name="ep_ok", load=_ok),
        types.SimpleNamespace(name="ep_none", load=_no_register),
        types.SimpleNamespace(name="ep_boom", load=_load_boom),
    ]
    monkeypatch.setattr(D, "_get_entry_points", lambda: eps)

    loaded = D.discover_from_entry_points(MagicMock())

    assert loaded == 1
    discovered = {name: ok for name, _src, ok in D.get_discovered_plugins()}
    assert discovered == {"ep_ok": True, "ep_none": False, "ep_boom": False}
    D.clear_discovered_plugins()


def test_discover_from_env_modules_branches(monkeypatch):
    D.clear_discovered_plugins()

    def _fake_import(name):
        if name == "g51_good":
            return types.SimpleNamespace(register=lambda registry: None)
        if name == "g51_bad":
            return types.SimpleNamespace()
        raise RuntimeError("boom")

    monkeypatch.setattr(D.importlib, "import_module", _fake_import)
    monkeypatch.setenv(D.PLUGIN_MODULES_ENV, "g51_good,g51_bad,g51_boom")

    loaded = D.discover_from_env_modules(MagicMock())

    assert loaded == 1
    discovered = {name: ok for name, _src, ok in D.get_discovered_plugins()}
    assert discovered == {
        "g51_good": True,
        "g51_bad": False,
        "g51_boom": False,
    }
    D.clear_discovered_plugins()


# ===========================================================================
# backends/reference/reference.py
# ===========================================================================


def test_reference_name_and_availability(monkeypatch):
    backend = ReferenceBackend()
    assert backend.name == "reference"

    ReferenceBackend._available = None
    assert backend.is_available() is True

    # Force ``import torch`` inside is_available() to raise ImportError.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ReferenceBackend, "_available", None)
        mp.setitem(sys.modules, "torch", None)
        assert backend.is_available() is False
    assert ReferenceBackend._available is True


def test_reference_dynamic_per_token_quant_int8():
    x = torch.tensor([[1.0, -2.0, 3.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    quantized, scale = ReferenceBackend().dynamic_per_token_quant_int8(x)
    assert quantized.shape == x.shape
    assert quantized.dtype == torch.int8
    assert scale.shape == (2, 1)


def test_reference_silu_and_gelu_and_mul():
    backend = ReferenceBackend()
    x = torch.randn(2, 8)
    silu_out = backend.silu_and_mul(None, x)
    gelu_out = backend.gelu_and_mul(types.SimpleNamespace(approximate="none"), x)
    assert silu_out.shape == (2, 4)
    assert gelu_out.shape == (2, 4)


def test_reference_rms_norm():
    backend = ReferenceBackend()
    x = torch.randn(2, 4)
    obj = types.SimpleNamespace(variance_epsilon=1e-6, weight=torch.ones(4))
    out = backend.rms_norm(obj, x)
    assert out.shape == (2, 4)


def test_reference_rotary_embedding():
    backend = ReferenceBackend()
    query = torch.randn(3, 2, 4)
    key = torch.randn(3, 2, 4)
    cos = torch.randn(5, 4)
    sin = torch.randn(5, 4)
    position_ids = torch.tensor([0, 1, 2], dtype=torch.int64)
    q_out, k_out = backend.rotary_embedding(
        None, query, key, cos, sin, position_ids
    )
    assert q_out.shape == query.shape
    assert k_out.shape == key.shape


def test_reference_attention_backend_paths(monkeypatch):
    class _AttentionBackendEnum:
        FLASHMLA_SPARSE = types.SimpleNamespace(get_path=lambda: "flashmla_sparse")
        FLASHMLA = types.SimpleNamespace(get_path=lambda: "flashmla")
        FLASH_ATTN = types.SimpleNamespace(get_path=lambda: "flash_attn")

    fake = _fake_module(
        "vllm.v1.attention.backends.registry",
        AttentionBackendEnum=_AttentionBackendEnum,
    )
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.registry", fake)

    backend = ReferenceBackend()
    assert backend.attention_backend() == "flash_attn"
    assert backend.attention_backend(use_mla=True) == "flashmla"
    assert backend.attention_backend(use_mla=True, use_sparse=True) == "flashmla_sparse"




def test_reference_moe_align_block_size_pad_sorted():
    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    sorted_ids, expert_ids, num_tokens_post_padded = (
        ReferenceBackend().moe_align_block_size(
            topk_ids,
            block_size=2,
            num_experts=4,
            expert_map=None,
            pad_sorted_ids=True,
        )
    )
    assert sorted_ids.dtype == torch.int32
    assert expert_ids.dtype == torch.int32
    assert num_tokens_post_padded.shape == (1,)
    assert int(num_tokens_post_padded[0]) % 2 == 0
    assert sorted_ids.numel() == expert_ids.numel() * 2


def test_reference_moe_sum():
    inp = torch.randn(6, 2, 4)
    out = torch.zeros(6, 4)
    ReferenceBackend().moe_sum(inp, out)
    assert torch.allclose(out, inp.sum(dim=1))
