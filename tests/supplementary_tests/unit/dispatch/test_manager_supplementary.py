# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for ``vllm_fl.dispatch.manager``.

The core suite (``tests/unit_tests/dispatch/test_manager.py``) covers
resolution order, policy filters, caching and fallback. The baseline still
misses:

* ``_record_default_flagos_op`` writing through a FlagGems-owned
  FileHandler (and its early-return / swallow paths);
* the post-fork reset (``_reset_after_fork``);
* ``ensure_initialized``'s init summary lines and the debug listing
  (``_print_registered_operators``) behind ``VLLM_FL_DISPATCH_DEBUG``;
* ``_matches_vendor_filters`` vendor None / deny / allow arms;
* ``_record_first_use`` first-use and switched-impl logging (and the
  DEFAULT-kind flagos recording);
* ``_resolve_impl``'s strict and non-strict no-selection errors and the
  dispatch-debug log line;
* ``_compute_candidates`` availability-check exception branch;
* ``resolve_candidates`` no-impl error and dedup across tokens;
* ``_call_with_hooks`` with IO dumping enabled (hook order, fn failure
  cleanup, hook exceptions swallowed);
* ``call``'s all-failed-earlier RuntimeError and the AMD-style
  no-retry arm (``_can_retry_failed_op``).

Everything runs against a fresh registry so no cross-test state leaks.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import pytest

from vllm_fl.dispatch import io_dumper
from vllm_fl.dispatch.manager import (
    OpManager,
    _can_retry_failed_op,
    _record_default_flagos_op,
    get_default_manager,
    reset_default_manager,
)
from vllm_fl.dispatch.policy import (
    SelectionPolicy,
    reset_global_policy,
    set_global_policy,
)
from vllm_fl.dispatch.registry import OpRegistry
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

pytestmark = [pytest.mark.supplementary]


def _impl(op, impl_id, fn=None, kind=BackendImplKind.REFERENCE,
          priority=BackendPriority.REFERENCE, vendor=None):
    return OpImpl(
        op_name=op,
        impl_id=impl_id,
        kind=kind,
        fn=fn or (lambda *a, **k: impl_id),
        priority=priority,
        vendor=vendor,
    )


@pytest.fixture(autouse=True)
def _policy_sandbox():
    reset_global_policy()
    yield
    reset_global_policy()


@pytest.fixture
def fresh_manager():
    """A manager over a private registry marked as initialized."""
    registry = OpRegistry()
    manager = OpManager(registry=registry)
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()
    return manager


def _disable_dumping(fn):
    """Run fn with the IO dumper forced off, restoring enabled state."""
    was_enabled = io_dumper.is_dump_enabled()
    if was_enabled:
        io_dumper.disable_io_dump()
    try:
        return fn()
    finally:
        if was_enabled:
            io_dumper.enable_io_dump()


class TestRecordDefaultFlagosOp:
    def _owned_handler(self, lines):
        class _Stream:
            def __init__(self):
                self.written = []

            def write(self, text):
                self.written.append(text)

            def flush(self):
                pass

        stream = _Stream()
        handler = logging.FileHandler(os.devnull)  # real FileHandler type
        handler.stream = stream  # swap in a memory stream
        handler._flaggems_owned = True
        return handler, stream

    def test_writes_through_flaggems_handlers_once(self, monkeypatch):
        import vllm_fl.dispatch.manager as mgr_mod

        handler, stream = self._owned_handler([])
        fg_logger = logging.getLogger("flag_gems")
        monkeypatch.setattr(fg_logger, "handlers", [handler])
        mgr_mod._RECORDED_FLAGOS_OPS.clear()
        impl = _impl("rms_norm", "flaggems.rms_norm", kind=BackendImplKind.DEFAULT)
        _record_default_flagos_op("rms_norm", impl)
        assert stream.written == [
            "[DEBUG] vllm_fl.dispatch.ops.rms_norm: flaggems.rms_norm\n"
        ]
        # Second call for the same (op, impl) is a no-op.
        _record_default_flagos_op("rms_norm", impl)
        assert len(stream.written) == 1

    def test_handler_write_failure_is_swallowed(self, monkeypatch):
        import vllm_fl.dispatch.manager as mgr_mod

        class _BadHandler(logging.FileHandler):
            pass

        bad = _BadHandler(os.devnull)
        bad._flaggems_owned = True

        def boom(text):
            raise OSError("handler closed")

        bad.stream = SimpleNamespace(write=boom, flush=lambda: None)
        fg_logger = logging.getLogger("flag_gems")
        monkeypatch.setattr(fg_logger, "handlers", [bad])
        mgr_mod._RECORDED_FLAGOS_OPS.clear()
        impl = _impl("silu_and_mul", "flaggems.silu", kind=BackendImplKind.DEFAULT)
        _record_default_flagos_op("silu_and_mul", impl)  # must not raise


class TestCanRetryFailedOp:
    def test_strict_policy_disables_retry(self):
        policy = SelectionPolicy(strict=True)
        set_global_policy(policy)
        assert _can_retry_failed_op("anything") is False

    def test_amd_sensitive_ops(self, monkeypatch):
        set_global_policy(SelectionPolicy(strict=False))
        fake_platform = SimpleNamespace(vendor_name="amd")
        monkeypatch.setitem(
            __import__("sys").modules, "vllm.platforms", SimpleNamespace(
                current_platform=fake_platform
            )
        )
        assert _can_retry_failed_op("rms_norm") is False
        assert _can_retry_failed_op("rotary_embedding") is False
        assert _can_retry_failed_op("matmul") is True

    def test_import_error_allows_retry(self, monkeypatch):
        import sys

        set_global_policy(SelectionPolicy(strict=False))
        monkeypatch.setitem(sys.modules, "vllm.platforms", None)
        # from-import of a None'd module raises ImportError -> retry allowed
        assert _can_retry_failed_op("rms_norm") is True


class TestResetAfterFork:
    def test_reset_clears_caches_and_bumps_epoch(self, fresh_manager):
        registry = fresh_manager._registry
        registry.register_impl(_impl("fork_op", "ref.a"))
        fresh_manager._dispatch_cache[("x", "p", 0)] = object()
        fresh_manager._candidates_cache[("x", "p", 0)] = [object()]
        fresh_manager._called_ops["fork_op"] = "ref.a"
        fresh_manager._failed_impls["fork_op"] = {"ref.a"}

        epoch = fresh_manager.policy_epoch
        fresh_manager._reset_after_fork()

        assert fresh_manager.policy_epoch == epoch + 1
        assert fresh_manager._dispatch_cache == {}
        assert fresh_manager._candidates_cache == {}
        assert fresh_manager._called_ops == {}
        assert fresh_manager._failed_impls == {}
        assert fresh_manager._state.initialized is False
        assert fresh_manager._state.init_pid == -1


class TestEnsureInitialized:
    def test_init_summary_and_builtin_registration(self, fresh_manager, caplog):
        fresh_manager._state.initialized = False
        fresh_manager._state.init_pid = -1
        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.manager"):
            fresh_manager.ensure_initialized()
        joined = "\n".join(r.message for r in caplog.records)
        assert "OpManager initialized:" in joined
        assert "ops with" in joined
        assert fresh_manager._state.initialized is True
        assert fresh_manager._state.init_pid == os.getpid()

    def test_second_call_is_short_circuit(self, fresh_manager, caplog):
        fresh_manager.ensure_initialized()
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.manager"):
            fresh_manager.ensure_initialized()  # pid matches -> return
        assert not any("OpManager initialized" in r.message for r in caplog.records)

    def test_debug_listing_printed_when_enabled(self, fresh_manager, caplog,
                                                monkeypatch):
        import vllm_fl.dispatch.manager as mgr_mod

        fresh_manager._registry.register_impl(
            _impl("dbg_op", "ref.one", kind=BackendImplKind.REFERENCE,
                  vendor="VND")
        )
        fresh_manager._registry.register_impl(_impl("dbg_op", "ref.two"))
        monkeypatch.setattr(mgr_mod, "_DISPATCH_DEBUG", True)
        fresh_manager._state.initialized = False
        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.manager"):
            fresh_manager.ensure_initialized()
        joined = "\n".join(r.message for r in caplog.records)
        assert "Registered Operators" in joined
        assert "dbg_op" in joined
        assert "ref.one" in joined and "ref.two" in joined
        assert "vendor=VND" in joined


class TestMatchesVendorFilters:
    def test_non_vendor_impl_passes_without_vendor(self, fresh_manager):
        impl = _impl("op", "ref.x", kind=BackendImplKind.REFERENCE)
        assert fresh_manager._matches_vendor_filters(
            impl, SelectionPolicy()
        ) is True

    def test_vendor_impl_without_vendor_rejected(self, fresh_manager):
        # Build the ill-formed impl via a raw dataclass replacement: VENDOR
        # kind without a vendor name. Construction validates, so replace the
        # frozen instance wholesale through a subclass-free object.__new__
        # dance using dataclasses.replace on a hand-filled shell.
        import dataclasses

        well_formed = _impl("op", "vendor.ok", kind=BackendImplKind.VENDOR,
                           vendor="OK")
        fields = {
            f.name: getattr(well_formed, f.name)
            for f in dataclasses.fields(well_formed)
        }
        fields["vendor"] = None  # simulate the corrupt state the guard tests
        impl = object.__new__(OpImpl)
        for name, value in fields.items():
            object.__setattr__(impl, name, value)
        assert fresh_manager._matches_vendor_filters(
            impl, SelectionPolicy()
        ) is False

    def test_deny_vendor_rejected(self, fresh_manager):
        impl = _impl("op", "vendor.x", kind=BackendImplKind.VENDOR, vendor="X")
        policy = SelectionPolicy(deny_vendors={"X"})
        assert fresh_manager._matches_vendor_filters(impl, policy) is False

    def test_allow_vendor_required(self, fresh_manager):
        impl = _impl("op", "vendor.x", kind=BackendImplKind.VENDOR, vendor="X")
        policy = SelectionPolicy(allow_vendors={"Y"})
        assert fresh_manager._matches_vendor_filters(impl, policy) is False
        policy2 = SelectionPolicy(allow_vendors={"X"})
        assert fresh_manager._matches_vendor_filters(impl, policy2) is True


class TestRecordFirstUse:
    def test_first_use_and_switch_logged(self, fresh_manager, caplog):
        registry = fresh_manager._registry
        registry.register_impl(_impl("switch_op", "ref.one"))
        registry.register_impl(_impl("switch_op", "ref.two"))
        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.manager"):
            fresh_manager._record_first_use("switch_op", registry.snapshot().impls_by_op["switch_op"][0])
        assert any("using 'ref.one'" in r.message for r in caplog.records)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.manager"):
            fresh_manager._record_first_use("switch_op", registry.snapshot().impls_by_op["switch_op"][1])
        assert any("switched from 'ref.one' to 'ref.two'" in r.message
                   for r in caplog.records)
        # Repeated recording of the same impl is a no-op.
        caplog.clear()
        fresh_manager._record_first_use("switch_op", registry.snapshot().impls_by_op["switch_op"][1])
        assert not caplog.records

    def test_default_kind_records_flagos(self, fresh_manager, monkeypatch, caplog):
        recorded = []

        monkeypatch.setattr(
            "vllm_fl.dispatch.manager._record_default_flagos_op",
            lambda op, impl: recorded.append((op, impl.impl_id)),
        )
        impl = _impl("fg_op", "flaggems.fg", kind=BackendImplKind.DEFAULT)
        fresh_manager._record_first_use("fg_op", impl)
        assert recorded == [("fg_op", "flaggems.fg")]
        assert fresh_manager._called_ops["fg_op"] == "flaggems.fg"


def _no_match_policy(strict: bool, op_name: str) -> SelectionPolicy:
    """A policy whose per-op order token matches no registered impl."""
    return SelectionPolicy(
        prefer="reference",
        strict=strict,
        per_op_order=((op_name, ("impl:nothing.matches",)),),
    )


class TestResolveErrors:
    def test_strict_policy_no_selection(self, fresh_manager):
        registry = fresh_manager._registry
        registry.register_impl(_impl("missing_op", "ref.hidden"))
        set_global_policy(_no_match_policy(True, "missing_op"))
        with pytest.raises(RuntimeError, match="strict policy"):
            fresh_manager.resolve("missing_op")

    def test_lenient_policy_no_selection(self, fresh_manager):
        registry = fresh_manager._registry
        registry.register_impl(_impl("orphan_op", "ref.hidden"))
        set_global_policy(_no_match_policy(False, "orphan_op"))
        with pytest.raises(RuntimeError, match="No implementation selected"):
            fresh_manager.resolve("orphan_op")

    def test_no_available_implementation(self, fresh_manager):
        registry = fresh_manager._registry

        def _never(*args, **kwargs):
            return "unused"

        _never._is_available = lambda: False
        impl = OpImpl(
            op_name="broken_op", impl_id="ref.never",
            kind=BackendImplKind.REFERENCE, fn=_never,
        )
        registry.register_impl(impl)
        with pytest.raises(RuntimeError, match="No available implementation"):
            fresh_manager.resolve("broken_op")

    def test_availability_check_exception_logged_and_skipped(self, fresh_manager,
                                                             caplog):
        """An is_available that raises is caught in OpImpl (returns False),
        so resolve falls through to the healthy impl without failing."""
        registry = fresh_manager._registry

        def _boom(*args, **kwargs):
            return "unused"

        def _exploding():
            raise RuntimeError("probe exploded")

        _boom._is_available = _exploding
        bad = OpImpl(
            op_name="explode_op", impl_id="ref.boom",
            kind=BackendImplKind.REFERENCE, fn=_boom,
        )
        good = _impl("explode_op", "ref.fine")
        registry.register_impl(bad)
        registry.register_impl(good)
        # The failing probe degrades to "not available" (Exception branch).
        assert bad.is_available() is False
        assert fresh_manager.resolve("explode_op") is good.fn
        # The manager-level exception branch (availability raising inside
        # _compute_candidates) is exercised with a stubbed instance whose
        # is_available raises directly.
        class _RaisingImpl:
            def __init__(self, base):
                self._base = base

            def __getattr__(self, name):
                return getattr(self._base, name)

            def is_available(self):
                raise RuntimeError("probe exploded")

        snap_impls = registry.snapshot().impls_by_op["explode_op"]
        with caplog.at_level(logging.WARNING,
                             logger="vllm_fl.dispatch.manager"):
            from vllm_fl.dispatch.manager import OpManager as _OM
            # Directly drive _compute_candidates with a raising stub.
            stub_registry = OpRegistry()
            stub_registry.register_impl(_impl("explode2", "ref.a"))
            stub_manager = OpManager(registry=stub_registry)
            stub_manager._state.initialized = True
            stub_manager._state.init_pid = os.getpid()
            real_snapshot = stub_registry.snapshot
            stub_registry.snapshot = lambda: SimpleNamespace(
                impls_by_op={"explode2": [
                    _RaisingImpl(_impl("explode2", "ref.a"))
                ]}
            )
            with pytest.raises(RuntimeError, match="No available implementation"):
                stub_manager._compute_candidates("explode2", SelectionPolicy())
        assert any("probe exploded" in r.getMessage() for r in caplog.records)

    def test_debug_resolve_log_line(self, fresh_manager, caplog, monkeypatch):
        import vllm_fl.dispatch.manager as mgr_mod

        impl = _impl("dbg_res", "ref.pick", vendor="VND")
        fresh_manager._registry.register_impl(impl)
        monkeypatch.setattr(mgr_mod, "_DISPATCH_DEBUG", True)
        fresh_manager._state.policy_epoch += 1  # invalidate any stale cache
        fresh_manager._dispatch_cache.clear()
        with caplog.at_level(logging.DEBUG, logger="vllm_fl.dispatch.manager"):
            fresh_manager.resolve("dbg_res")
        assert any("DISPATCH" in r.message and "dbg_res" in r.message
                   for r in caplog.records)


class TestResolveCandidates:
    def test_dedup_and_error(self, fresh_manager):
        registry = fresh_manager._registry
        a = _impl("cand_op", "ref.a", priority=BackendPriority.REFERENCE)
        b = _impl("cand_op", "ref.b", priority=BackendPriority.REFERENCE)
        registry.register_impl(a)
        registry.register_impl(b)
        candidates = fresh_manager.resolve_candidates("cand_op")
        assert {c.impl_id for c in candidates} == {"ref.a", "ref.b"}
        # A policy that dedups the same token set still yields each impl once.
        assert len(candidates) == 2
        registry.register_impl(_impl("ghost_op", "ref.ghost"))
        set_global_policy(_no_match_policy(False, "ghost_op"))
        with pytest.raises(RuntimeError, match="No implementation selected"):
            fresh_manager.resolve_candidates("ghost_op")

    def test_candidates_cached_by_epoch(self, fresh_manager):
        registry = fresh_manager._registry
        registry.register_impl(_impl("cache_op", "ref.only"))
        first = fresh_manager.resolve_candidates("cache_op")
        second = fresh_manager.resolve_candidates("cache_op")
        assert first is second  # same cached list object
        fresh_manager.bump_policy_epoch()
        third = fresh_manager.resolve_candidates("cache_op")
        assert third is not first


class TestCallWithHooks:
    def test_call_without_dump_runs_plain(self, fresh_manager):
        def fn(x, *, k=None):
            return x * 2

        assert fresh_manager._call_with_hooks("op", fn, (3,), {}) == 6

    def test_call_with_dump_enabled(self, fresh_manager, tmp_path):
        io_dumper.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        try:
            def fn(a, b):
                return a + b

            out = fresh_manager._call_with_hooks("hooked_op", fn, (1, 2), {})
            assert out == 3
            io_dumper._wait_and_flush()
            op_dir = tmp_path / "rank_0000" / "step_0000" / "hooked_op"
            assert (op_dir / "input.json").exists()
        finally:
            io_dumper.disable_io_dump()

    def test_call_fn_failure_cleans_dump_pairing(self, fresh_manager, tmp_path):
        io_dumper.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        try:
            def fn():
                raise ValueError("op exploded")

            with pytest.raises(ValueError, match="op exploded"):
                fresh_manager._call_with_hooks("failing_op", fn, (), {})
            stack = getattr(io_dumper._dump_pairing, "stack", {})
            assert all(not entries for entries in stack.values())
        finally:
            io_dumper.disable_io_dump()

    def test_hook_failures_swallowed(self, fresh_manager, monkeypatch, tmp_path):
        io_dumper.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        try:
            def boom(*args, **kwargs):
                raise RuntimeError("hook blew up")

            monkeypatch.setattr("vllm_fl.dispatch.manager.dump_before", boom)
            monkeypatch.setattr("vllm_fl.dispatch.manager.dump_after", boom)
            out = fresh_manager._call_with_hooks(
                "badhook_op", lambda: "value", (), {}
            )
            assert out == "value"
        finally:
            io_dumper.disable_io_dump()


class TestCallFallbackEdges:
    def test_all_failed_previously_raises(self, fresh_manager):
        registry = fresh_manager._registry
        registry.register_impl(_impl("dead_op", "ref.dead"))
        fresh_manager._failed_impls["dead_op"] = {"ref.dead"}
        with pytest.raises(RuntimeError, match="failed previously"):
            fresh_manager.call("dead_op")

    def test_all_candidates_fail_raises_with_last_error(self, fresh_manager):
        registry = fresh_manager._registry
        attempts = []

        def make_fn(name):
            def fn():
                attempts.append(name)
                raise RuntimeError(name)
            return fn

        registry.register_impl(_impl("multi_fail", "ref.one", fn=make_fn("one")))
        registry.register_impl(_impl("multi_fail", "ref.two", fn=make_fn("two")))
        with pytest.raises(RuntimeError, match="Last error"):
            fresh_manager.call("multi_fail")
        # Both implementations were attempted (order is impl_id sorted).
        assert sorted(attempts) == ["one", "two"]
        assert fresh_manager.get_failed_impls("multi_fail") == {
            "multi_fail": {"ref.one", "ref.two"}
        }

    def test_fallback_after_failure_succeeds(self, fresh_manager):
        registry = fresh_manager._registry
        registry.register_impl(
            _impl("fallback_op", "ref.bad", fn=lambda: (_ for _ in ()).throw(
                RuntimeError("first impl bad")))
        )
        registry.register_impl(_impl("fallback_op", "ref.good", fn=lambda: "OK"))
        result = fresh_manager.call("fallback_op")
        assert result == "OK"
        assert fresh_manager._called_ops["fallback_op"] == "ref.good"
        # A retry skips the recorded failed impl without raising.
        assert fresh_manager.call("fallback_op") == "OK"

    def test_get_selected_impl_id(self, fresh_manager):
        registry = fresh_manager._registry
        registry.register_impl(_impl("id_op", "ref.chosen"))
        assert fresh_manager.get_selected_impl_id("id_op") == "ref.chosen"


class TestDefaultManager:
    def test_reset_and_recreate(self):
        first = get_default_manager()
        reset_default_manager()
        second = get_default_manager()
        assert first is not second
        assert get_default_manager() is second
