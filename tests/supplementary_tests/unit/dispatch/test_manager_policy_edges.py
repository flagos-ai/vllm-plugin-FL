# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for ``vllm_fl.dispatch.manager`` and ``policy`` edges.

The core suite (``tests/unit_tests/dispatch/test_manager.py`` /
``test_policy.py``) covers the main dispatch-selection flow.  This module
targets the gaps that remain uncovered in the supplementary suite only:

Manager (``vllm_fl/dispatch/manager.py``):

* ``_can_retry_failed_op`` — every branch, including the ImportError
  fallback and the AMD vendor gate;
* ``_record_default_flagos_op`` — dedup, FlagGems-owned FileHandler
  pass-through, and the swallow-everything guard;
* ``_reset_after_fork`` / ``bump_policy_epoch`` / ``clear_failed_impls`` /
  ``get_failed_impls`` state transitions;
* ``ensure_initialized`` idempotence and epoch invalidation (with a
  ``vllm.platforms`` stub on Windows so vendor detection succeeds);
* ``_matches_vendor_filters`` incl. the vendor=None VENDOR edge;
* ``_record_first_use`` first-use/switch logging and flagos recording;
* ``_resolve_impl`` / ``_compute_candidates`` / ``resolve_candidates``
  error paths (strict, no-token-match, availability exceptions);
* ``call`` fallback flow: primary failure warning, fallback log, all-failed
  RuntimeError with ``__cause__``, all-previously-failed, and the strict
  direct path;
* ``_call_with_hooks`` with IO dumping enabled (exec order + op tags) and
  the fn-raise cleanup path;
* ``get_default_manager`` / ``reset_default_manager``.

Policy (``vllm_fl/dispatch/policy.py``):

* ``SelectionPolicy`` — ``from_dict`` normalization, ``per_op_order_dict``,
  ``get_per_op_order``, ``get_default_order``, ``is_vendor_allowed``,
  ``fingerprint`` branches, and hash/equality;
* ``PolicyManager`` — singleton init guard, ``get_instance``, epoch bumps,
  ``get/set/reset_global_policy``, ``create_policy_context``;
* parsers — ``_parse_csv_set``, ``_parse_per_op``, ``_parse_op_config``;
* ``_policy_from_config`` — missing file, invalid YAML, invalid prefer,
  list vs csv-string deny/allow, list vs string op_backends;
* ``_policy_from_env`` — ``VLLM_FL_CONFIG`` override, platform config via
  ``VLLM_FL_PLATFORM``, invalid strict warning, env overrides;
* ``_PolicyContext`` enter/exit epoch bumps and the module-level
  convenience context managers.
"""

import json
import logging
import os
import sys
import types

import pytest
import torch

import vllm_fl.dispatch.manager as mgr_mod
from vllm_fl.dispatch.manager import OpManager
from vllm_fl.dispatch.policy import (
    PREFER_DEFAULT,
    PREFER_REFERENCE,
    PREFER_VENDOR,
    PolicyManager,
    SelectionPolicy,
    get_policy,
    policy_from_config,
    policy_from_env,
    reset_global_policy,
    set_global_policy,
    with_allowed_vendors,
    with_denied_vendors,
    with_preference,
    with_strict_mode,
)
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

pytestmark = [pytest.mark.supplementary]


def _make_impl(
    op_name, impl_id, kind, *, fn=None, vendor=None, priority=None, available=True
):
    """Build an OpImpl with an attached availability probe."""
    if fn is None:

        def fn(*args, **kwargs):  # noqa: ANN002, ANN003
            return f"{impl_id}-result"

    if available is True:
        fn._is_available = lambda: True
    else:
        fn._is_available = lambda: False
    if priority is None:
        priority = {
            BackendImplKind.DEFAULT: BackendPriority.DEFAULT,
            BackendImplKind.VENDOR: BackendPriority.VENDOR,
            BackendImplKind.REFERENCE: BackendPriority.REFERENCE,
        }[kind]
    return OpImpl(
        op_name=op_name,
        impl_id=impl_id,
        kind=kind,
        fn=fn,
        vendor=vendor,
        priority=priority,
    )


@pytest.fixture(autouse=True)
def _fresh_manager_state(clean_fl_env, monkeypatch):
    """Give each test a pristine policy/manager/flagos-record state.

    ``clean_fl_env`` (from the supplementary conftest) deletes all
    ``VLLM_FL_*`` env vars for the duration of the test and restores them
    afterwards, so platform/env-derived policies are the built-in defaults.
    """
    saved_recorded = set(mgr_mod._RECORDED_FLAGOS_OPS)
    saved_platform = sys.modules.get("vllm.platforms")
    PolicyManager.get_instance().reset_global_policy()
    yield
    mgr_mod._RECORDED_FLAGOS_OPS.clear()
    mgr_mod._RECORDED_FLAGOS_OPS.update(saved_recorded)
    if saved_platform is None:
        monkeypatch.delitem(sys.modules, "vllm.platforms", raising=False)
    else:
        monkeypatch.setitem(sys.modules, "vllm.platforms", saved_platform)
    PolicyManager.get_instance().reset_global_policy()


@pytest.fixture()
def stubbed_vllm_platforms(monkeypatch):
    """Install a minimal vllm.platforms stub (Windows has no vllm).

    On Linux with real vLLM the real module is restored afterwards.
    """
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = types.SimpleNamespace(vendor_name="nvidia")
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)


class TestCanRetryFailedOp:
    """_can_retry_failed_op branch matrix."""

    @pytest.mark.supplementary
    def test_strict_policy_disables_retry(self):
        """A strict policy never retries, regardless of op or platform."""
        with with_strict_mode():
            assert mgr_mod._can_retry_failed_op("rms_norm") is False
            assert mgr_mod._can_retry_failed_op("anything_else") is False

    @pytest.mark.supplementary
    def test_unlisted_ops_always_retry(self):
        """Ops outside (rms_norm, rotary_embedding) retry even on AMD."""
        assert mgr_mod._can_retry_failed_op("silu_and_mul") is True
        assert mgr_mod._can_retry_failed_op("topk_softmax") is True

    @pytest.mark.supplementary
    def test_import_error_falls_back_to_retry(self, monkeypatch):
        """A missing vllm.platforms module keeps retry enabled."""
        monkeypatch.setitem(sys.modules, "vllm.platforms", None)
        assert mgr_mod._can_retry_failed_op("rms_norm") is True
        assert mgr_mod._can_retry_failed_op("rotary_embedding") is True

    @pytest.mark.supplementary
    def test_nvidia_vendor_allows_retry(self, stubbed_vllm_platforms):
        """NVIDIA kernels report errors before mutating inputs."""
        assert mgr_mod._can_retry_failed_op("rms_norm") is True
        assert mgr_mod._can_retry_failed_op("rotary_embedding") is True

    @pytest.mark.supplementary
    def test_amd_vendor_blocks_retry(self, monkeypatch):
        """AMD kernels can mutate inputs before failing — no retry."""
        platforms = types.ModuleType("vllm.platforms")
        platforms.current_platform = types.SimpleNamespace(vendor_name="amd")
        monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
        assert mgr_mod._can_retry_failed_op("rms_norm") is False
        assert mgr_mod._can_retry_failed_op("rotary_embedding") is False


class TestRecordDefaultFlagosOp:
    """_record_default_flagos_op: FileHandler pass-through and guards."""

    @pytest.mark.supplementary
    def test_writes_through_flaggems_owned_filehandler(self, tmp_path):
        """A DEFAULT impl appends a '[DEBUG] vllm_fl.dispatch.ops...' line."""
        oplist = tmp_path / "oplist.txt"
        handler = logging.FileHandler(oplist, mode="w", encoding="utf-8")
        handler._flaggems_owned = True  # noqa: SLF001 - marker of ownership
        fg_logger = logging.getLogger("flag_gems")
        fg_logger.addHandler(handler)
        try:
            impl = _make_impl("rms_norm", "flaggems.default", BackendImplKind.DEFAULT)
            mgr_mod._record_default_flagos_op("rms_norm", impl)
        finally:
            fg_logger.removeHandler(handler)
            handler.close()
        content = oplist.read_text(encoding="utf-8")
        assert content == ("[DEBUG] vllm_fl.dispatch.ops.rms_norm: flaggems.default\n")

    @pytest.mark.supplementary
    def test_dedup_by_op_and_impl(self, tmp_path):
        """A repeated (op_name, impl_id) pair writes only once."""
        oplist = tmp_path / "oplist2.txt"
        handler = logging.FileHandler(oplist, mode="w", encoding="utf-8")
        handler._flaggems_owned = True
        fg_logger = logging.getLogger("flag_gems")
        fg_logger.addHandler(handler)
        try:
            impl = _make_impl("rms_norm", "flaggems.default", BackendImplKind.DEFAULT)
            mgr_mod._record_default_flagos_op("rms_norm", impl)
            mgr_mod._record_default_flagos_op("rms_norm", impl)
            assert ("rms_norm", "flaggems.default") in mgr_mod._RECORDED_FLAGOS_OPS
            # A different impl_id for the same op still writes.
            other = _make_impl("rms_norm", "flaggems.alt", BackendImplKind.DEFAULT)
            mgr_mod._record_default_flagos_op("rms_norm", other)
        finally:
            fg_logger.removeHandler(handler)
            handler.close()
        lines = oplist.read_text(encoding="utf-8").splitlines()
        assert lines == [
            "[DEBUG] vllm_fl.dispatch.ops.rms_norm: flaggems.default",
            "[DEBUG] vllm_fl.dispatch.ops.rms_norm: flaggems.alt",
        ]

    @pytest.mark.supplementary
    def test_closed_handler_exception_is_swallowed(self, tmp_path):
        """I/O failures never propagate (diagnostics must not break runs)."""
        handler = logging.FileHandler(tmp_path / "x.txt", mode="w", encoding="utf-8")
        handler._flaggems_owned = True
        fg_logger = logging.getLogger("flag_gems")
        fg_logger.addHandler(handler)
        handler.close()  # writing to a closed FileHandler raises ValueError
        try:
            impl = _make_impl("opx", "flaggems.default", BackendImplKind.DEFAULT)
            mgr_mod._record_default_flagos_op("opx", impl)  # must not raise
        finally:
            fg_logger.removeHandler(handler)

    @pytest.mark.supplementary
    def test_non_file_handlers_are_ignored(self, tmp_path, caplog):
        """StreamHandlers and unmarked FileHandlers get no writes."""
        marker = types.SimpleNamespace()
        marker.written = []

        class PseudoStreamHandler(logging.StreamHandler):
            def __init__(self):
                super().__init__(io_dummy)
                self._flaggems_owned = True

        io_dummy = types.SimpleNamespace(
            write=marker.written.append, flush=lambda: None
        )
        fg_logger = logging.getLogger("flag_gems")
        handler = PseudoStreamHandler()
        fg_logger.addHandler(handler)
        try:
            impl = _make_impl("opy", "flaggems.default", BackendImplKind.DEFAULT)
            mgr_mod._record_default_flagos_op("opy", impl)
            assert marker.written == []  # not a FileHandler -> skipped
            assert ("opy", "flaggems.default") in mgr_mod._RECORDED_FLAGOS_OPS
        finally:
            fg_logger.removeHandler(handler)


class TestManagerStateTransitions:
    """Fork reset, epoch bumps, failed-impl bookkeeping, default manager."""

    @staticmethod
    def _primed(op="opA", impl_id="ref.one"):
        """A manager whose state looks initialized in this process."""
        manager = OpManager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()
        manager._state.policy_epoch = 3
        manager._dispatch_cache[("k",)] = "stale"
        manager._candidates_cache[("k",)] = ["stale"]
        manager._called_ops[op] = impl_id
        manager._failed_impls[op] = {impl_id}
        return manager

    @pytest.mark.supplementary
    def test_reset_after_fork_clears_everything(self):
        """_reset_after_fork un-initializes and invalidates all caches."""
        manager = self._primed()
        manager._reset_after_fork()
        assert manager._state.initialized is False
        assert manager._state.init_pid == -1
        assert manager._state.policy_epoch == 4
        assert manager._dispatch_cache == {}
        assert manager._candidates_cache == {}
        assert manager._called_ops == {}
        assert manager._failed_impls == {}

    @pytest.mark.supplementary
    def test_bump_policy_epoch_invalidates_caches(self):
        """bump_policy_epoch clears dispatch/candidates/failed caches."""
        manager = self._primed()
        assert manager.policy_epoch == 3
        manager.bump_policy_epoch()
        assert manager.policy_epoch == 4
        assert manager._dispatch_cache == {}
        assert manager._candidates_cache == {}
        assert manager._failed_impls == {}

    @pytest.mark.supplementary
    def test_clear_failed_impls_all_and_single(self):
        """clear_failed_impls with no arg clears all; with an op, just that."""
        manager = self._primed()
        manager._failed_impls["opB"] = {"x"}
        manager.clear_failed_impls("opA")
        assert manager.get_failed_impls() == {"opB": {"x"}}
        manager.clear_failed_impls()  # None → all
        assert manager.get_failed_impls() == {}
        manager.clear_failed_impls("never-had")  # unknown op: no error

    @pytest.mark.supplementary
    def test_get_failed_impls_scoping_and_copies(self):
        """Per-op view returns a copy; mutating it does not affect state."""
        manager = self._primed()
        manager._failed_impls["opB"] = {"y"}
        assert manager.get_failed_impls("opA") == {"opA": {"ref.one"}}
        assert manager.get_failed_impls("missing") == {}
        view = manager.get_failed_impls()
        view["opA"].add("mutated")
        assert manager.get_failed_impls("opA") == {"opA": {"ref.one"}}

    @pytest.mark.supplementary
    def test_registry_property(self):
        """registry exposes the underlying registry instance."""
        manager = OpManager()
        assert manager.registry is manager._registry

    @pytest.mark.supplementary
    def test_default_manager_get_and_reset(self):
        """get_default_manager caches; reset_default_manager clears."""
        first = mgr_mod.get_default_manager()
        assert mgr_mod.get_default_manager() is first
        mgr_mod.reset_default_manager()
        second = mgr_mod.get_default_manager()
        assert second is not first


class TestEnsureInitialized:
    """ensure_initialized: idempotence, re-init after fork, epoch bump."""

    @pytest.mark.supplementary
    def test_idempotent_when_already_initialized(self, stubbed_vllm_platforms):
        """A second call in the same process is a no-op."""
        manager = OpManager()
        manager.ensure_initialized()
        ops_before = set(manager._registry.snapshot().impls_by_op)
        epoch_before = manager.policy_epoch
        manager.ensure_initialized()
        assert set(manager._registry.snapshot().impls_by_op) == ops_before
        assert manager.policy_epoch == epoch_before

    @pytest.mark.supplementary
    def test_initializes_and_bumps_epoch(self, stubbed_vllm_platforms):
        """First init registers builtins and bumps the policy epoch."""
        manager = OpManager()
        assert manager.policy_epoch == 0
        manager.ensure_initialized()
        assert manager._state.initialized is True
        assert manager._state.init_pid == os.getpid()
        assert manager.policy_epoch == 1
        snap = manager._registry.snapshot().impls_by_op
        assert "silu_and_mul" in snap
        assert "rms_norm" in snap
        # Every op has at least one implementation registered.
        assert all(impls for impls in snap.values())

    @pytest.mark.supplementary
    def test_reinit_after_fork_reset(self, stubbed_vllm_platforms, caplog):
        """After a fork reset, ensure_initialized re-registers everything."""
        manager = OpManager()
        manager.ensure_initialized()
        count_before = len(manager._registry.snapshot().impls_by_op)
        manager._reset_after_fork()
        with caplog.at_level(logging.WARNING):
            manager.ensure_initialized()  # duplicate registrations warn
        assert manager._state.initialized is True
        assert len(manager._registry.snapshot().impls_by_op) == count_before


class TestVendorFilters:
    """_matches_vendor_filters and _record_first_use."""

    @pytest.mark.supplementary
    def test_non_vendor_kinds_pass_without_checks(self):
        """DEFAULT/REFERENCE impls skip vendor filtering entirely."""
        manager = OpManager()
        policy = SelectionPolicy.from_dict(deny_vendors={"cuda"})
        assert (
            manager._matches_vendor_filters(
                _make_impl("op", "flagos.x", BackendImplKind.DEFAULT), policy
            )
            is True
        )
        assert (
            manager._matches_vendor_filters(
                _make_impl("op", "ref.x", BackendImplKind.REFERENCE), policy
            )
            is True
        )

    @pytest.mark.supplementary
    def test_vendor_with_none_vendor_name_rejected(self):
        """A VENDOR impl with vendor=None is filtered out."""
        manager = OpManager()
        policy = SelectionPolicy()
        impl = _make_impl("op", "vendor.x", BackendImplKind.VENDOR, vendor="cuda")
        object.__setattr__(impl, "vendor", None)  # simulate missing vendor
        assert manager._matches_vendor_filters(impl, policy) is False

    @pytest.mark.supplementary
    def test_deny_and_allow_lists(self):
        """Deny removes; allow whitelists; None allow permits all."""
        manager = OpManager()
        impl = _make_impl("op", "vendor.cuda", BackendImplKind.VENDOR, vendor="cuda")
        assert (
            manager._matches_vendor_filters(
                impl, SelectionPolicy.from_dict(deny_vendors={"cuda"})
            )
            is False
        )
        assert (
            manager._matches_vendor_filters(
                impl, SelectionPolicy.from_dict(deny_vendors={"ascend"})
            )
            is True
        )
        assert (
            manager._matches_vendor_filters(
                impl,
                SelectionPolicy.from_dict(allow_vendors={"ascend"}),
            )
            is False
        )
        assert (
            manager._matches_vendor_filters(
                impl, SelectionPolicy.from_dict(allow_vendors={"cuda"})
            )
            is True
        )

    @pytest.mark.supplementary
    def test_default_order_delegates_to_policy(self):
        """_default_order forwards to the policy's get_default_order."""
        manager = OpManager()
        assert manager._default_order(SelectionPolicy()) == [
            "flagos",
            "vendor",
            "reference",
        ]
        assert manager._default_order(SelectionPolicy(prefer=PREFER_VENDOR)) == [
            "vendor",
            "flagos",
            "reference",
        ]

    @pytest.mark.supplementary
    def test_record_first_use_logs_first_and_switch(self, caplog):
        """First use logs 'using'; a change logs 'switched from'."""
        manager = OpManager()
        first = _make_impl("op", "ref.one", BackendImplKind.REFERENCE)
        second = _make_impl("op", "ref.two", BackendImplKind.REFERENCE)
        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.manager"):
            manager._record_first_use("op", first)
            manager._record_first_use("op", first)  # same: silent
            manager._record_first_use("op", second)
        messages = [r.getMessage() for r in caplog.records]
        assert any(
            "Op 'op' using 'ref.one' (kind=reference, vendor=None)" in m
            for m in messages
        )
        assert any(
            "Op 'op' switched from 'ref.one' to 'ref.two' "
            "(kind=reference, vendor=None)" in m
            for m in messages
        )
        # 'using' logged once and 'switched' once (silent re-use skipped).
        assert len(messages) == 2
        assert manager._called_ops["op"] == "ref.two"

    @pytest.mark.supplementary
    def test_record_first_use_marks_failed_impl(self):
        """_mark_failed_impl accumulates impl ids per op."""
        manager = OpManager()
        manager._mark_failed_impl("op", "a")
        manager._mark_failed_impl("op", "a")  # set semantics: no duplicate
        manager._mark_failed_impl("op", "b")
        assert manager.get_failed_impls("op") == {"op": {"a", "b"}}


class TestResolveEdges:
    """_resolve_impl / _compute_candidates / resolve_candidates error paths."""

    @pytest.mark.supplementary
    def test_resolve_impl_strict_raise(self):
        """Strict + no matching candidate raises 'under strict policy'."""
        manager = OpManager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()
        manager._registry.register_impl(
            _make_impl("opx", "ref.one", BackendImplKind.REFERENCE)
        )
        # Strict + a per-op order whose tokens match none of the
        # registered candidates → the strict-specific error message.
        policy = SelectionPolicy.from_dict(
            strict=True, per_op_order={"opx": ["impl:does-not-exist"]}
        )
        set_global_policy(policy)
        try:
            with pytest.raises(RuntimeError, match="under strict policy"):
                manager._resolve_impl("opx")
        finally:
            reset_global_policy()

    @pytest.mark.supplementary
    def test_resolve_impl_no_token_match(self):
        """Candidates exist but none match the per-op order tokens."""
        manager = OpManager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()
        manager._registry.register_impl(
            _make_impl("opx", "ref.one", BackendImplKind.REFERENCE)
        )
        policy = SelectionPolicy.from_dict(
            per_op_order={"opx": ["impl:does-not-exist"]}
        )
        set_global_policy(policy)
        try:
            with pytest.raises(RuntimeError, match="No implementation selected"):
                manager._resolve_impl("opx")
        finally:
            reset_global_policy()

    @pytest.mark.supplementary
    def test_compute_candidates_unavailable_and_raising(self):
        """Unavailable and raising-probe impls are both dropped."""
        manager = OpManager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()

        def boom():
            raise RuntimeError("probe crashed")

        bad = _make_impl("opy", "vendor.bad", BackendImplKind.VENDOR, vendor="cuda")
        bad.fn._is_available = boom
        manager._registry.register_impl(bad)
        manager._registry.register_impl(
            _make_impl(
                "opy",
                "vendor.off",
                BackendImplKind.VENDOR,
                vendor="cuda",
                available=False,
            )
        )
        manager._registry.register_impl(
            _make_impl("opy", "ref.ok", BackendImplKind.REFERENCE)
        )
        # The raising probe is swallowed by OpImpl.is_available (returns
        # False), so both vendor impls are filtered out here.
        candidates = manager._compute_candidates("opy", SelectionPolicy())
        assert [c.impl_id for c in candidates] == ["ref.ok"]

    @pytest.mark.supplementary
    def test_compute_candidates_none_available_raises(self):
        """Zero available candidates raises 'No available implementation'."""
        manager = OpManager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()
        manager._registry.register_impl(
            _make_impl("opz", "ref.off", BackendImplKind.REFERENCE, available=False)
        )
        with pytest.raises(RuntimeError, match="No available implementation"):
            manager._compute_candidates("opz", SelectionPolicy())

    @pytest.mark.supplementary
    def test_resolve_candidates_dedup_and_no_match(self, stubbed_vllm_platforms):
        """resolve_candidates dedups across tokens and raises when empty."""
        manager = OpManager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()
        manager._registry.register_impl(
            _make_impl("opw", "ref.one", BackendImplKind.REFERENCE)
        )
        manager._registry.register_impl(
            _make_impl("opw", "flagos.two", BackendImplKind.DEFAULT)
        )
        result = manager.resolve_candidates("opw")
        # Default order flagos first; both candidates returned.
        assert [i.impl_id for i in result] == ["flagos.two", "ref.one"]
        # Cached second call returns the same list object.
        assert manager.resolve_candidates("opw") is result

        policy = SelectionPolicy.from_dict(per_op_order={"opw": ["impl:missing"]})
        set_global_policy(policy)
        try:
            with pytest.raises(RuntimeError, match="No implementation selected"):
                manager.resolve_candidates("opw")
        finally:
            reset_global_policy()

    @pytest.mark.supplementary
    def test_resolve_returns_fn_and_selected_impl_id(self):
        """resolve() returns the callable; get_selected_impl_id its id."""
        manager = OpManager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()
        manager._registry.register_impl(
            _make_impl("opq", "ref.one", BackendImplKind.REFERENCE)
        )
        fn = manager.resolve("opq")
        assert fn() == "ref.one-result"
        assert manager.get_selected_impl_id("opq") == "ref.one"


class TestCallFallbackFlow:
    """OpManager.call: fallback, strict, and all-failure paths."""

    @staticmethod
    def _manager_with(op, impls):
        manager = OpManager()
        manager._state.initialized = True
        manager._state.init_pid = os.getpid()
        for impl in impls:
            manager._registry.register_impl(impl)
        return manager

    @pytest.mark.supplementary
    def test_call_falls_back_and_updates_called_ops(self, caplog):
        """Primary failure warns; fallback succeeds and updates _called_ops."""

        def primary(*args, **kwargs):
            raise ValueError("primary boom")

        impls = [
            _make_impl("opc", "flagos.primary", BackendImplKind.DEFAULT, fn=primary),
            _make_impl("opc", "ref.fallback", BackendImplKind.REFERENCE),
        ]
        manager = self._manager_with("opc", impls)
        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.manager"):
            result = manager.call("opc", 1, 2)
        assert result == "ref.fallback-result"
        messages = [r.getMessage() for r in caplog.records]
        assert any(
            "Implementation 'flagos.primary' failed for op 'opc'" in m for m in messages
        )
        assert any(
            "Op 'opc' fallback to 'ref.fallback' (kind=reference, vendor=None)" in m
            for m in messages
        )
        assert manager._called_ops["opc"] == "ref.fallback"
        assert manager.get_failed_impls("opc") == {"opc": {"flagos.primary"}}

    @pytest.mark.supplementary
    def test_call_all_fail_raises_with_cause(self, caplog):
        """Every candidate failing raises RuntimeError chained to the last."""

        def explode(msg):
            def fn(*args, **kwargs):
                raise ValueError(msg)

            return fn

        impls = [
            _make_impl(
                "opd", "flagos.one", BackendImplKind.DEFAULT, fn=explode("one failed")
            ),
            _make_impl(
                "opd", "ref.two", BackendImplKind.REFERENCE, fn=explode("two failed")
            ),
        ]
        manager = self._manager_with("opd", impls)
        caplog.set_level(logging.ERROR, logger="vllm_fl.dispatch.manager")
        with pytest.raises(RuntimeError, match="All 2 implementation") as excinfo:
            manager.call("opd")
        error_records = [r.getMessage() for r in caplog.records]
        assert any(
            "Last implementation 'ref.two' failed for op 'opd'" in m
            for m in error_records
        )
        # The raised RuntimeError chains the last ValueError via __cause__.
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert str(excinfo.value.__cause__) == "two failed"

    @pytest.mark.supplementary
    def test_call_all_previously_failed(self):
        """When every impl already failed before, call raises immediately."""

        def explode(*args, **kwargs):
            raise ValueError("boom")

        impls = [
            _make_impl("ope", "flagos.one", BackendImplKind.DEFAULT, fn=explode),
        ]
        manager = self._manager_with("ope", impls)
        manager._failed_impls["ope"] = {"flagos.one"}
        with pytest.raises(RuntimeError, match="have failed previously"):
            manager.call("ope")

    @pytest.mark.supplementary
    def test_call_strict_raises_original(self):
        """In strict mode the primary's original exception propagates."""

        def primary(*args, **kwargs):
            raise ValueError("strict boom")

        impls = [
            _make_impl("opf", "flagos.primary", BackendImplKind.DEFAULT, fn=primary),
            _make_impl("opf", "ref.fallback", BackendImplKind.REFERENCE),
        ]
        manager = self._manager_with("opf", impls)
        with with_strict_mode(), pytest.raises(ValueError, match="strict boom"):
            manager.call("opf")
        # The fallback was never attempted (and never recorded as failed).
        assert manager.get_failed_impls("opf") == {}

    @pytest.mark.supplementary
    def test_call_success_records_default_flagos_op(self, tmp_path):
        """A successful DEFAULT impl call records into the flagos oplist."""
        handler = logging.FileHandler(tmp_path / "op.txt", mode="w", encoding="utf-8")
        handler._flaggems_owned = True
        fg_logger = logging.getLogger("flag_gems")
        fg_logger.addHandler(handler)
        try:
            impls = [
                _make_impl("opg", "flaggems.default", BackendImplKind.DEFAULT),
            ]
            manager = self._manager_with("opg", impls)
            assert manager.call("opg") == "flaggems.default-result"
        finally:
            fg_logger.removeHandler(handler)
            handler.close()
        assert ("opg", "flaggems.default") in mgr_mod._RECORDED_FLAGOS_OPS
        content = (tmp_path / "op.txt").read_text(encoding="utf-8")
        assert "[DEBUG] vllm_fl.dispatch.ops.opg: flaggems.default" in content


class TestCallWithHooks:
    """_call_with_hooks dump integration and the fn-raise cleanup path."""

    @pytest.mark.supplementary
    def test_hooks_disabled_passthrough(self):
        """Without dumping, _call_with_hooks is a plain fn call."""
        manager = OpManager()
        assert manager._call_with_hooks("op", lambda a, b: a + b, (1, 2), {}) == 3

    @pytest.mark.supplementary
    def test_hooks_with_dump_enabled(self, tmp_path):
        """With dumping on, exec_order/op_tag land in the JSON metadata."""
        from vllm_fl.dispatch import io_common, io_dumper

        io_common.reset_exec_order()
        io_dumper.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        try:
            manager = OpManager()
            manager._state.initialized = True
            manager._state.init_pid = os.getpid()

            def add(a, b):
                return a + b

            impl = _make_impl("oph", "ref.add", BackendImplKind.REFERENCE, fn=add)
            manager._registry.register_impl(impl)
            result = manager._call_with_hooks(
                "oph", impl.fn, (torch.tensor([1.0]), torch.tensor([2.0])), {}
            )
            assert result.item() == 3.0  # independent oracle
            io_dumper._wait_and_flush()
            op_dir = tmp_path / "rank_0000" / "step_0000" / "oph"
            entry = None
            for line in (
                (op_dir / "input.json").read_text(encoding="utf-8").splitlines()
            ):
                entry = list(json.loads(line).values())[0]
            assert entry is not None
            assert entry["op_name"] == "oph"
            # The manager allocates the order before fn runs, so the hooked
            # op's exec_order is first among all intercepted calls.
            hooked_order = io_common.get_exec_order()
            assert 1 <= entry["exec_order"] < hooked_order
            assert entry["op_tag"].startswith("[op=")
            assert entry["op_tag"].endswith(",1]")
        finally:
            io_dumper.disable_io_dump()

    @pytest.mark.supplementary
    def test_hooks_cleanup_pairing_on_fn_raise(self, tmp_path):
        """A raising fn triggers dump_cleanup so no stale pairing remains."""
        from vllm_fl.dispatch import io_dumper

        io_dumper.enable_io_dump(dump_dir=str(tmp_path), with_metas=True)
        try:

            def boom(*args, **kwargs):
                raise ValueError("hooked op failed")

            manager = OpManager()
            with pytest.raises(ValueError, match="hooked op failed"):
                manager._call_with_hooks("opraise", boom, (torch.zeros(2),), {})
            stack = getattr(io_dumper._dump_pairing, "stack", {})
            assert all(not entries for entries in stack.values())
        finally:
            io_dumper.disable_io_dump()


class TestSelectionPolicy:
    """SelectionPolicy normalization, orders, vendor gates, fingerprint."""

    @pytest.mark.supplementary
    def test_from_dict_normalizes_and_sorts(self):
        """prefer lowercases; per_op_order becomes a sorted tuple-of-tuples."""
        policy = SelectionPolicy.from_dict(
            prefer="VENDOR",
            per_op_order={"op_b": ["x", "y"], "op_a": ["z"]},
            deny_vendors=["nvidia"],
            allow_vendors=["cuda"],
        )
        assert policy.prefer == "vendor"
        assert policy.per_op_order == (
            ("op_a", ("z",)),
            ("op_b", ("x", "y")),
        )
        assert policy.deny_vendors == frozenset({"nvidia"})
        assert policy.allow_vendors == frozenset({"cuda"})

    @pytest.mark.supplementary
    def test_from_dict_empty_collections(self):
        """Empty/None deny/allow collapse to empty frozenset / None."""
        policy = SelectionPolicy.from_dict(deny_vendors=[], allow_vendors=[])
        assert policy.deny_vendors == frozenset()
        assert policy.allow_vendors is None
        assert policy.per_op_order == ()

    @pytest.mark.supplementary
    def test_invalid_prefer_raises(self):
        """__post_init__ rejects anything outside VALID_PREFER_VALUES."""
        with pytest.raises(ValueError, match="Invalid prefer value: 'bogus'"):
            SelectionPolicy(prefer="bogus")

    @pytest.mark.supplementary
    def test_per_op_order_dict_and_lookup(self):
        """per_op_order_dict copies; get_per_op_order finds or misses."""
        policy = SelectionPolicy.from_dict(
            per_op_order={"opA": ["flagos", "reference"]}
        )
        as_dict = policy.per_op_order_dict
        assert as_dict == {"opA": ["flagos", "reference"]}
        as_dict["opA"].append("mutated")
        # Returned dict is a copy — the policy stays frozen.
        assert policy.get_per_op_order("opA") == ["flagos", "reference"]
        assert policy.get_per_op_order("missing") is None

    @pytest.mark.supplementary
    def test_get_default_order_all_prefers(self):
        """Each prefer maps to its documented kind order."""
        assert SelectionPolicy().get_default_order() == [
            "flagos",
            "vendor",
            "reference",
        ]
        assert SelectionPolicy(prefer=PREFER_REFERENCE).get_default_order() == [
            "reference",
            "flagos",
            "vendor",
        ]
        assert SelectionPolicy(prefer=PREFER_VENDOR).get_default_order() == [
            "vendor",
            "flagos",
            "reference",
        ]

    @pytest.mark.supplementary
    def test_is_vendor_allowed_matrix(self):
        """Deny always wins; allow-list gates when present; default allows."""
        policy = SelectionPolicy.from_dict(
            deny_vendors={"npu"}, allow_vendors={"cuda", "rocm"}
        )
        assert policy.is_vendor_allowed("npu") is False  # denied
        assert policy.is_vendor_allowed("cuda") is True  # in allow set
        assert policy.is_vendor_allowed("xpu") is False  # not in allow set
        # No filters at all → everything allowed.
        assert SelectionPolicy().is_vendor_allowed("anything") is True
        # Deny only (allow None) → non-denied vendors pass.
        deny_only = SelectionPolicy.from_dict(deny_vendors={"npu"})
        assert deny_only.is_vendor_allowed("cuda") is True
        assert deny_only.is_vendor_allowed("npu") is False

    @pytest.mark.supplementary
    def test_fingerprint_sections(self):
        """Fingerprint lists only the non-empty sections, in fixed order."""
        base = SelectionPolicy()
        assert base.fingerprint() == "prefer=flagos;st=0"

        full = SelectionPolicy.from_dict(
            prefer="vendor",
            strict=True,
            allow_vendors={"c"},
            deny_vendors={"a", "b"},
            per_op_order={"o": ["x", "y"]},
        )
        assert full.fingerprint() == ("prefer=vendor;st=1;allow=c;deny=a,b;per=o=x|y")
        # Deny/allow sets are sorted inside their section.
        unsorted = SelectionPolicy.from_dict(deny_vendors={"z", "m"})
        assert "deny=m,z" in unsorted.fingerprint()

    @pytest.mark.supplementary
    def test_hash_and_equality(self):
        """Two identically-built policies hash and compare equal."""
        one = SelectionPolicy.from_dict(
            prefer="vendor",
            strict=True,
            per_op_order={"op": ["a", "b"]},
            deny_vendors={"x"},
            allow_vendors={"y"},
        )
        two = SelectionPolicy.from_dict(
            prefer="vendor",
            strict=True,
            per_op_order={"op": ["a", "b"]},
            deny_vendors={"x"},
            allow_vendors={"y"},
        )
        assert one == two
        assert hash(one) == hash(two)
        assert len({one, two}) == 1
        # ...and a differing policy is distinct.
        assert hash(one) != hash(SelectionPolicy())


class TestPolicyManager:
    """PolicyManager singleton lifecycle and epoch bookkeeping."""

    @pytest.mark.supplementary
    def test_init_reentry_guard(self):
        """Re-calling __init__ on the singleton keeps existing state."""
        manager = PolicyManager.get_instance()
        epoch_before = manager.get_policy_epoch()
        manager.__init__()  # re-entry guard: no reset
        assert manager.get_policy_epoch() == epoch_before

    @pytest.mark.supplementary
    def test_singleton_identity(self):
        """get_instance always returns the same object."""
        assert PolicyManager.get_instance() is PolicyManager.get_instance()

    @pytest.mark.supplementary
    def test_epoch_bump_increments(self):
        """bump_policy_epoch returns the new value each time."""
        manager = PolicyManager.get_instance()
        before = manager.get_policy_epoch()
        first = manager.bump_policy_epoch()
        second = manager.bump_policy_epoch()
        assert first == before + 1
        assert second == before + 2

    @pytest.mark.supplementary
    def test_global_policy_set_and_reset(self):
        """set_global_policy returns the old policy; reset falls back to env."""
        manager = PolicyManager.get_instance()
        env_derived = manager.get_policy()
        marker = SelectionPolicy.from_dict(strict=True)
        old = manager.set_global_policy(marker)
        assert get_policy() is marker
        assert old == env_derived  # env-derived default returned
        manager.reset_global_policy()
        after = manager.get_policy()
        assert after is not marker
        assert after == env_derived

    @pytest.mark.supplementary
    def test_policy_context_precedence_and_epoch(self):
        """Context policies beat the global one; epochs bump on enter/exit."""
        manager = PolicyManager.get_instance()
        global_policy = SelectionPolicy.from_dict(strict=True)
        manager.set_global_policy(global_policy)
        try:
            assert get_policy().strict is True
            before = manager.get_policy_epoch()
            with manager.create_policy_context(
                SelectionPolicy.from_dict(prefer="reference")
            ):
                assert get_policy().prefer == "reference"
                mid = manager.get_policy_epoch()
                assert mid == before + 1
            assert get_policy() is global_policy
            assert manager.get_policy_epoch() == mid + 1
        finally:
            manager.reset_global_policy()


class TestPolicyParsers:
    """_parse_csv_set, _parse_per_op and _parse_op_config oracles."""

    @pytest.mark.supplementary
    def test_parse_csv_set(self):
        """Empty string → empty set; items stripped and de-blanked."""
        assert PolicyManager._parse_csv_set("") == set()
        assert PolicyManager._parse_csv_set(" a , b ,, c ") == {"a", "b", "c"}

    @pytest.mark.supplementary
    def test_parse_per_op(self):
        """Malformed segments (no '=', blank op, empty order) are skipped."""
        parsed = PolicyManager._parse_per_op("op1=a|b|c;op2=x|y;junk;=empty;op3=")
        assert parsed == {"op1": ["a", "b", "c"], "op2": ["x", "y"]}
        assert PolicyManager._parse_per_op("") == {}

    @pytest.mark.supplementary
    def test_parse_op_config_valid(self):
        """Each op maps to a single-element order of the lowercased backend."""
        parsed = PolicyManager._parse_op_config(
            {"rms_norm": "REFERENCE", "silu": "vendor"}
        )
        assert parsed == {"rms_norm": ["reference"], "silu": ["vendor"]}

    @pytest.mark.supplementary
    def test_parse_op_config_invalid_backend(self):
        """An unknown backend token raises with the offending values."""
        with pytest.raises(ValueError, match="Unsupported backend 'nope' for op 'op'."):
            PolicyManager._parse_op_config({"op": "nope"})


class TestPolicyFromConfig:
    """_policy_from_config parsing branches via tmp_path YAML files."""

    @pytest.mark.supplementary
    def test_missing_file_raises(self):
        """A nonexistent config path raises FileNotFoundError verbatim."""
        with pytest.raises(
            FileNotFoundError,
            match="Config file '.*missing.yaml' not found.",
        ):
            PolicyManager.get_instance()._policy_from_config("definitely/missing.yaml")

    @pytest.mark.supplementary
    def test_invalid_yaml_raises_valueerror(self, tmp_path):
        """Unparseable YAML surfaces as ValueError with the path."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("prefer: [unclosed\n  - broken", encoding="utf-8")
        with pytest.raises(ValueError, match="Failed to load config file"):
            policy_from_config(str(bad))

    @pytest.mark.supplementary
    def test_invalid_prefer_falls_back_to_default(self, tmp_path):
        """An unknown prefer string is replaced with 'flagos'."""
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("prefer: quantum\n", encoding="utf-8")
        policy = policy_from_config(str(cfg))
        assert policy.prefer == PREFER_DEFAULT

    @pytest.mark.supplementary
    def test_deny_allow_as_list_and_csv(self, tmp_path):
        """deny/allow accept both YAML lists and csv strings."""
        as_list = tmp_path / "list.yaml"
        as_list.write_text(
            "deny_vendors: [ascend, npu]\nallow_vendors: [cuda, rocm]\n",
            encoding="utf-8",
        )
        policy = policy_from_config(str(as_list))
        assert policy.deny_vendors == frozenset({"ascend", "npu"})
        assert policy.allow_vendors == frozenset({"cuda", "rocm"})

        as_csv = tmp_path / "csv.yaml"
        as_csv.write_text(
            "deny_vendors: ascend, npu\nallow_vendors: cuda, rocm\n",
            encoding="utf-8",
        )
        policy = policy_from_config(str(as_csv))
        assert policy.deny_vendors == frozenset({"ascend", "npu"})
        assert policy.allow_vendors == frozenset({"cuda", "rocm"})

    @pytest.mark.supplementary
    def test_op_backends_list_and_string(self, tmp_path):
        """op_backends accept YAML lists and 'vendor:cuda|flagos' strings."""
        as_list = tmp_path / "list.yaml"
        as_list.write_text(
            "op_backends:\n"
            "  rms_norm:\n"
            "    - vendor\n"
            "    - flagos\n"
            "  silu_and_mul: vendor:cuda|flagos\n",
            encoding="utf-8",
        )
        policy = policy_from_config(str(as_list))
        assert policy.per_op_order_dict == {
            "rms_norm": ["vendor", "flagos"],
            "silu_and_mul": ["vendor:cuda", "flagos"],
        }

    @pytest.mark.supplementary
    def test_empty_and_partial_config(self, tmp_path):
        """An empty file yields defaults; missing sections stay default."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")
        policy = policy_from_config(str(empty))
        assert policy == SelectionPolicy()


class TestPolicyFromEnv:
    """_policy_from_env priority: config > env > platform > default."""

    @pytest.mark.supplementary
    def test_config_file_completely_overrides(self, tmp_path, monkeypatch):
        """VLLM_FL_CONFIG wins over every env var set alongside it."""
        cfg = tmp_path / "full.yaml"
        cfg.write_text("prefer: reference\nstrict: true\n", encoding="utf-8")
        monkeypatch.setenv("VLLM_FL_CONFIG", str(cfg))
        monkeypatch.setenv("VLLM_FL_PREFER", "vendor")
        monkeypatch.setenv("VLLM_FL_STRICT", "0")
        policy = policy_from_env()
        assert policy.prefer == "reference"
        assert policy.strict is True

    @pytest.mark.supplementary
    def test_platform_config_used_as_base(self, monkeypatch):
        """The platform config feeds prefer/allow/per-op defaults.

        On Linux with real vLLM, ``current_platform.vendor_name`` for the
        CPU platform is ``None`` (not an ImportError), so the
        ``VLLM_FL_PLATFORM`` env override is only consulted where vLLM is
        absent (Windows). To stay platform-independent, the platform
        config is injected through ``get_config_path`` — the same seam
        ``_policy_from_env`` calls via ``vllm_fl.dispatch.config``.
        """
        import vllm_fl.dispatch.config as dispatch_config
        from vllm_fl.dispatch.config.utils import _CONFIG_DIR

        nvidia_yaml = _CONFIG_DIR / "nvidia.yaml"
        monkeypatch.setattr(dispatch_config, "get_config_path", lambda: nvidia_yaml)
        policy = policy_from_env()
        assert policy.prefer == PREFER_DEFAULT
        assert policy.allow_vendors == frozenset({"cuda"})
        per_op = policy.per_op_order_dict
        assert "attention_backend" in per_op
        assert "rms_norm" in per_op

    @pytest.mark.supplementary
    def test_invalid_strict_warns_and_defaults(self, monkeypatch, caplog):
        """A non-0/1 VLLM_FL_STRICT warns and falls back to False."""
        monkeypatch.setenv("VLLM_FL_STRICT", "yes")
        with caplog.at_level(logging.WARNING, logger="vllm_fl.dispatch.policy"):
            policy = policy_from_env()
        assert policy.strict is False
        assert any(
            "Invalid VLLM_FL_STRICT value 'yes'" in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.supplementary
    def test_env_vars_override_platform(self, monkeypatch):
        """VLLM_FL_PREFER / DENY / ALLOW / PER_OP beat the platform file."""
        monkeypatch.setenv("VLLM_FL_PREFER", "reference")
        monkeypatch.setenv("VLLM_FL_DENY_VENDORS", "npu, ascend")
        monkeypatch.setenv("VLLM_FL_ALLOW_VENDORS", "cuda")
        monkeypatch.setenv("VLLM_FL_PER_OP", "my_op=reference|flagos")
        policy = policy_from_env()
        assert policy.prefer == "reference"
        assert policy.deny_vendors == frozenset({"npu", "ascend"})
        assert policy.allow_vendors == frozenset({"cuda"})
        assert policy.per_op_order_dict == {"my_op": ["reference", "flagos"]}

    @pytest.mark.supplementary
    def test_defaults_without_any_env(self):
        """With a clean environment the built-in defaults come back."""
        policy = policy_from_env()
        assert policy == SelectionPolicy()


class TestPolicyContextManagers:
    """Module-level convenience context managers."""

    @pytest.mark.supplementary
    def test_with_strict_mode_toggles(self):
        """with_strict_mode keeps the current prefer and adds strict."""
        with with_strict_mode():
            active = get_policy()
            assert active.strict is True
            assert active.prefer == PREFER_DEFAULT
        assert get_policy().strict is False

    @pytest.mark.supplementary
    def test_with_preference_switches(self):
        """with_preference swaps only the prefer field."""
        with with_preference(PREFER_REFERENCE):
            assert get_policy().prefer == "reference"
        assert get_policy().prefer == PREFER_DEFAULT

    @pytest.mark.supplementary
    def test_with_allowed_vendors_replaces_allowlist(self):
        """with_allowed_vendors replaces the allow set wholesale."""
        with with_allowed_vendors("cuda", "rocm"):
            assert get_policy().allow_vendors == frozenset({"cuda", "rocm"})
        assert get_policy().allow_vendors is None

    @pytest.mark.supplementary
    def test_with_denied_vendors_unions(self):
        """with_denied_vendors adds to the existing deny set (union)."""
        with with_denied_vendors("npu"), with_denied_vendors("ascend"):
            assert get_policy().deny_vendors == frozenset({"npu", "ascend"})
        assert get_policy().deny_vendors == frozenset()

    @pytest.mark.supplementary
    def test_contexts_nest_and_restore(self):
        """Nested contexts unwind LIFO back to the base policy."""
        base = get_policy()
        with with_preference(PREFER_VENDOR):
            with with_strict_mode():
                inner = get_policy()
                assert inner.prefer == "vendor"
                assert inner.strict is True
            mid = get_policy()
            assert mid.prefer == "vendor"
            assert mid.strict is False
        assert get_policy() is base
