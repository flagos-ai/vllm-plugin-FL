# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary tests for the remaining uncovered lines of the dispatch core.

The combined core+supplementary coverage report shows exactly these gaps:

``vllm_fl/dispatch/manager.py`` (16 lines):

* lines 123/125 — the ``except AttributeError`` arm of the
  ``os.register_at_fork`` guard in ``OpManager.__init__`` (the arm taken on
  platforms without ``os.register_at_fork``, and on any platform where the
  call itself raises ``AttributeError``);
* line 247 — the ``if _DISPATCH_DEBUG: self._print_registered_operators()``
  branch of ``ensure_initialized``;
* lines 251-270 — the whole body of ``_print_registered_operators`` (header
  banner, per-operator blocks sorted by name, per-implementation lines with
  availability checkmarks, vendor info and the footer banner).

``vllm_fl/dispatch/discovery.py`` (6 lines):

* lines 39-46 — the ``ImportError`` fallback chain of ``_get_entry_points``:
  stdlib ``importlib.metadata`` missing ``entry_points`` falls back to the
  ``importlib_metadata`` backport; if that is missing too, a debug log is
  emitted and an empty list returned.

Everything else in both modules is already covered by the existing core and
supplementary suites (see ``test_manager_policy_edges.py`` and
``test_discovery_builtin_ops.py``); the tests below only add what is missing.
"""

import logging
import os
import sys
import types

import pytest

from vllm_fl.dispatch import discovery, manager as mgr_mod
from vllm_fl.dispatch.manager import OpManager
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl



def _make_impl(op_name, impl_id, kind, *, vendor=None, priority=None, available=True):
    """Build an OpImpl whose availability probe reports *available*."""
    if priority is None:
        priority = {
            BackendImplKind.DEFAULT: BackendPriority.DEFAULT,
            BackendImplKind.VENDOR: BackendPriority.VENDOR,
            BackendImplKind.REFERENCE: BackendPriority.REFERENCE,
        }[kind]

    def fn(*args, **kwargs):  # noqa: ANN002, ANN003
        return f"{impl_id}-result"

    fn._is_available = (lambda: True) if available else (lambda: False)
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
    """Give each test a pristine policy/manager state and vllm.platforms stub.

    ``clean_fl_env`` (dispatch conftest) drops every ``VLLM_FL_*`` env
    var for the test, so ``VLLM_FL_DISPATCH_DEBUG`` never leaks between tests
    and policy defaults apply.  A minimal ``vllm.platforms`` stub is kept in
    place where one is absent so vendor detection inside ``builtin_ops``
    succeeds on machines without the real vLLM (Windows); on WSL with real
    vLLM the real module is snapshotted and restored.
    """
    saved_platform = sys.modules.get("vllm.platforms")
    if saved_platform is None:
        platforms = types.ModuleType("vllm.platforms")
        platforms.current_platform = types.SimpleNamespace(vendor_name="nvidia")
        monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    from vllm_fl.dispatch.policy import PolicyManager

    PolicyManager.get_instance().reset_global_policy()
    yield
    if saved_platform is None:
        monkeypatch.delitem(sys.modules, "vllm.platforms", raising=False)
    else:
        monkeypatch.setitem(sys.modules, "vllm.platforms", saved_platform)
    PolicyManager.get_instance().reset_global_policy()


class TestForkRegistrationGuard:
    """OpManager.__init__ at_fork registration (manager lines 121-125)."""

    def test_raising_register_at_fork_is_swallowed(self, monkeypatch):
        """An AttributeError from the registration call is silently ignored.

        ``os.register_at_fork`` exists on Linux but not Windows; either way a
        call raising ``AttributeError`` must take the except arm (lines
        123/125) and still produce a fully constructed manager.
        """

        def boom(**kwargs):
            raise AttributeError("register_at_fork unavailable")

        monkeypatch.setattr(os, "register_at_fork", boom, raising=False)
        manager = OpManager()  # must not raise
        assert manager._state.initialized is False
        assert manager._dispatch_cache == {}

    def test_successful_registration_registers_callback(self, monkeypatch):
        """A working os.register_at_fork receives the _reset_after_fork hook.

        On Windows the real function is absent, so a stand-in is installed;
        on Linux it replaces the real one for the duration of the test.  The
        manager must hand its bound ``_reset_after_fork`` method over.
        """
        registered = []

        def fake_register(**kwargs):
            registered.append(kwargs)

        monkeypatch.setattr(os, "register_at_fork", fake_register, raising=False)
        manager = OpManager()
        assert registered == [{"after_in_child": manager._reset_after_fork}]


class TestDispatchDebugPrint:
    """_DISPATCH_DEBUG wiring and _print_registered_operators (247, 251-270)."""

    def test_ensure_initialized_prints_operators_when_debug_enabled(
        self, monkeypatch, caplog
    ):
        """With _DISPATCH_DEBUG on, ensure_initialized prints the op table.

        The module-level flag is read at call time (line 246), so patching
        the module attribute flips the branch without re-importing.
        """
        monkeypatch.setattr(mgr_mod, "_DISPATCH_DEBUG", True)
        manager = OpManager()
        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.manager"):
            manager.ensure_initialized()
        messages = [record.getMessage() for record in caplog.records]
        assert any("VLLM-FL Dispatch: Registered Operators" in m for m in messages)
        assert any("[Operator: rms_norm]" in m for m in messages)
        # The builtin registry always registers the reference torch kernels.
        assert any("reference.torch" in m for m in messages)

    def test_ensure_initialized_skips_print_when_debug_disabled(self, caplog):
        """Without the debug flag the op table is never printed."""
        assert mgr_mod._DISPATCH_DEBUG is False  # clean_fl_env removed the env var
        manager = OpManager()
        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.manager"):
            manager.ensure_initialized()
        assert not any(
            "VLLM-FL Dispatch: Registered Operators" in record.getMessage()
            for record in caplog.records
        )

    def test_print_registered_operators_lines_and_ordering(self, caplog):
        """The op table lists ops sorted by name, impls by priority desc.

        Independent oracle: two ops with names that sort in the opposite
        order to their registration, and two impls whose impl_id sort order
        opposes their priority order — the output must follow name/priority
        order, not registration or lexical order.
        """
        manager = OpManager()
        manager._registry.register_impl(
            _make_impl("zeta_op", "lowprio.impl", BackendImplKind.REFERENCE, priority=1)
        )
        manager._registry.register_impl(
            _make_impl(
                "zeta_op", "highprio.impl", BackendImplKind.REFERENCE, priority=99
            )
        )
        manager._registry.register_impl(
            _make_impl("alpha_op", "alpha.impl", BackendImplKind.REFERENCE)
        )
        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.manager"):
            manager._print_registered_operators()
        messages = [record.getMessage() for record in caplog.records]
        # Header/footer banners and per-op blocks.
        assert any(m.startswith("\n[Operator: alpha_op]") for m in messages)
        assert any(m.startswith("\n[Operator: zeta_op]") for m in messages)
        assert any("=" * 80 in m for m in messages)
        # alpha_op sorts before zeta_op despite registering last.
        alpha_idx = next(i for i, m in enumerate(messages) if "alpha_op" in m)
        zeta_idx = next(i for i, m in enumerate(messages) if "zeta_op" in m)
        assert alpha_idx < zeta_idx
        # Within zeta_op the priority-99 impl is printed before priority-1.
        high_idx = next(i for i, m in enumerate(messages) if "highprio.impl" in m)
        low_idx = next(i for i, m in enumerate(messages) if "lowprio.impl" in m)
        assert high_idx < low_idx
        assert "priority=99" in messages[high_idx]
        assert "priority=1" in messages[low_idx]

    def test_print_registered_operators_availability_and_vendor(self, caplog):
        """Unavailable impls print a cross; vendor impls add vendor=<name>."""
        manager = OpManager()
        manager._registry.register_impl(
            _make_impl("opq", "ref.off", BackendImplKind.REFERENCE, available=False)
        )
        manager._registry.register_impl(
            _make_impl("opq", "vendor.cuda", BackendImplKind.VENDOR, vendor="cuda")
        )
        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.manager"):
            manager._print_registered_operators()
        joined = "\n".join(record.getMessage() for record in caplog.records)
        assert "✗ ref.off" in joined
        assert "✓ vendor.cuda" in joined
        assert "vendor=cuda" in joined
        # A reference impl without vendor prints no vendor suffix.
        ref_line = next(
            line for line in joined.splitlines() if "ref.off" in line and "✗" in line
        )
        assert "vendor=" not in ref_line

    def test_print_registered_operators_empty_registry(self, caplog):
        """An empty registry still prints the banners (no per-op blocks)."""
        manager = OpManager()
        with caplog.at_level(logging.INFO, logger="vllm_fl.dispatch.manager"):
            manager._print_registered_operators()
        messages = [record.getMessage() for record in caplog.records]
        assert any("VLLM-FL Dispatch: Registered Operators" in m for m in messages)
        assert not any("[Operator:" in m for m in messages)


class TestGetEntryPointsImportFallback:
    """_get_entry_points ImportError fallback chain (discovery lines 39-46)."""

    @staticmethod
    def _stdlib_without_entry_points():
        """A stand-in importlib.metadata module lacking entry_points."""
        import contextlib
        import importlib.metadata as real

        stub = types.ModuleType("importlib.metadata")
        for name in dir(real):
            if name != "entry_points":
                with contextlib.suppress(AttributeError, TypeError):
                    setattr(stub, name, getattr(real, name))
        return stub

    def test_stdlib_failure_falls_back_to_importlib_metadata(self, monkeypatch):
        """A broken stdlib entry_points defers to the backport package.

        Independent oracle: the backport returns a dict-style container
        (Python 3.9 convention) holding one plugin entry, which must come
        back from ``_get_entry_points`` verbatim.
        """
        expected = ["EP_FROM_BACKPORT"]
        backport = types.ModuleType("importlib_metadata")
        backport.entry_points = lambda: {discovery.PLUGIN_GROUP: expected}
        monkeypatch.setitem(
            sys.modules, "importlib.metadata", self._stdlib_without_entry_points()
        )
        monkeypatch.setitem(sys.modules, "importlib_metadata", backport)
        assert discovery._get_entry_points() == expected

    def test_both_imports_missing_returns_empty(self, monkeypatch, caplog):
        """With both providers broken, a debug note is logged and [] returned."""
        broken_backport = types.ModuleType("importlib_metadata")
        monkeypatch.setitem(
            sys.modules, "importlib.metadata", self._stdlib_without_entry_points()
        )
        monkeypatch.setitem(sys.modules, "importlib_metadata", broken_backport)
        with caplog.at_level(logging.DEBUG, logger="vllm_fl.dispatch"):
            assert discovery._get_entry_points() == []
        assert any(
            "importlib.metadata not available" in record.getMessage()
            for record in caplog.records
        )

    def test_fallback_feeds_discover_from_entry_points(self, monkeypatch):
        """The fallback result flows through discover_from_entry_points.

        With both import providers broken, entry-point discovery loads zero
        plugins and records nothing — proving the empty-list short circuit
        (discovery line 127) is reached from the ImportError chain.
        """
        broken_backport = types.ModuleType("importlib_metadata")
        monkeypatch.setitem(
            sys.modules, "importlib.metadata", self._stdlib_without_entry_points()
        )
        monkeypatch.setitem(sys.modules, "importlib_metadata", broken_backport)
        discovery.clear_discovered_plugins()
        try:
            assert discovery.discover_from_entry_points(object()) == 0
            assert discovery.get_discovered_plugins() == []
        finally:
            discovery.clear_discovered_plugins()
