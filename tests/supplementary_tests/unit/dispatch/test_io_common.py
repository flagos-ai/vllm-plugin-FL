# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary CPU tests for ``vllm_fl.dispatch.io_common`` gap areas.

Existing dispatch tests (tests/unit_tests/dispatch/test_io_dumper.py) cover
io_common only through the dumper's integration paths.  This module targets
the directly exported helpers that remain weakly covered: ``ModeManager``
LIFO lifecycle, re-entrancy guards, exec-order / per-step counters, step
callback machinery, rank resolution, filter parsers, module-context stacks,
dispatch-key probing, formatting helpers, torch-func tag management, and
YAML config parsing corners.
"""

import logging
import threading

import pytest
import torch

from vllm_fl.dispatch import io_common


class _FakeMode:
    """Minimal context manager recording enter/exit counts (no torch stack)."""

    def __init__(self, name):
        self.name = name
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        self.exited += 1
        return False


def _drain_module_context():
    """Pop every entry from the thread-local module-context stack."""
    stack = getattr(io_common._module_context, "stack", None)
    while stack:
        stack.pop()


@pytest.fixture(autouse=True)
def _reset_io_common_state():
    """Reset every io_common global (and step callbacks) before and after each test."""
    saved_stats = list(io_common._TENSOR_STAT_REGISTRY)
    io_common.reset_step()
    io_common.reset_exec_order()
    io_common.reset_rank()
    io_common.dispatch_mode_mgr.exit_all()
    io_common.func_mode_mgr.exit_all()
    io_common._module_path_map.clear()
    io_common._dispatch_table_cache.clear()
    saved_callbacks = list(io_common._step_callbacks)
    io_common._step_callbacks.clear()
    io_common.set_io_active(False)
    _drain_module_context()
    yield
    io_common.reset_step()
    io_common.reset_exec_order()
    io_common.reset_rank()
    io_common.dispatch_mode_mgr.exit_all()
    io_common.func_mode_mgr.exit_all()
    io_common._module_path_map.clear()
    io_common._dispatch_table_cache.clear()
    io_common._step_callbacks.clear()
    io_common._step_callbacks.extend(saved_callbacks)
    io_common._TENSOR_STAT_REGISTRY[:] = saved_stats
    io_common.set_io_active(False)
    _drain_module_context()


class TestModeManager:
    """ModeManager LIFO entry/exit bookkeeping without touching torch's mode stack."""

    @pytest.mark.supplementary
    def test_enter_is_idempotent_per_name(self):
        """Entering the same name twice enters the underlying mode only once."""
        mgr = io_common.ModeManager()
        mode = _FakeMode("dispatch")
        mgr.enter("dumper", mode)
        mgr.enter("dumper", mode)
        assert mode.entered == 1
        assert mgr.is_entered("dumper")

    @pytest.mark.supplementary
    def test_request_exit_of_unknown_name_is_noop(self):
        """Requesting exit for a name never entered leaves the stack untouched."""
        mgr = io_common.ModeManager()
        mode = _FakeMode("m")
        mgr.enter("a", mode)
        mgr.request_exit("never-entered")
        assert mode.exited == 0
        assert mgr.is_entered("a")

    @pytest.mark.supplementary
    def test_non_top_exit_deferred_until_drain(self):
        """Non-top modes stay entered until every mode above them also exits."""
        mgr = io_common.ModeManager()
        low, mid, top = _FakeMode("low"), _FakeMode("mid"), _FakeMode("top")
        mgr.enter("low", low)
        mgr.enter("mid", mid)
        mgr.enter("top", top)

        # Neither "low" nor "mid" is on top: both exits stay pending.
        mgr.request_exit("low")
        assert (low.exited, mid.exited, top.exited) == (0, 0, 0)
        assert mgr.is_entered("low") and mgr.is_entered("mid")

        mgr.request_exit("mid")
        assert (low.exited, mid.exited, top.exited) == (0, 0, 0)

        # Exiting the top drains every pending exit below it in LIFO order.
        mgr.request_exit("top")
        assert (low.exited, mid.exited, top.exited) == (1, 1, 1)
        assert not mgr.is_entered("low")

    @pytest.mark.supplementary
    def test_exit_all_clears_pending_and_exits_lifo(self):
        """exit_all exits every mode in reverse order and forgets pending exits."""
        mgr = io_common.ModeManager()
        first, second = _FakeMode("first"), _FakeMode("second")
        mgr.enter("first", first)
        mgr.enter("second", second)
        mgr.request_exit("first")  # pending, not top
        mgr.exit_all()
        assert (first.exited, second.exited) == (1, 1)

        # Pending set was cleared: re-entering "first" then exiting "second"
        # must not drain the fresh "first".
        fresh = _FakeMode("first")
        mgr.enter("first", fresh)
        mgr.request_exit("second")
        assert fresh.exited == 0

    @pytest.mark.supplementary
    def test_global_managers_are_independent(self):
        """The module-level dispatch/func mode managers hold separate stacks."""
        io_common.dispatch_mode_mgr.enter("probe", _FakeMode("d"))
        assert io_common.dispatch_mode_mgr.is_entered("probe")
        assert not io_common.func_mode_mgr.is_entered("probe")
        io_common.dispatch_mode_mgr.exit_all()


class TestIoActiveFlag:
    """set_io_active / is_io_active toggle and managed_inference_mode behaviour."""

    @pytest.mark.supplementary
    def test_flag_roundtrip(self):
        """The flag is a plain toggle reflected by is_io_active."""
        assert io_common.is_io_active() is False
        io_common.set_io_active(True)
        assert io_common.is_io_active() is True
        io_common.set_io_active(False)
        assert io_common.is_io_active() is False

    @pytest.mark.supplementary
    def test_managed_inference_mode_downgrades_when_io_active(self):
        """With IO active the decorator runs under no_grad, not inference_mode."""
        io_common.set_io_active(True)

        @io_common.managed_inference_mode()
        def probe():
            return torch.is_grad_enabled(), torch.is_inference_mode_enabled()

        with torch.enable_grad():
            grad_enabled, inference_enabled = probe()
        assert grad_enabled is False
        assert inference_enabled is False

    @pytest.mark.supplementary
    def test_managed_inference_mode_uses_inference_mode_when_inactive(self):
        """Without IO the decorator behaves exactly like torch.inference_mode."""
        io_common.set_io_active(False)

        @io_common.managed_inference_mode()
        def probe():
            return torch.is_grad_enabled(), torch.is_inference_mode_enabled()

        with torch.enable_grad():
            grad_enabled, inference_enabled = probe()
        assert grad_enabled is False
        assert inference_enabled is True


class TestRankResolution:
    """get_rank resolution order: cached value, RANK, LOCAL_RANK, 0."""

    @pytest.mark.supplementary
    def test_rank_env_var_is_cached(self, monkeypatch):
        """The first successful env lookup is cached for later calls."""
        monkeypatch.setenv("RANK", "3")
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert io_common.get_rank() == 3
        monkeypatch.setenv("RANK", "5")
        assert io_common.get_rank() == 3  # cached

    @pytest.mark.supplementary
    def test_invalid_rank_falls_through_to_local_rank(self, monkeypatch):
        """A non-integer RANK is skipped and LOCAL_RANK is consulted next."""
        monkeypatch.setenv("RANK", "not-a-number")
        monkeypatch.setenv("LOCAL_RANK", "7")
        assert io_common.get_rank() == 7

    @pytest.mark.supplementary
    def test_zero_fallback_is_not_cached(self, monkeypatch):
        """The 0 fallback is recomputed so a later env var is still honoured."""
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert io_common.get_rank() == 0
        monkeypatch.setenv("RANK", "2")
        assert io_common.get_rank() == 2

    @pytest.mark.supplementary
    def test_parse_rank_filter_variants(self):
        """parse_rank_filter handles all/empty/sets and skips bad tokens."""
        assert io_common.parse_rank_filter("all") is None
        assert io_common.parse_rank_filter("") is None
        assert io_common.parse_rank_filter(" ALL ") is None
        assert io_common.parse_rank_filter("0, 2,4") == {0, 2, 4}
        assert io_common.parse_rank_filter("x,1,, 3") == {1, 3}
        assert io_common.parse_rank_filter("only-junk") is None


class TestGuardsAndCounters:
    """make_guard isolation and the exec-order / module / op counters."""

    @pytest.mark.supplementary
    def test_guards_are_independent_per_caller(self):
        """Each make_guard() pair is independent from other guards."""
        guard_a, set_a = io_common.make_guard()
        guard_b, set_b = io_common.make_guard()
        set_a(True)
        assert guard_a() is True
        assert guard_b() is False
        set_a(False)
        set_b(True)
        assert guard_a() is False
        assert guard_b() is True

    @pytest.mark.supplementary
    def test_guard_is_thread_local(self):
        """A guard set in this thread is invisible in another thread."""
        guard, setter = io_common.make_guard()
        setter(True)
        seen = {}

        def other_thread():
            seen["value"] = guard()

        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        assert seen["value"] is False
        assert guard() is True

    @pytest.mark.supplementary
    def test_exec_order_monotonic_then_reset(self):
        """next_exec_order increments monotonically; reset returns it to zero."""
        assert io_common.next_exec_order() == 1
        assert io_common.next_exec_order() == 2
        io_common.reset_exec_order()
        assert io_common.get_exec_order() == 0
        assert io_common.next_exec_order() == 1

    @pytest.mark.supplementary
    def test_module_and_op_counters_assign_sequential_type_indices(self):
        """First sight of a name gets a new type index; call counts increase."""
        assert io_common.next_module_counter("Linear") == (0, 1)
        assert io_common.next_module_counter("Linear") == (0, 2)
        assert io_common.next_module_counter("Conv") == (1, 1)
        assert io_common.next_op_counter("mm") == (0, 1)
        assert io_common.next_op_counter("mm") == (0, 2)
        assert io_common.next_op_counter("add") == (1, 1)

    @pytest.mark.supplementary
    def test_counters_are_thread_safe_under_contention(self):
        """Concurrent counter bumps produce exactly the summed call count."""
        bumps_per_thread = 200
        threads = [
            threading.Thread(
                target=lambda: [
                    (
                        io_common.next_module_counter("L"),
                        io_common.next_op_counter("op"),
                    )
                    for _ in range(bumps_per_thread)
                ]
            )
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        total = 4 * bumps_per_thread
        assert io_common.next_module_counter("L") == (0, total + 1)
        assert io_common.next_op_counter("op") == (0, total + 1)


class TestStepMachinery:
    """advance_step / reset_step and the step-callback registry."""

    @pytest.mark.supplementary
    def test_advance_step_returns_new_step_and_resets_counters(self):
        """Each advance returns the new step number and restarts exec order."""
        io_common.next_exec_order()
        io_common.next_module_counter("Linear")
        assert io_common.advance_step() == 1
        assert io_common.advance_step() == 2
        assert io_common.get_step() == 2
        assert io_common.get_exec_order() == 0
        assert io_common.next_module_counter("Linear") == (0, 1)

    @pytest.mark.supplementary
    def test_callbacks_receive_completed_step_summary(self):
        """Callbacks get (prev_step, seen_modules, seen_ops) of the finished step."""
        seen_calls = []

        def callback(step, modules, ops):
            seen_calls.append((step, sorted(modules), sorted(ops)))

        io_common.register_step_callback(callback)
        io_common.record_seen("mm", module_name="Linear")
        io_common.advance_step()
        assert seen_calls == [(0, ["Linear"], ["mm"])]

        # Unregister stops further invocations.
        io_common.unregister_step_callback(callback)
        io_common.advance_step()
        assert len(seen_calls) == 1

    @pytest.mark.supplementary
    def test_duplicate_registration_is_ignored(self):
        """Registering the same callback twice fires it once per step."""
        calls = []

        def callback(step, modules, ops):
            calls.append(step)

        io_common.register_step_callback(callback)
        io_common.register_step_callback(callback)
        io_common.advance_step()
        assert calls == [0]

    @pytest.mark.supplementary
    def test_unregister_unknown_callback_is_noop(self):
        """Removing a callback that was never registered does not raise."""
        io_common.unregister_step_callback(lambda step, m, o: None)

    @pytest.mark.supplementary
    def test_failing_callback_is_swallowed_and_logged(self, caplog):
        """A raising callback is logged at WARNING and does not break the step."""

        def bad_callback(step, modules, ops):
            raise RuntimeError("callback exploded")

        io_common.register_step_callback(bad_callback)
        with caplog.at_level(logging.WARNING, logger="vllm_fl.dispatch.io_common"):
            new_step = io_common.advance_step()
        assert new_step == 1
        assert any("Step callback" in rec.getMessage() for rec in caplog.records)

    @pytest.mark.supplementary
    def test_reset_step_zeroes_counter_and_clears_seen(self):
        """reset_step returns to step 0 and drops seen-module/op sets."""
        io_common.record_seen("mm", module_name="Linear")
        io_common.reset_step()
        assert io_common.get_step() == 0
        assert io_common._seen_ops == set()
        assert io_common._seen_modules == set()
        assert io_common.next_module_counter("Linear") == (0, 1)


class TestStepRangeParsing:
    """parse_step_range and parse_step_range_env inclusive dash formats."""

    @pytest.mark.supplementary
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("0-2", (0, 3)),
            ("5", (5, 6)),
            ("  4 - 6 ", (4, 7)),
            ("10-10", (10, 11)),
        ],
    )
    def test_valid_ranges(self, raw, expected):
        """Inclusive "start-end" and bare-integer forms parse to half-open tuples."""
        assert io_common.parse_step_range(raw) == expected

    @pytest.mark.supplementary
    @pytest.mark.parametrize(
        "raw", [None, "", "   ", 5, "junk", "a-b", "-3", "3-", "1-2-3"]
    )
    def test_invalid_ranges_return_none(self, raw):
        """Unset, non-string, or malformed values return None instead of raising."""
        assert io_common.parse_step_range(raw) is None

    @pytest.mark.supplementary
    def test_env_lookup_takes_first_valid(self, monkeypatch):
        """The first env var with a parsable value wins; junk vars are skipped."""
        monkeypatch.setenv("IO_DUMP_STEPS", "2-4")
        monkeypatch.setenv("OTHER_STEPS", "0-1")
        assert io_common.parse_step_range_env("MISSING", "IO_DUMP_STEPS") == (2, 5)
        assert io_common.parse_step_range_env("IO_DUMP_STEPS", "OTHER_STEPS") == (2, 5)
        assert io_common.parse_step_range_env() is None

    @pytest.mark.supplementary
    def test_env_blank_values_are_skipped(self, monkeypatch):
        """Blank env values are skipped rather than treated as invalid input."""
        monkeypatch.setenv("BLANK_STEPS", "   ")
        monkeypatch.setenv("GOOD_STEPS", "7")
        assert io_common.parse_step_range_env("BLANK_STEPS", "GOOD_STEPS") == (7, 8)


class TestModuleContextStack:
    """push/pop module context, path registry, and layer matching."""

    @pytest.mark.supplementary
    def test_register_module_paths_and_lookup(self):
        """register_module_paths maps id(module) to dotted named_modules paths."""
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        model[0].sub = torch.nn.Linear(2, 2)
        io_common.register_module_paths(model)
        assert io_common.get_module_path(model[0]) == "0"
        assert io_common.get_module_path(model[0].sub) == "0.sub"
        assert io_common.get_module_path(model) == ""
        unknown = torch.nn.Linear(2, 2)
        assert io_common.get_module_path(unknown) == ""

    @pytest.mark.supplementary
    def test_context_stack_tracks_counters_and_paths(self):
        """The thread-local stack exposes counters, class names and paths."""
        model = torch.nn.Linear(2, 2)
        model.sub = torch.nn.Linear(2, 2)
        io_common.register_module_paths(model)

        io_common.push_module_context("Linear", model)
        io_common.push_module_context("Linear", model.sub)

        assert io_common.get_current_module() == "Linear"
        assert io_common.get_current_module_counter() == (0, 2)
        assert io_common.get_current_module_path() == "sub"

        io_common.pop_module_context()
        assert io_common.get_current_module_path() == ""
        io_common.pop_module_context()
        assert io_common.get_current_module() is None
        assert io_common.get_current_module_counter() is None
        assert io_common.get_current_module_path() == ""

    @pytest.mark.supplementary
    def test_pop_on_empty_stack_is_noop(self):
        """Popping with an empty context stack does not raise."""
        io_common.pop_module_context()
        assert io_common.get_current_module() is None

    @pytest.mark.supplementary
    def test_module_context_matches_any_frame(self):
        """module_context_matches scans innermost-to-outermost over the stack."""

        class Outer(torch.nn.Module):
            pass

        outer = Outer()
        io_common.push_module_context("Outer", outer)
        io_common.push_module_context("Linear", None)
        assert io_common.module_context_matches({"Outer"})
        assert io_common.module_context_matches({"Linear"})
        assert io_common.module_context_matches({"Conv"}) is False
        assert io_common.module_context_matches(set()) is False

    @pytest.mark.supplementary
    def test_layer_path_matches_prefix_and_glob(self):
        """layer_path_matches honours exact/prefix and fnmatch glob entries."""
        model = torch.nn.Linear(2, 2)
        model.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        io_common.register_module_paths(model)

        io_common.push_module_context("Linear", model.blocks[0])
        assert io_common.get_current_module_path() == "blocks.0"
        assert io_common.layer_path_matches({"blocks.0"})
        assert io_common.layer_path_matches({"blocks"})  # prefix boundary
        assert io_common.layer_path_matches({"blocks.*"})
        assert io_common.layer_path_matches({"*.0"})
        assert io_common.layer_path_matches({"blocks.00"}) is False
        assert io_common.layer_path_matches({"other"}) is False

    @pytest.mark.supplementary
    def test_layer_matching_with_empty_stack(self):
        """With no module context on the stack nothing matches."""
        assert io_common.layer_path_matches({"anything"}) is False
        assert io_common.module_context_matches({"anything"}) is False


class TestStackDerivedContext:
    """get_module_context_from_stack and the *_from_stack matchers."""

    @pytest.mark.supplementary
    def test_stack_walk_finds_enclosing_modules(self):
        """The Python-stack walk finds enclosing nn.Module forward frames."""

        class Inner(torch.nn.Module):
            def forward(self):
                return io_common.get_module_context_from_stack()

        class Outer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()

        model = Outer()
        io_common.register_module_paths(model)
        context = model.inner.forward()
        assert context == [("Inner", "inner")]
        assert io_common.module_context_matches_from_stack({"Inner"}, context)
        assert io_common.module_context_matches_from_stack({"Outer"}, context) is False
        assert io_common.layer_path_matches_from_stack({"inner"}, context)
        assert io_common.layer_path_matches_from_stack({"outer"}, context) is False

    @pytest.mark.supplementary
    def test_stack_matchers_without_context(self):
        """Empty derived context matches nothing for both matcher flavours."""
        assert io_common.layer_path_matches_from_stack({"x"}, []) is False
        assert io_common.module_context_matches_from_stack({"x"}, []) is False


class TestLayerSpecExpansion:
    """expand_layer_specs and parse_layers_env (extra corners beyond io_dumper)."""

    @pytest.mark.supplementary
    def test_parse_layers_env_expands_first_non_empty(self, monkeypatch):
        """The first non-empty env var is expanded; later vars are ignored."""
        monkeypatch.setenv("LAYERS_A", "0-1,lm_head")
        monkeypatch.setenv("LAYERS_B", "5")
        result = io_common.parse_layers_env("MISSING_VAR", "LAYERS_A", "LAYERS_B")
        assert result == {"model.layers.0", "model.layers.1", "lm_head"}
        assert io_common.parse_layers_env() == set()

    @pytest.mark.supplementary
    def test_parse_layers_env_skips_blank_values(self, monkeypatch):
        """A whitespace-only env value is skipped, not treated as empty input."""
        monkeypatch.setenv("LAYERS_BLANK", "  ")
        monkeypatch.setenv("LAYERS_SET", "3")
        assert io_common.parse_layers_env("LAYERS_BLANK", "LAYERS_SET") == {
            "model.layers.3"
        }


class TestTorchFuncFiltering:
    """should_inspect_torch_func and its name helpers."""

    @pytest.mark.supplementary
    def test_disabled_never_inspects(self):
        """A disabled torch-func capture rejects every function."""
        assert (
            io_common.should_inspect_torch_func("mm", False, set(), True, set())
            is False
        )

    @pytest.mark.supplementary
    def test_explicit_filter_restricts_to_members(self):
        """A non-empty filter set admits only listed names."""
        args = (True, {"softmax"}, True, set())
        assert io_common.should_inspect_torch_func("softmax", *args) is True
        assert io_common.should_inspect_torch_func("matmul", *args) is False

    @pytest.mark.supplementary
    def test_match_all_skips_dunder_and_trivial(self):
        """In match-all mode dunder internals and SKIP_TORCH_FUNCS are dropped."""
        assert (
            io_common.should_inspect_torch_func("mm", True, set(), True, set()) is True
        )
        assert (
            io_common.should_inspect_torch_func("_private", True, set(), True, set())
            is False
        )
        for trivial in ("size", "dim", "is_contiguous", "device", "dtype"):
            assert (
                io_common.should_inspect_torch_func(trivial, True, set(), True, set())
                is False
            )

    @pytest.mark.supplementary
    def test_op_filter_narrows_match_all(self):
        """When an op filter is set, match-all only admits ops in that filter."""
        assert (
            io_common.should_inspect_torch_func("mm", True, set(), False, {"mm"})
            is True
        )
        assert (
            io_common.should_inspect_torch_func("add", True, set(), False, {"mm"})
            is False
        )

    @pytest.mark.supplementary
    def test_get_torch_func_name(self):
        """Function objects give __name__; other callables degrade to str()."""
        assert io_common.get_torch_func_name(torch.add) == "add"
        assert io_common.get_torch_func_name(42) == "42"


class TestDispatchOpHelpers:
    """Dispatch op naming, namespace, filtering and dispatch-key probing."""

    @pytest.mark.supplementary
    def test_get_dispatch_op_name_strips_namespace(self):
        """OpOverload names like "aten::mm" reduce to their last segment."""

        class FakeOp:
            def name(self):
                return "aten::_softmax"

        class NamespacedOp:
            def name(self):
                return "mylib::weird_op"

            namespace = "mylib"

        assert io_common.get_dispatch_op_name(FakeOp()) == "_softmax"
        assert io_common.get_dispatch_op_name(NamespacedOp()) == "weird_op"
        assert io_common.get_dispatch_op_name("bare-string") == "bare-string"
        assert io_common.get_dispatch_op_namespace(NamespacedOp()) == "mylib"
        assert io_common.get_dispatch_op_namespace(object()) == "aten"

    @pytest.mark.supplementary
    def test_should_inspect_dispatch_op(self):
        """An empty op filter matches everything; a set filter restricts."""
        assert io_common.should_inspect_dispatch_op("mm", True, set()) is True
        assert io_common.should_inspect_dispatch_op("mm", True, {"add"}) is False
        assert io_common.should_inspect_dispatch_op("add", True, {"add"}) is True

    @pytest.mark.supplementary
    def test_infer_backend_from_paths(self):
        """Registration paths map to backend labels by pattern priority."""

        def infer(path):
            return io_common._infer_backend_from_path(path)

        assert infer("/pytorch/build/aten/src/ATen/RegisterCPU_0.cpp:3456") == "CPU"
        assert infer("RegisterCUDA_0.cpp:1") == "CUDA"
        assert (
            infer("/b/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:7805")
            == "CompositeExplicitAutogradNonFunctional"
        )
        assert infer("/site/flag_gems/__init__.py:20") == "FlagGems"
        assert infer("/site/TRITON/ops/matmul.py:42") == "Triton"
        assert infer("/torch/_meta_registrations.py:50") == "Meta"
        assert infer("/somewhere/unknown.py:9") == "unknown"

    @pytest.mark.supplementary
    def test_get_dispatch_keys_returns_cached_triples(self):
        """get_dispatch_keys returns (key, impl, is_default) triples, then caches."""
        mm = torch.ops.aten.mm.default
        first = io_common.get_dispatch_keys(mm)
        assert first
        assert all(len(entry) == 3 for entry in first)
        assert io_common.get_dispatch_keys(mm) is io_common.get_dispatch_keys(mm)

    @pytest.mark.supplementary
    def test_get_dispatch_keys_without_name_falls_back(self):
        """A func without .name() yields the single unknown-unknown triple."""

        class NoName:
            namespace = "aten"

        assert io_common.get_dispatch_keys(NoName()) == [("unknown", "unknown", True)]


class TestTensorStatsRegistry:
    """register_tensor_stat validation, replacement and skip rules."""

    @pytest.mark.supplementary
    def test_empty_name_rejected(self):
        """Blank or whitespace-only stat names raise ValueError."""
        for bad in ("", "   "):
            with pytest.raises(ValueError, match="non-empty string"):
                io_common.register_tensor_stat(bad, lambda t: 0.0)

    @pytest.mark.supplementary
    def test_custom_stat_registered_then_replaced(self):
        """A new stat is added; re-registering the same name replaces it."""
        io_common.register_tensor_stat("supp_probe", lambda t: 42.0)
        assert io_common.tensor_stats(torch.ones(2))["supp_probe"] == 42.0
        io_common.register_tensor_stat("supp_probe", lambda t: 7.0)
        assert io_common.tensor_stats(torch.ones(2))["supp_probe"] == 7.0

    @pytest.mark.supplementary
    def test_float_only_stats_skip_integer_tensors(self):
        """min/max run on integer tensors but float-only stats are skipped."""
        stats = io_common.tensor_stats(
            torch.tensor([[1, 4], [2, 3]], dtype=torch.int64)
        )
        assert stats["min"] == 1 and stats["max"] == 4
        assert "mean" not in stats and "std" not in stats

    @pytest.mark.supplementary
    def test_empty_and_complex_tensors_return_meta_only(self):
        """Empty or complex tensors produce only shape/dtype/device metadata."""
        empty = io_common.tensor_stats(torch.zeros(0))
        assert empty == {"shape": [0], "dtype": "torch.float32", "device": "cpu"}
        complex_stats = io_common.tensor_stats(torch.tensor([1 + 2j]))
        assert "min" not in complex_stats


class TestFormattingHelpers:
    """format_value / format_result / tag and label builders."""

    @pytest.mark.supplementary
    def test_format_value_scalar_and_container_kinds(self):
        """Scalars, bools, lists and unknown objects each format distinctly."""
        assert io_common.format_value(None) == "None"
        assert io_common.format_value(True) == "True"
        assert io_common.format_value(3) == "3"
        assert io_common.format_value(2.5) == "2.5"
        assert io_common.format_value([1, 2]) == "list([1, 2])"
        assert io_common.format_value((1, 2)) == "tuple([1, 2])"
        assert io_common.format_value(list(range(6))) == "list(len=6)"
        assert io_common.format_value(object()) == "object(...)"

    @pytest.mark.supplementary
    def test_format_value_tensor_includes_stats(self):
        """Tensor formatting embeds shape/dtype/device and registered stats."""
        formatted = io_common.format_value(torch.tensor([[1.0, 2.0]]))
        assert formatted.startswith("Tensor(")
        assert "shape=[1, 2]" in formatted
        assert "min=1" in formatted and "max=2" in formatted

    @pytest.mark.supplementary
    def test_format_result_single_and_tuple(self):
        """Single results get one line; tuples get indexed lines."""
        tensor = torch.zeros(1)
        single = io_common.format_result(tensor)
        assert single.startswith("  result: Tensor(")
        pair = io_common.format_result((tensor, 5))
        assert pair.splitlines()[0].startswith("  result[0]: Tensor(")
        assert pair.splitlines()[1] == "  result[1]: 5"

    @pytest.mark.supplementary
    def test_make_module_tag_from_ctx_variants(self):
        """Tag building honours empty names, paths and the for_json flag."""
        assert io_common.make_module_tag_from_ctx("Linear", "a.b") == "[Linear:a.b]"
        assert io_common.make_module_tag_from_ctx("Linear", "") == "[Linear]"
        assert io_common.make_module_tag_from_ctx("", "x") == ""
        assert io_common.make_module_tag_from_ctx("L", "p", for_json=True) == "L:p"

    @pytest.mark.supplementary
    def test_make_module_tag_from_live_context(self):
        """make_module_tag reads the live module counter stack."""
        assert io_common.make_module_tag() == ""
        io_common.push_module_context("Linear", None)
        assert io_common.make_module_tag() == "[module=0,1]"

    @pytest.mark.supplementary
    def test_make_op_tag_increments_counter(self):
        """Each make_op_tag call advances that op's per-step counter."""
        first = io_common.make_op_tag("mm")
        second = io_common.make_op_tag("mm")
        other = io_common.make_op_tag("add")
        assert first == "[op=0,1]"
        assert second == "[op=0,2]"
        assert other == "[op=1,1]"

    @pytest.mark.supplementary
    def test_make_label_composes_parts(self):
        """Labels combine module, layer and dispatch-key segments."""
        label = io_common.make_label(
            "mm",
            (),
            module_name="Linear",
            layer_path="model.layers.0",
            dispatch_keys=[("CPU", "CPU", False), ("CUDA", "FlagGems", True)],
        )
        assert label.startswith("mm (module=Linear, layer=model.layers.0, ")
        assert label.endswith("dispatch_keys=[(CPU, CPU), (CUDA, FlagGems)])")

    @pytest.mark.supplementary
    def test_make_label_defaults_from_module_context(self):
        """Without explicit args the label falls back to the context stack."""
        io_common.push_module_context("Linear", None)
        label = io_common.make_label("mm")
        assert label == "mm (module=Linear)"
        args_label = io_common.make_label("mm", (torch.nn.Linear(2, 2),))
        assert args_label == "mm (module=Linear)"

    @pytest.mark.supplementary
    def test_get_module_class_name_from_args(self):
        """args[0] being an nn.Module yields its class name; otherwise None."""
        assert io_common.get_module_class_name((torch.nn.Linear(1, 1), 2)) == "Linear"
        assert io_common.get_module_class_name((1, 2)) is None
        assert io_common.get_module_class_name(()) is None

    @pytest.mark.supplementary
    def test_record_seen_tracks_ops_and_modules(self):
        """record_seen fills the per-step seen sets from args or explicit names."""
        io_common.record_seen("mm", (torch.nn.Linear(1, 1),))
        io_common.record_seen("add", module_name="Conv")
        assert io_common._seen_ops == {"mm", "add"}
        assert io_common._seen_modules == {"Linear", "Conv"}


class TestTorchFuncTags:
    """acquire/release torch-func tag coordination between nested modes."""

    @pytest.mark.supplementary
    def test_nested_calls_share_outer_tags(self):
        """Nested acquisitions reuse the outermost tags without double counting."""
        io_common.reset_exec_order()
        module_tag, op_tag, order = io_common.acquire_torch_func_tags("mm")
        assert order == 1 and op_tag == "[op=0,1]"
        nested = io_common.acquire_torch_func_tags("mm")
        assert nested == (module_tag, op_tag, order)
        io_common.release_torch_func_tags()
        io_common.release_torch_func_tags()

        # After full release a fresh acquisition generates new tags.
        _, new_op_tag, new_order = io_common.acquire_torch_func_tags("add")
        assert new_order == 2 and new_op_tag == "[op=1,1]"
        io_common.release_torch_func_tags()

    @pytest.mark.supplementary
    def test_release_without_acquire_is_noop(self):
        """Releasing tags with depth zero does not raise or corrupt state."""
        io_common.release_torch_func_tags()
        module_tag, op_tag, order = io_common.acquire_torch_func_tags("mm")
        assert order == 1
        io_common.release_torch_func_tags()


class TestYamlConfigParsing:
    """parse_io_config_from_yaml and its section parsers."""

    @pytest.mark.supplementary
    def test_missing_and_invalid_files_return_empty(self, tmp_path):
        """Missing files, broken YAML and missing io_dump sections all yield {}."""
        assert io_common.parse_io_config_from_yaml(str(tmp_path / "nope.yaml")) == {}
        bad = tmp_path / "bad.yaml"
        bad.write_text(":\n  - {broken", encoding="utf-8")
        assert io_common.parse_io_config_from_yaml(str(bad)) == {}
        no_section = tmp_path / "other.yaml"
        no_section.write_text("unrelated: true\n", encoding="utf-8")
        assert io_common.parse_io_config_from_yaml(str(no_section)) == {}

    @pytest.mark.supplementary
    def test_full_section_roundtrip(self, tmp_path):
        """Every supported key parses into its normalized form."""
        config = tmp_path / "full.yaml"
        config.write_text(
            "\n".join(
                [
                    "io_dump:",
                    "  dir: /tmp/dump",
                    "  with_print: true",
                    "  with_values: true",
                    "  with_metas: false",
                    "  ops: [rms_norm, silu_and_mul]",
                    '  modules: "Linear,Conv"',
                    '  layers: "0-2,lm_head"',
                    "  max_calls: 7",
                    '  step_range: "5-15"',
                    "  with_torch_funcs: [matmul, softmax]",
                    "  ranks: [0, 2]",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        parsed = io_common.parse_io_config_from_yaml(str(config))["io_dump"]
        assert parsed["dir"] == "/tmp/dump"
        assert parsed["ops"] == {"rms_norm", "silu_and_mul"}
        assert parsed["modules"] == {"Linear", "Conv"}
        assert parsed["layers"] == {
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
            "lm_head",
        }
        assert parsed["max_calls"] == 7
        assert parsed["step_range"] == (5, 16)
        assert parsed["with_torch_funcs"] == (True, {"matmul", "softmax"})
        assert parsed["ranks"] == {0, 2}
        assert parsed["with_values"] is True
        assert parsed["with_metas"] is False
        assert parsed["with_print"] is True

    @pytest.mark.supplementary
    @pytest.mark.parametrize(
        ("step_range_yaml", "expected"),
        [
            ("5", (5, 6)),
            ('"5-15"', (5, 16)),
            ("[3, 9]", (3, 10)),
        ],
    )
    def test_step_range_yaml_forms(self, tmp_path, step_range_yaml, expected):
        """int, dash-string and legacy [start, end] list forms all normalize."""
        config = tmp_path / "sr.yaml"
        config.write_text(
            f"io_dump:\n  step_range: {step_range_yaml}\n", encoding="utf-8"
        )
        parsed = io_common.parse_io_config_from_yaml(str(config))["io_dump"]
        assert parsed["step_range"] == expected

    @pytest.mark.supplementary
    @pytest.mark.parametrize("step_range_yaml", ["junk", '"a-b"', "[1, 2, 3]"])
    def test_step_range_invalid_forms_return_none(self, tmp_path, step_range_yaml):
        """Unparsable step_range values yield None instead of raising."""
        config = tmp_path / "bad.yaml"
        config.write_text(
            f"io_dump:\n  step_range: {step_range_yaml}\n", encoding="utf-8"
        )
        assert (
            io_common.parse_io_config_from_yaml(str(config))["io_dump"]["step_range"]
            is None
        )

    @pytest.mark.supplementary
    def test_invalid_max_calls_falls_back_to_zero(self, tmp_path):
        """A non-integer max_calls degrades to 0, and absent optionals stay absent."""
        config = tmp_path / "mc.yaml"
        config.write_text("io_dump:\n  max_calls: notanint\n", encoding="utf-8")
        parsed = io_common.parse_io_config_from_yaml(str(config))["io_dump"]
        assert parsed["max_calls"] == 0
        for absent in ("with_torch_funcs", "with_metas", "with_print"):
            assert absent not in parsed

    @pytest.mark.supplementary
    @pytest.mark.parametrize(
        ("ranks_yaml", "expected"),
        [
            ("all", None),
            ("3", {3}),
            ("[0, x, 4]", {0, 4}),
            ('"0,2,4"', {0, 2, 4}),
            ("[x, y]", None),
        ],
    )
    def test_ranks_yaml_forms(self, tmp_path, ranks_yaml, expected):
        """ranks accepts lists, ints, strings and 'all', dropping bad entries."""
        config = tmp_path / "r.yaml"
        config.write_text(f"io_dump:\n  ranks: {ranks_yaml}\n", encoding="utf-8")
        parsed = io_common.parse_io_config_from_yaml(str(config))["io_dump"]
        assert parsed["ranks"] == expected

    @pytest.mark.supplementary
    @pytest.mark.parametrize(
        ("torch_funcs_yaml", "expected"),
        [
            ("true", (True, set())),
            ("false", (False, set())),
            ("matmul,softmax", (True, {"matmul", "softmax"})),
            ('"1"', (True, set())),
            ('"0"', (False, set())),
            ("[]", (False, set())),
            ("{a: 1}", (False, set())),
            ("1", (False, set())),
        ],
    )
    def test_with_torch_funcs_yaml_forms(self, tmp_path, torch_funcs_yaml, expected):
        """with_torch_funcs accepts booleans, strings and lists.

        Note the corner: a bare YAML int ``1`` is neither ``True`` nor a
        string, so it falls through to ``(False, set())`` — only the quoted
        string ``"1"`` enables match-all capture.
        """
        config = tmp_path / "tf.yaml"
        config.write_text(
            f"io_dump:\n  with_torch_funcs: {torch_funcs_yaml}\n", encoding="utf-8"
        )
        parsed = io_common.parse_io_config_from_yaml(str(config))["io_dump"]
        assert parsed["with_torch_funcs"] == expected


class TestListModelLayers:
    """list_model_layers depth filtering (complement to io_dumper tests)."""

    @pytest.mark.supplementary
    def test_max_depth_limits_reported_modules(self):
        """Depth-N output lists only modules nested up to N levels."""

        class Child(torch.nn.Module):
            pass

        class Parent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Child()
                self.child.grand = Child()

        model = Parent()
        depth1 = io_common.list_model_layers(model, max_depth=1)
        assert depth1 == ["child (Child)"]
        depth2 = io_common.list_model_layers(model, max_depth=2)
        assert depth2 == ["child (Child)", "child.grand (Child)"]
        unlimited = io_common.list_model_layers(model)
        assert unlimited == depth2
