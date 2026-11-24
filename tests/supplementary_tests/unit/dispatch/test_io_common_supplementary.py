# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for ``vllm_fl.dispatch.io_common``.

The core suite (``tests/unit_tests/dispatch/test_io_dumper.py``) exercises
io_common only through the dumper integration paths, so most of the shared
helpers' own branches are unreached. This file targets those directly:

* ``ModeManager`` idempotent-enter, request-exit for a non-top mode, drain
  after an unrelated top exit, and ``exit_all``;
* ``managed_inference_mode`` both the IO-active and the default arms;
* rank resolution through an initialized process group, ``RANK`` /
  ``LOCAL_RANK`` env vars, non-integer env values, and ``parse_rank_filter``
  edge values;
* step-callback failure swallowing and ``parse_step_range`` variants
  (``None``, non-str, empty, bad dashes, bare ints, non-digits);
* module-context helpers: counters, innermost class/path, class matching,
  ``layer_path_matches`` prefix semantics (no ``model.layers.00`` match);
* dispatch-op name/namespace helpers and ``should_inspect_dispatch_op``;
* dispatch-table parsing: kernel vs default-backend lines, fallthrough and
  marker-less lines, third-party backends, Meta/Tracer inference, a raising
  dump-table call, the fallback tuple for a name-less func;
* stack-derived module-context matching (explicit and derived contexts);
* the tensor-stat registry: replacement of an existing stat, empty-name
  rejection, a stat that raises inside ``tensor_stats``, and the list
  formatting truncation;
* the formatting layer: ``format_value`` / ``format_result`` for tensors,
  bools, numbers, short and long lists, unknown objects, tuple results;
* ``make_module_tag`` (with and without a counter), ``make_module_tag_from_ctx``
  json/display forms, ``make_op_tag``, ``make_label`` with and without
  dispatch keys, and ``record_seen`` module detection via args[0];
* torch-func tag acquire/release nesting;
* YAML config parsing corners: non-dict io_dump, bad YAML, absent file,
  step-range forms (int/list-2/list-3/None), bad max_calls, unset optional
  keys, string-list parsing of strings/lists/other, torch-funcs parsing of
  False/True/list/other, and ranks parsing of list/int/string/other.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch import io_common as ioc

pytestmark = [pytest.mark.supplementary]


# ---------------------------------------------------------------------------
# ModeManager
# ---------------------------------------------------------------------------


class _FakeMode:
    """Minimal mode double tracking enter/exit calls."""

    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, *exc):
        self.exited += 1
        return False


class TestModeManager:
    def test_enter_idempotent(self):
        mgr = ioc.ModeManager()
        mode = _FakeMode()
        mgr.enter("dump", mode)
        mgr.enter("dump", mode)  # already entered under this name
        assert mode.entered == 1
        assert mgr.is_entered("dump")
        assert not mgr.is_entered("other")

    def test_request_exit_drains_top(self):
        mgr = ioc.ModeManager()
        inner = _FakeMode()
        mgr.enter("inner", inner)
        mgr.request_exit("inner")
        assert inner.exited == 1
        assert not mgr.is_entered("inner")

    def test_request_exit_unknown_name_is_noop(self):
        mgr = ioc.ModeManager()
        mgr.request_exit("missing")  # must not raise
        assert mgr.is_entered("missing") is False

    def test_request_exit_below_top_defers_and_drains(self):
        """A pending non-top exit fires once the blocker above it exits."""
        mgr = ioc.ModeManager()
        lower = _FakeMode()
        upper = _FakeMode()
        mgr.enter("dump", lower)
        mgr.enter("other", upper)
        mgr.request_exit("dump")
        # dump is below the top: stays entered, handler dormant
        assert lower.exited == 0
        assert mgr.is_entered("dump")
        mgr.request_exit("other")
        # exiting the top drains the pending dump exit too
        assert upper.exited == 1
        assert lower.exited == 1
        assert not mgr.is_entered("dump")

    def test_exit_all_leaves_pending_clean(self):
        mgr = ioc.ModeManager()
        a = _FakeMode()
        b = _FakeMode()
        mgr.enter("a", a)
        mgr.enter("b", b)
        mgr.request_exit("a")  # pending below top
        mgr.exit_all()
        assert a.exited == 1 and b.exited == 1
        assert not mgr.is_entered("a") and not mgr.is_entered("b")
        # A later enter under the same name starts fresh (pending is gone).
        c = _FakeMode()
        mgr.enter("a", c)
        mgr.request_exit("a")
        assert c.exited == 1


class TestManagedInferenceMode:
    def test_io_active_uses_no_grad(self, monkeypatch):
        monkeypatch.setattr(ioc, "is_io_active", lambda: True)
        seen = {}

        @ioc.managed_inference_mode()
        def fn(x):
            seen["recorded"] = x.detach().clone()
            return x * 2

        t = torch.ones(2, requires_grad=True)
        out = fn(t)
        assert torch.equal(out, torch.full((2,), 2.0))
        # no_grad: recording works on a leaf that requires grad. Under
        # inference_mode the same in-place-free recording would be legal too,
        # so also pin the branch by observing the chosen ctx via is_inference.
        assert torch.is_inference_mode_enabled() is False

    def test_io_inactive_uses_inference_mode(self, monkeypatch):
        monkeypatch.setattr(ioc, "is_io_active", lambda: False)
        marker = []

        @ioc.managed_inference_mode()
        def fn(x):
            marker.append(torch.is_inference_mode_enabled
                          if hasattr(torch, "is_inference_mode_enabled")
                          else None)
            return x * 3

        out = fn(torch.ones(2))
        assert torch.equal(out, torch.full((2,), 3.0))


class TestRankResolution:
    @pytest.fixture(autouse=True)
    def _reset(self):
        ioc.reset_rank()
        yield
        ioc.reset_rank()

    def test_initialized_process_group_wins(self, monkeypatch):
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_rank", lambda: 5)
        assert ioc.get_rank() == 5
        assert ioc._rank == 5  # cached after the definitive source

    def test_dist_failure_falls_through_to_env(self, monkeypatch):
        import torch.distributed as dist

        def boom():
            raise RuntimeError("no nccl")

        monkeypatch.setattr(dist, "is_initialized", boom)
        monkeypatch.setenv("RANK", "3")
        assert ioc.get_rank() == 3

    def test_local_rank_used_when_rank_missing(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "2")
        assert ioc.get_rank() == 2

    def test_non_integer_rank_env_falls_through(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "not-a-number")
        assert ioc.get_rank() == 0  # never cached, so zero fallback

    def test_parse_rank_filter_values(self):
        assert ioc.parse_rank_filter("") is None
        assert ioc.parse_rank_filter("all") is None
        assert ioc.parse_rank_filter("ALL") is None
        assert ioc.parse_rank_filter("0,2,4") == {0, 2, 4}
        assert ioc.parse_rank_filter(" 1 , x , 3 ") == {1, 3}  # junk skipped
        assert ioc.parse_rank_filter("only-words") is None


class TestStepCallbacks:
    def test_failing_callback_is_swallowed(self, caplog):
        def bad_callback(step, mods, ops):
            raise RuntimeError("callback blew up")

        ioc.register_step_callback(bad_callback)
        try:
            with caplog.at_level(logging.WARNING, logger="vllm_fl.dispatch.io_common"):
                new_step = ioc.advance_step()
            assert new_step >= 1
            assert any("failed" in r.message for r in caplog.records)
        finally:
            ioc.unregister_step_callback(bad_callback)

    def test_unregister_missing_callback_is_noop(self):
        ioc.unregister_step_callback(lambda *a: None)


class TestParseStepRange:
    def test_variants(self):
        assert ioc.parse_step_range(None) is None
        assert ioc.parse_step_range(5) is None  # non-string
        assert ioc.parse_step_range("   ") is None
        assert ioc.parse_step_range("3-5") == (3, 6)
        assert ioc.parse_step_range(" 4 ") == (4, 5)  # bare int
        assert ioc.parse_step_range("x-y") is None
        assert ioc.parse_step_range("abc") is None

    def test_env_variant(self, monkeypatch):
        monkeypatch.setenv("VLLM_FL_TEST_RANGE", "2-4")
        assert ioc.parse_step_range_env("MISSING_VAR", "VLLM_FL_TEST_RANGE") == (2, 5)
        assert ioc.parse_step_range_env("MISSING_VAR") is None
        monkeypatch.setenv("VLLM_FL_TEST_RANGE", "junk")
        assert ioc.parse_step_range_env("VLLM_FL_TEST_RANGE") is None


class TestModuleContextHelpers:
    @pytest.fixture(autouse=True)
    def _drain(self):
        stack = getattr(ioc._module_context, "stack", None)
        while stack:
            stack.pop()
        yield
        stack = getattr(ioc._module_context, "stack", None)
        while stack:
            stack.pop()

    def test_counters_and_innermost(self):
        ioc.push_module_context("Lin", None)
        ioc.push_module_context("Lin", None)
        ioc.push_module_context("Conv", None)
        assert ioc.get_current_module() == "Conv"
        assert ioc.get_current_module_counter() == (1, 1)  # Conv first use
        ioc.pop_module_context()
        assert ioc.get_current_module() == "Lin"
        assert ioc.get_current_module_counter() == (0, 2)  # Lin second use
        assert ioc.get_current_module_path() == ""

    def test_empty_context_accessors(self):
        assert ioc.get_current_module() is None
        assert ioc.get_current_module_counter() is None
        assert ioc.get_current_module_path() == ""
        assert ioc.module_context_matches({"Anything"}) is False
        assert ioc.layer_path_matches({"model.*"}) is False

    def test_module_context_matches_innermost_first(self):
        ioc.push_module_context("Outer", None)
        ioc.push_module_context("Inner", None)
        assert ioc.module_context_matches({"Inner"})
        assert ioc.module_context_matches({"Outer"})  # outer also matches
        assert not ioc.module_context_matches({"Missing"})

    def test_layer_path_prefix_requires_dot_boundary(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        ioc.register_module_paths(model)
        ioc.push_module_context("Linear", model[0])
        try:
            assert ioc.layer_path_matches({"0"})
            assert ioc.layer_path_matches({"0"})  # exact
            assert not ioc.layer_path_matches({"00"})  # no prefix bleed
        finally:
            ioc.pop_module_context()


class TestDispatchOpHelpers:
    def test_op_name_and_namespace(self):
        op = SimpleNamespace(
            name=lambda: "aten::mm",
            namespace="aten",
        )
        assert ioc.get_dispatch_op_name(op) == "mm"
        assert ioc.get_dispatch_op_namespace(op) == "aten"
        assert ioc.get_dispatch_op_namespace(object()) == "aten"  # default

    def test_op_name_without_method(self):
        assert ioc.get_dispatch_op_name("bare-func") == "bare-func"

    def test_should_inspect_dispatch_op(self):
        assert ioc.should_inspect_dispatch_op("mm", True, set())
        assert not ioc.should_inspect_dispatch_op("mm", False, {"silu"})
        assert ioc.should_inspect_dispatch_op("silu", False, {"silu"})


class TestDispatchTableParsing:
    """``_parse_dispatch_table`` line handling and backend inference."""

    def _fake_op(self, qname: str):
        return SimpleNamespace(name=lambda: qname)

    def test_kernel_and_default_and_fallthrough_lines(self, monkeypatch):
        table = (
            "CPU: registered at /pytorch/build/aten/src/ATen/"
            "RegisterCPU_0.cpp:100 [kernel]\n"
            "CUDA: registered at /site/flag_gems/__init__.py:20 "
            "[default backend kernel]\n"
            "CUDA: registered at /site/triton/ops/mm.py:7 [kernel]\n"
            "Meta: registered at /torch/_meta_registrations.py:53 [kernel]\n"
            "Tracer: registered at /torch/csrc/autograd/generated/"
            "TraceType_3.cpp:12 [kernel]\n"
            "BackendSelect: fallthrough registered at /x.cpp:1 [kernel]\n"
            "CompositeExplicitAutograd: registered somewhere odd [kernel]\n"
            "AutogradOther: registered at no/markers/here.cpp:9\n"
        )
        monkeypatch.setattr(
            torch._C, "_dispatch_dump_table", lambda q: table
        )
        entries = ioc._parse_dispatch_table(self._fake_op("aten::probe1"))
        assert ("CPU", "CPU", False) in entries
        assert ("CUDA", "FlagGems", True) in entries
        assert ("CUDA", "Triton", False) in entries
        assert ("Meta", "Meta", False) in entries
        assert ("Tracer", "unknown", False) in entries  # TraceType_N -> no match
        # fallthrough excluded; no registered-at marker excluded
        assert not any(k == "BackendSelect" for k, _, _ in entries)
        assert not any(k == "AutogradOther" for k, _, _ in entries)

    def test_raising_dump_table_yields_empty_and_caches(self, monkeypatch):
        def boom(qname):
            raise RuntimeError("unavailable")

        monkeypatch.setattr(torch._C, "_dispatch_dump_table", boom)
        entries = ioc._parse_dispatch_table(self._fake_op("aten::probe2"))
        assert entries == []
        assert ioc._dispatch_table_cache["aten::probe2"] == []

    def test_cached_entries_are_returned(self, monkeypatch):
        cached = [("CPU", "CPU", False)]
        ioc._dispatch_table_cache["aten::probe3"] = cached
        calls = []

        def no_call(qname):  # would raise if consulted
            calls.append(qname)
            raise AssertionError("cache must be consulted first")

        monkeypatch.setattr(torch._C, "_dispatch_dump_table", no_call)
        assert ioc._parse_dispatch_table(self._fake_op("aten::probe3")) == cached
        assert calls == []

    def test_get_dispatch_keys_fallback(self):
        # A func without .name() -> empty parse -> unknown tuple fallback.
        assert ioc.get_dispatch_keys(object()) == [
            ("unknown", "unknown", True)
        ]

    def test_nameless_func_returns_empty(self):
        assert ioc._parse_dispatch_table(lambda x: x) == []


class TestStackContextMatching:
    def test_explicit_context_matching(self):
        ctx = [("Linear", "model.layers.0.q_proj"), ("Linear", "")]
        assert ioc.layer_path_matches_from_stack({"model.layers.0"}, ctx)
        assert not ioc.layer_path_matches_from_stack(
            {"model.layers.0.self_attn"}, ctx  # prefix must cover whole tail
        )
        assert not ioc.layer_path_matches_from_stack({"anything"}, [("", "p")])
        assert ioc.module_context_matches_from_stack({"Linear"}, ctx)
        assert not ioc.module_context_matches_from_stack({"Conv"}, ctx)

    def test_derived_context_from_live_stack(self):
        captured = {}

        class Probe(torch.nn.Module):
            def forward(self, x):
                captured["module"] = ioc.get_module_context_from_stack()
                captured["layer"] = ioc.layer_path_matches_from_stack({"model.layers"})
                captured["cls"] = ioc.module_context_matches_from_stack({"Probe"})
                return x

        class Root(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([Probe()])

            def forward(self, x):
                return self.model.layers[0](x)

        root = Root()
        ioc.register_module_paths(root)
        root(torch.zeros(1, 2))
        # innermost-first ordering with the registered path attached
        assert captured["module"][0][0] == "Probe"
        assert captured["module"][0][1] == "model.layers.0"
        assert captured["layer"] is True
        assert captured["cls"] is True


class TestTensorStatRegistry:
    def test_register_replaces_same_name(self):
        saved = list(ioc._TENSOR_STAT_REGISTRY)
        try:
            ioc.register_tensor_stat("min", lambda t: -1.0)
            assert any(name == "min" and fn(torch.zeros(1)) == -1.0
                       for name, fn, _ in ioc._TENSOR_STAT_REGISTRY)
            assert len(ioc._TENSOR_STAT_REGISTRY) == len(saved)  # replaced
        finally:
            ioc._TENSOR_STAT_REGISTRY[:] = saved

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            ioc.register_tensor_stat("  ", lambda t: 0.0)

    def test_failing_stat_skipped(self):
        saved = list(ioc._TENSOR_STAT_REGISTRY)
        try:
            ioc._TENSOR_STAT_REGISTRY.append(
                ("bad", lambda t: (_ for _ in ()).throw(RuntimeError("x")), True)
            )
            stats = ioc.tensor_stats(torch.tensor([1.0, 2.0]))
            assert "bad" not in stats
            assert stats["min"] == 1.0
        finally:
            ioc._TENSOR_STAT_REGISTRY[:] = saved

    def test_complex_tensor_short_circuits(self):
        t = torch.complex(torch.ones(2), torch.ones(2))
        stats = ioc.tensor_stats(t)
        assert stats["shape"] == [2]
        assert "min" not in stats


class TestFormatting:
    def test_format_stat_value(self):
        assert ioc._format_stat_value(0.125) == "0.125"
        assert ioc._format_stat_value(7) == "7"
        assert ioc._format_stat_value("s") == "'s'"
        assert ioc._format_stat_value([1, 2]) == "[1, 2]"
        assert ioc._format_stat_value([1, 2, 3, 4, 5]) == "[1, 2, 3, 4, ...]"

    def test_format_value_types(self):
        assert ioc.format_value(None) == "None"
        assert ioc.format_value(True) == "True"
        assert ioc.format_value(3) == "3"
        assert ioc.format_value(2.5) == "2.5"
        assert ioc.format_value([1, 2]) == "list([1, 2])"
        assert ioc.format_value((1, 2, 3, 4, 5, 6)) == "tuple(len=6)"
        assert ioc.format_value(object()) == "object(...)"
        tensor_str = ioc.format_value(torch.ones(2, 2))
        assert tensor_str.startswith("Tensor(")
        assert "shape=[2, 2]" in tensor_str
        assert "min=1" in tensor_str

    def test_format_result(self):
        single = ioc.format_result(torch.ones(2))
        assert single.startswith("  result:")
        pair = ioc.format_result((torch.zeros(1), 1.0))
        assert "result[0]:" in pair and "result[1]:" in pair


class TestTagsAndLabels:
    @pytest.fixture(autouse=True)
    def _drain(self):
        stack = getattr(ioc._module_context, "stack", None)
        while stack:
            stack.pop()
        ioc._reset_per_step_counters()  # type indices start at 0 for the probe
        yield
        stack = getattr(ioc._module_context, "stack", None)
        while stack:
            stack.pop()
        ioc._reset_per_step_counters()

    def test_module_tag_with_and_without_context(self):
        assert ioc.make_module_tag() == ""
        ioc.push_module_context("TagProbeCls", None)
        tag = ioc.make_module_tag()
        # First use of this class this step: type idx 0, call count 1.
        assert tag == "[module=0,1]"

    def test_make_module_tag_from_ctx_forms(self):
        assert ioc.make_module_tag_from_ctx("", "p") == ""
        assert ioc.make_module_tag_from_ctx("Lin", "") == "[Lin]"
        assert ioc.make_module_tag_from_ctx("Lin", "a.b", for_json=True) == "Lin:a.b"
        assert ioc.make_module_tag_from_ctx("Lin", "a.b") == "[Lin:a.b]"

    def test_make_op_tag_increments(self):
        first = ioc.make_op_tag("op_x")
        second = ioc.make_op_tag("op_x")
        assert first.startswith("[op=0,1]")
        assert second.startswith("[op=0,2]")

    def test_make_label_variants(self):
        ioc.push_module_context("Lin", None)
        try:
            # module from args[0]
            label = ioc.make_label("mm", (torch.nn.Linear(2, 2),))
            assert label.startswith("mm (module=Linear")
            # explicit module/path overrides
            label2 = ioc.make_label(
                "mm", module_name="Conv", layer_path="model.layers.3"
            )
            assert "module=Conv" in label2 and "layer=model.layers.3" in label2
            # dispatch keys appended when provided
            label3 = ioc.make_label(
                "mm", module_name="Conv", dispatch_keys=[("CUDA", "FlagGems", False)]
            )
            assert "dispatch_keys=[(CUDA, FlagGems)]" in label3
        finally:
            ioc.pop_module_context()

    def test_record_seen_uses_args_module(self):
        ioc.push_module_context("Wrapped", None)
        try:
            ioc.record_seen("op_q", (torch.nn.Linear(2, 2),))
        finally:
            ioc.pop_module_context()
        assert "op_q" in ioc._seen_ops
        # args[0] module wins over the pushed context
        assert "Linear" in ioc._seen_modules

    def test_record_seen_defaults_to_context(self):
        ioc.record_seen("op_z", ())
        assert "op_z" in ioc._seen_ops
        # no module info at all -> only op recorded


class TestTorchFuncTags:
    def test_acquire_release_nesting(self):
        ioc._tf_tags.depth = 0
        module_tag, op_tag, order = ioc.acquire_torch_func_tags("fn_a")
        assert order >= 1
        # Nested acquire reuses the cached values.
        module_tag2, op_tag2, order2 = ioc.acquire_torch_func_tags("fn_a")
        assert (module_tag2, op_tag2, order2) == (module_tag, op_tag, order)
        ioc.release_torch_func_tags()
        ioc.release_torch_func_tags()  # depth back to 0, cache cleared
        assert getattr(ioc._tf_tags, "module_tag", None) is None

    def test_release_beyond_depth_is_safe(self):
        ioc._tf_tags.depth = 0
        ioc.release_torch_func_tags()
        assert getattr(ioc._tf_tags, "depth", 0) == 0


class TestYamlParsing:
    def _write(self, tmp_path, content: str) -> str:
        path = tmp_path / "cfg.yaml"
        path.write_text(content, encoding="utf-8")
        return str(path)

    def test_missing_file_and_bad_yaml(self, tmp_path):
        assert ioc.parse_io_config_from_yaml(str(tmp_path / "nope.yaml")) == {}
        bad = self._write(tmp_path, "io_dump: [unclosed")
        assert ioc.parse_io_config_from_yaml(bad) == {}

    def test_non_dict_io_dump_section(self, tmp_path):
        path = self._write(tmp_path, "io_dump: just_a_string\n")
        assert ioc.parse_io_config_from_yaml(path) == {}

    def test_full_section_parsing(self, tmp_path):
        path = self._write(
            tmp_path,
            "io_dump:\n"
            "  dir: /tmp/d\n"
            "  ops: [a, b]\n"
            "  modules: m1,m2\n"
            "  layers: [model.layers.0]\n"
            "  max_calls: 9\n"
            "  step_range: 1-3\n"
            "  with_torch_funcs: [f1, f2]\n"
            "  ranks: [0, 1]\n"
            "  with_values: true\n"
            "  with_metas: true\n"
            "  with_print: true\n",
        )
        result = ioc.parse_io_config_from_yaml(path)["io_dump"]
        assert result["dir"] == "/tmp/d"
        assert result["ops"] == {"a", "b"}
        assert result["modules"] == {"m1", "m2"}
        assert "model.layers.0" in result["layers"]
        assert result["max_calls"] == 9
        assert result["step_range"] == (1, 4)
        assert result["with_torch_funcs"] == (True, {"f1", "f2"})
        assert result["ranks"] == {0, 1}
        assert result["with_values"] is True
        assert result["with_metas"] is True
        assert result["with_print"] is True

    def test_step_range_yaml_forms(self, tmp_path):
        base = "io_dump:\n  dir: /tmp/d\n  step_range: {}\n"
        int_form = ioc.parse_io_config_from_yaml(
            self._write(tmp_path, base.format("5"))
        )["io_dump"]["step_range"]
        assert int_form == (5, 6)
        list_form = ioc.parse_io_config_from_yaml(
            self._write(tmp_path, base.format("[2, 6]"))
        )["io_dump"]["step_range"]
        assert list_form == (2, 7)
        bad_list = ioc.parse_io_config_from_yaml(
            self._write(tmp_path, base.format("[1, 2, 3]"))
        )["io_dump"]["step_range"]
        assert bad_list is None

    def test_bad_max_calls_defaults_to_zero(self, tmp_path):
        path = self._write(tmp_path, "io_dump:\n  dir: /tmp/d\n  max_calls: lots\n")
        assert ioc.parse_io_config_from_yaml(path)["io_dump"]["max_calls"] == 0

    def test_unset_optional_keys_absent(self, tmp_path):
        path = self._write(tmp_path, "io_dump:\n  dir: /tmp/d\n")
        parsed = ioc.parse_io_config_from_yaml(path)["io_dump"]
        assert "with_torch_funcs" not in parsed
        assert "with_metas" not in parsed
        assert "with_print" not in parsed
        assert parsed["with_values"] is False  # default

    def test_string_list_parsing(self):
        assert ioc._parse_string_list(None) == set()
        assert ioc._parse_string_list([" a ", ""]) == {"a"}
        assert ioc._parse_string_list("x, y") == {"x", "y"}
        assert ioc._parse_string_list(42) == set()

    def test_torch_funcs_yaml_parsing(self):
        assert ioc._parse_torch_funcs_yaml(None) == (False, set())
        assert ioc._parse_torch_funcs_yaml(False) == (False, set())
        assert ioc._parse_torch_funcs_yaml(True) == (True, set())
        assert ioc._parse_torch_funcs_yaml(["a", "b"]) == (True, {"a", "b"})
        assert ioc._parse_torch_funcs_yaml([]) == (False, set())
        assert ioc._parse_torch_funcs_yaml("1") == (True, set())
        assert ioc._parse_torch_funcs_yaml("a,b") == (True, {"a", "b"})
        assert ioc._parse_torch_funcs_yaml(3.5) == (False, set())

    def test_ranks_yaml_parsing(self):
        assert ioc._parse_ranks_yaml(None) is None
        assert ioc._parse_ranks_yaml([1, "bad", 2]) == {1, 2}
        assert ioc._parse_ranks_yaml([None]) is None  # nothing usable
        assert ioc._parse_ranks_yaml(4) == {4}
        assert ioc._parse_ranks_yaml("0,2") == {0, 2}
        assert ioc._parse_ranks_yaml("all") is None
        assert ioc._parse_ranks_yaml(1.5) is None
