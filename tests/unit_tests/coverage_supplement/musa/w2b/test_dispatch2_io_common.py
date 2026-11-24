# Copyright (c) 2026 BAAI. All rights reserved.

"""Extra branch coverage for ``vllm_fl.dispatch.io_common``.

Complements the existing ModeManager tests (w2) and the repo-level
io_dumper tests by exercising rank resolution, step/counter bookkeeping,
module context, dispatch-table parsing, formatting, tag helpers and the
YAML config parser.
"""

import types

import pytest
import torch
import torch.nn as nn

from vllm_fl.dispatch import io_common as c


@pytest.fixture(autouse=True)
def _reset_globals():
    saved_stats = list(c._TENSOR_STAT_REGISTRY)
    saved_callbacks = list(c._step_callbacks)
    c.reset_rank()
    c.reset_step()
    c.reset_exec_order()
    c._dispatch_table_cache.clear()
    c._module_path_map.clear()
    c._seen_modules.clear()
    c._seen_ops.clear()
    for tls in (c._module_context, c._tf_tags):
        for attr in ("stack", "depth", "module_tag", "op_tag", "exec_order"):
            try:
                delattr(tls, attr)
            except AttributeError:
                pass
    yield
    c.set_io_active(False)
    c.reset_rank()
    c.reset_step()
    c.reset_exec_order()
    c._TENSOR_STAT_REGISTRY[:] = saved_stats
    c._step_callbacks[:] = saved_callbacks
    c._dispatch_table_cache.clear()
    c._module_path_map.clear()
    c._seen_modules.clear()
    c._seen_ops.clear()
    for tls in (c._module_context, c._tf_tags):
        for attr in ("stack", "depth", "module_tag", "op_tag", "exec_order"):
            try:
                delattr(tls, attr)
            except AttributeError:
                pass


# ── Rank resolution ──


class TestGetRank:
    def test_distributed_rank_wins(self, monkeypatch):
        dist = pytest.importorskip("torch.distributed")
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_rank", lambda: 3)
        assert c.get_rank() == 3
        assert c._rank == 3

    def test_distributed_exception_falls_back_to_env(self, monkeypatch):
        dist = pytest.importorskip("torch.distributed")

        def _boom():
            raise RuntimeError("dist unavailable")

        monkeypatch.setattr(dist, "is_initialized", _boom)
        monkeypatch.setenv("RANK", "2")
        assert c.get_rank() == 2

    def test_env_rank_cached(self, monkeypatch):
        monkeypatch.setenv("RANK", "4")
        assert c.get_rank() == 4
        monkeypatch.setenv("RANK", "9")
        assert c.get_rank() == 4  # cached

    def test_invalid_rank_uses_local_rank(self, monkeypatch):
        monkeypatch.setenv("RANK", "not-int")
        monkeypatch.setenv("LOCAL_RANK", "5")
        assert c.get_rank() == 5

    def test_invalid_envs_return_zero_uncached(self, monkeypatch):
        monkeypatch.setenv("RANK", "x")
        monkeypatch.setenv("LOCAL_RANK", "y")
        assert c.get_rank() == 0
        assert c._rank is None  # fallback is never cached
        monkeypatch.setenv("RANK", "6")
        assert c.get_rank() == 6

    def test_reset_rank_clears_cache(self, monkeypatch):
        monkeypatch.setenv("RANK", "1")
        assert c.get_rank() == 1
        c.reset_rank()
        assert c._rank is None


class TestParseRankFilter:
    @pytest.mark.parametrize("value", ["", "   ", "all", "ALL", " All "])
    def test_all_forms(self, value):
        assert c.parse_rank_filter(value) is None

    def test_single(self):
        assert c.parse_rank_filter("0") == {0}

    def test_csv_with_spaces(self):
        assert c.parse_rank_filter("0, 2 ,,4") == {0, 2, 4}

    def test_all_invalid(self):
        assert c.parse_rank_filter("a,b") is None

    def test_mixed_valid_invalid(self):
        assert c.parse_rank_filter("1,a") == {1}


# ── Execution order / counters ──


class TestExecOrder:
    def test_monotonic_and_reset(self):
        assert c.next_exec_order() == 1
        assert c.next_exec_order() == 2
        assert c.get_exec_order() == 2
        c.reset_exec_order()
        assert c.get_exec_order() == 0
        assert c.next_exec_order() == 1


class TestCounters:
    def test_module_counter(self):
        assert c.next_module_counter("Linear") == (0, 1)
        assert c.next_module_counter("Linear") == (0, 2)
        assert c.next_module_counter("RMSNorm") == (1, 1)

    def test_op_counter(self):
        assert c.next_op_counter("mm") == (0, 1)
        assert c.next_op_counter("mm") == (0, 2)
        assert c.next_op_counter("add") == (1, 1)

    def test_reset_per_step_returns_seen_and_clears(self):
        c.next_module_counter("Linear")
        c.next_op_counter("mm")
        c._seen_modules.add("Linear")
        c._seen_ops.add("mm")
        seen_mods, seen_ops = c._reset_per_step_counters()
        assert seen_mods == {"Linear"}
        assert seen_ops == {"mm"}
        # counters restart at 0 after reset
        assert c.next_module_counter("Linear") == (0, 1)
        assert c.next_op_counter("mm") == (0, 1)


# ── Step tracking ──


class TestStepTracking:
    def test_advance_fires_callbacks_and_resets(self, monkeypatch):
        calls = []
        mods = []
        monkeypatch.setattr(c, "_seen_modules", {"Linear"})
        monkeypatch.setattr(c, "_seen_ops", {"mm"})

        def cb(prev, seen_m, seen_o):
            calls.append(prev)
            mods.append((set(seen_m), set(seen_o)))

        c.register_step_callback(cb)
        assert c.advance_step() == 1
        assert calls == [0]
        assert mods == [({"Linear"}, {"mm"})]
        assert c.get_exec_order() == 0

    def test_callback_exception_is_swallowed(self):
        def bad(prev, seen_m, seen_o):
            raise ValueError("callback failed")

        c.register_step_callback(bad)
        assert c.advance_step() == 1  # must not raise

    def test_register_is_idempotent_and_unregister(self):
        calls = []

        def cb(*_):
            calls.append(1)

        c.register_step_callback(cb)
        c.register_step_callback(cb)
        assert c._step_callbacks.count(cb) == 1
        c.unregister_step_callback(cb)
        c.unregister_step_callback(cb)  # missing -> no error
        c.advance_step()
        assert calls == []

    def test_reset_step(self):
        c.advance_step()
        c.reset_step()
        assert c.get_step() == 0


class TestParseStepRange:
    @pytest.mark.parametrize("value", [None, 5, [0, 2], "", "   "])
    def test_invalid_inputs(self, value):
        assert c.parse_step_range(value) is None

    def test_dash_range(self):
        assert c.parse_step_range("0-2") == (0, 3)

    def test_dash_range_with_spaces(self):
        assert c.parse_step_range(" 3 - 5 ") == (3, 6)

    def test_bare_int(self):
        assert c.parse_step_range("5") == (5, 6)

    def test_bad_dash(self):
        assert c.parse_step_range("a-b") is None

    def test_non_digit_str(self):
        assert c.parse_step_range("abc") is None


class TestParseStepRangeEnv:
    def test_first_valid(self, monkeypatch):
        monkeypatch.setenv("A_RANGE", "")
        monkeypatch.setenv("B_RANGE", "0-1")
        monkeypatch.setenv("C_RANGE", "5")
        assert c.parse_step_range_env("A_RANGE", "B_RANGE", "C_RANGE") == (0, 2)

    def test_invalid_then_next(self, monkeypatch):
        monkeypatch.setenv("A_RANGE", "bad")
        monkeypatch.setenv("B_RANGE", "2")
        assert c.parse_step_range_env("A_RANGE", "B_RANGE") == (2, 3)

    def test_none(self, monkeypatch):
        monkeypatch.delenv("X_RANGE", raising=False)
        assert c.parse_step_range_env("X_RANGE") is None


# ── Module paths / context ──


class _PlaceholderModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)


class TestModulePathRegistry:
    def test_register_and_lookup(self):
        model = _PlaceholderModule()
        c.register_module_paths(model)
        assert c.get_module_path(model) == ""
        assert c.get_module_path(model.linear) == "linear"

    def test_unknown_module_returns_empty(self):
        assert c.get_module_path(nn.Linear(1, 1)) == ""


class TestModuleContext:
    def test_push_pop_and_lookup(self):
        model = _PlaceholderModule()
        c.register_module_paths(model)
        c.push_module_context("_PlaceholderModule", model)
        try:
            assert c.get_current_module() == "_PlaceholderModule"
            assert c.get_current_module_path() == ""
            assert c.get_current_module_counter() == (0, 1)
            assert c.module_context_matches({"_PlaceholderModule"}) is True
            assert c.module_context_matches({"Linear"}) is False
            c.push_module_context("Linear", model.linear)
            try:
                assert c.get_current_module() == "Linear"
                assert c.get_current_module_path() == "linear"
                assert c.module_context_matches({"Linear"}) is True
            finally:
                c.pop_module_context()
        finally:
            c.pop_module_context()

    def test_empty_context(self):
        assert c.get_current_module() is None
        assert c.get_current_module_counter() is None
        assert c.get_current_module_path() == ""
        assert c.module_context_matches({"Linear"}) is False
        # pop on empty stack must not raise
        c.pop_module_context()


class TestLayerMatching:
    def test_no_stack(self):
        assert c.layer_path_matches({"model.layers.0"}) is False

    def test_exact_prefix_glob_and_miss(self):
        model = _PlaceholderModule()
        c.register_module_paths(model)
        c.push_module_context("Linear", model.linear)  # path == "linear"
        try:
            assert c.layer_path_matches({"linear"}) is True
            assert c.layer_path_matches({"lin"}) is False  # prefix requires '.'
            assert c.layer_path_matches({"li*"}) is True   # glob
            assert c.layer_path_matches({"nope"}) is False
        finally:
            c.pop_module_context()

    def test_prefix_dot_semantics(self):
        model = _PlaceholderModule()
        c.register_module_paths(model)
        c.push_module_context("_PlaceholderModule", model)
        c.push_module_context("Linear", model.linear)
        try:
            assert c.layer_path_matches({"linear.sub"}) is False
            assert c.layer_path_matches({"linear.*"}) is False
        finally:
            c.pop_module_context()
            c.pop_module_context()


class TestExpandAndParseLayers:
    def test_expand_empty_and_whitespace(self):
        assert c.expand_layer_specs({"", "  "}) == set()

    def test_expand_integer_range_path_glob(self):
        got = c.expand_layer_specs({"0", "1-2", "custom.path", "a.*"})
        assert got == {
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
            "custom.path",
            "a.*",
        }

    def test_expand_custom_prefix(self):
        assert c.expand_layer_specs({"3"}, prefix="x.") == {"x.3"}

    def test_parse_layers_env_empty(self, monkeypatch):
        monkeypatch.delenv("NO_LAYERS", raising=False)
        assert c.parse_layers_env("NO_LAYERS") == set()

    def test_parse_layers_env_expands(self, monkeypatch):
        monkeypatch.setenv("MY_LAYERS", " 0 , 1-2 , model.layers.*.attn ")
        got = c.parse_layers_env("MY_LAYERS")
        assert "model.layers.0" in got
        assert "model.layers.1" in got
        assert "model.layers.2" in got
        assert "model.layers.*.attn" in got


# ── Torch function filtering ──


class TestTorchFuncHelpers:
    def test_get_torch_func_name(self):
        def matmul():
            pass

        assert c.get_torch_func_name(matmul) == "matmul"
        obj = object()
        assert c.get_torch_func_name(obj) == str(obj)

    def test_should_inspect_disabled(self):
        assert c.should_inspect_torch_func("mm", False, set(), True, set()) is False

    def test_should_inspect_explicit_filter(self):
        assert c.should_inspect_torch_func("mm", True, {"mm"}, False, set()) is True
        assert c.should_inspect_torch_func("add", True, {"mm"}, False, set()) is False

    def test_should_inspect_match_all_skips(self):
        assert c.should_inspect_torch_func("_x", True, set(), True, set()) is False
        assert c.should_inspect_torch_func("size", True, set(), True, set()) is False

    def test_should_inspect_op_filter(self):
        assert c.should_inspect_torch_func("mm", True, set(), False, {"mm"}) is True
        assert c.should_inspect_torch_func("add", True, set(), False, {"mm"}) is False

    def test_should_inspect_plain_all(self):
        assert c.should_inspect_torch_func("mm", True, set(), True, set()) is True


class TestDispatchNames:
    def test_short_name(self):
        class _F:
            def name(self):
                return "aten::mm"

        assert c.get_dispatch_op_name(_F()) == "mm"

    def test_name_without_namespace(self):
        class _F:
            def name(self):
                return "plain"

        assert c.get_dispatch_op_name(_F()) == "plain"

    def test_no_name_attr(self):
        obj = object()
        assert c.get_dispatch_op_name(obj) == str(obj)

    def test_namespace(self):
        assert c.get_dispatch_op_namespace(types.SimpleNamespace(namespace="vllm")) == "vllm"
        assert c.get_dispatch_op_namespace(object()) == "aten"


# ── Dispatch table parsing ──


class _NamedOp:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class TestInferBackend:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/pytorch/build/aten/src/ATen/RegisterCPU_0.cpp:1", "CPU"),
            ("/x/RegisterCUDA_0.cpp:2", "CUDA"),
            (
                "/x/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:3",
                "CompositeExplicitAutogradNonFunctional",
            ),
            ("/x/flag_gems/__init__.py:20", "FlagGems"),
            ("/x/triton/ops/matmul.py:42", "Triton"),
            ("/x/torch/_meta_registrations.py:50", "Meta"),
            ("weird/path", "unknown"),
        ],
    )
    def test_infer(self, path, expected):
        assert c._infer_backend_from_path(path) == expected


class TestDispatchTable:
    TABLE = "\n".join(
        [
            "CPU: registered at /pytorch/RegisterCPU_0.cpp:1 [kernel]",
            "CUDA: registered at /x/flag_gems/__init__.py:20 [kernel]",
            "Meta: registered at /x/torch/_meta_registrations.py:50 [default backend kernel]",
            "HIP: registered at /x/RegisterCompositeExplicitAutogradNonFunctional_0.cpp:3 [default backend kernel]",
            "Foo: registered at /x/RegisterCPU_0.cpp:1 [kernel] fallthrough",
            "Bar: [kernel]",
        ]
    )

    def test_parse_entries_and_cache(self, monkeypatch):
        op = _NamedOp("mylib::op")
        monkeypatch.setattr(torch._C, "_dispatch_dump_table", lambda q: self.TABLE)
        entries = c._parse_dispatch_table(op)
        assert entries == [
            ("CPU", "CPU", False),
            ("CUDA", "FlagGems", False),
            ("Meta", "Meta", True),
            ("HIP", "CompositeExplicitAutogradNonFunctional", True),
        ]
        # second call is served from cache even if the dump changes
        monkeypatch.setattr(torch._C, "_dispatch_dump_table", lambda q: "changed")
        assert c._parse_dispatch_table(op) == entries

    def test_parse_without_name(self):
        assert c._parse_dispatch_table(object()) == []

    def test_parse_exception_returns_empty(self, monkeypatch):
        op = _NamedOp("mylib::raises")
        monkeypatch.setattr(
            torch._C,
            "_dispatch_dump_table",
            lambda q: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert c._parse_dispatch_table(op) == []
        assert c.get_dispatch_keys(op) == [("unknown", "unknown", True)]

    def test_get_dispatch_keys_nonempty(self, monkeypatch):
        op = _NamedOp("mylib::ok")
        monkeypatch.setattr(torch._C, "_dispatch_dump_table", lambda q: self.TABLE)
        assert c.get_dispatch_keys(op)[0] == ("CPU", "CPU", False)


class TestShouldInspectDispatch:
    def test_filter(self):
        assert c.should_inspect_dispatch_op("mm", True, {"mm"}) is True
        assert c.should_inspect_dispatch_op("add", True, {"mm"}) is False
        assert c.should_inspect_dispatch_op("add", True, set()) is True


# ── Stack-based module context ──


class _StackModule(nn.Module):
    def forward(self):
        return c.get_module_context_from_stack()


class TestStackContext:
    def test_finds_module_in_forward(self):
        model = _StackModule()
        c.register_module_paths(model)
        result = model()
        assert ("_StackModule", "") in result

    def test_outside_module_is_empty(self):
        assert c.get_module_context_from_stack() == []

    def test_layer_path_matches_from_stack(self):
        ctx = [("Linear", "model.layers.0.self_attn")]
        assert c.layer_path_matches_from_stack({"model.layers.0"}, ctx) is True
        assert c.layer_path_matches_from_stack({"model.layers.*"}, ctx) is True
        assert c.layer_path_matches_from_stack({"model.layers.1"}, ctx) is False
        assert c.layer_path_matches_from_stack({"x"}, [("A", "")]) is False

    def test_layer_path_matches_from_stack_derives(self):
        assert c.layer_path_matches_from_stack({"x"}) is False

    def test_module_context_matches_from_stack(self):
        ctx = [("Linear", "a"), ("RMSNorm", "b")]
        assert c.module_context_matches_from_stack({"RMSNorm"}, ctx) is True
        assert c.module_context_matches_from_stack({"Other"}, ctx) is False
        assert c.module_context_matches_from_stack({"Other"}) is False


# ── Tensor statistics / formatting ──


class TestTensorStats:
    def test_empty_tensor(self):
        stats = c.tensor_stats(torch.zeros(0))
        assert set(stats) == {"shape", "dtype", "device"}

    def test_complex_tensor(self):
        stats = c.tensor_stats(torch.tensor([1 + 2j]))
        assert set(stats) == {"shape", "dtype", "device"}

    def test_int_tensor_skips_float_only(self):
        stats = c.tensor_stats(torch.tensor([1, 2, 3]))
        assert stats["min"] == 1
        assert stats["max"] == 3
        assert "mean" not in stats
        assert "std" not in stats

    def test_float_tensor_has_all(self):
        stats = c.tensor_stats(torch.tensor([1.0, 3.0]))
        assert stats["mean"] == pytest.approx(2.0)
        assert "std" in stats

    def test_register_empty_name_raises(self):
        with pytest.raises(ValueError):
            c.register_tensor_stat("", lambda t: 0)

    def test_register_custom_and_replace(self):
        base_len = len(c._TENSOR_STAT_REGISTRY)
        c.register_tensor_stat("custom_len", lambda t: int(t.numel()))
        assert len(c._TENSOR_STAT_REGISTRY) == base_len + 1
        c.register_tensor_stat("custom_len", lambda t: -1)
        assert len(c._TENSOR_STAT_REGISTRY) == base_len + 1
        assert c.tensor_stats(torch.ones(4))["custom_len"] == -1

    def test_failing_stat_is_swallowed(self):
        def boom(_t):
            raise RuntimeError("stat failed")

        c.register_tensor_stat("boom_stat", boom)
        assert "boom_stat" not in c.tensor_stats(torch.ones(2))


class TestFormatting:
    def test_format_stat_value(self):
        assert c._format_stat_value(0.123456789) == "0.123457"
        assert c._format_stat_value([1.23456789, 2]) == "[1.23457, 2]"
        long_list = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert c._format_stat_value(long_list).endswith(", ...]")
        assert c._format_stat_value((1, "a")) == "[1, 'a']"
        assert c._format_stat_value(7) == "7"

    def test_format_value_tensor(self):
        out = c.format_value(torch.ones(2, 3))
        assert out.startswith("Tensor(")
        assert "shape=[2, 3]" in out

    def test_format_value_scalars(self):
        assert c.format_value(None) == "None"
        assert c.format_value(True) == "True"
        assert c.format_value(5) == "5"
        assert c.format_value(1.5) == "1.5"

    def test_format_value_sequences(self):
        assert c.format_value([1, 2]).startswith("list([")
        assert c.format_value((1, 2, 3, 4)).startswith("tuple([")
        assert c.format_value([1, 2, 3, 4, 5]) == "list(len=5)"

    def test_format_value_other(self):
        obj = object()
        assert c.format_value(obj) == "object(...)"
        assert c.format_value("text") == "str(...)"

    def test_format_result(self):
        assert c.format_result((torch.zeros(1), torch.ones(1))).count("result[") == 2
        assert c.format_result(1).startswith("  result: ")


# ── Tag / label helpers ──


class TestTagsAndLabels:
    def test_get_module_class_name(self):
        assert c.get_module_class_name((nn.Linear(2, 2),)) == "Linear"
        assert c.get_module_class_name(()) is None
        assert c.get_module_class_name((5,)) is None

    def test_make_module_tag(self):
        assert c.make_module_tag() == ""
        c.push_module_context("Linear")
        try:
            assert c.make_module_tag() == "[module=0,1]"
        finally:
            c.pop_module_context()
        assert c.make_module_tag() == ""

    def test_make_op_tag_counts(self):
        assert c.make_op_tag("mm") == "[op=0,1]"
        assert c.make_op_tag("mm") == "[op=0,2]"

    def test_make_module_tag_from_ctx(self):
        assert c.make_module_tag_from_ctx("", "p") == ""
        assert c.make_module_tag_from_ctx("Linear", "") == "[Linear]"
        assert c.make_module_tag_from_ctx("Linear", "a", for_json=False) == "[Linear:a]"
        assert c.make_module_tag_from_ctx("Linear", "a", for_json=True) == "Linear:a"

    def test_make_label_variants(self):
        label = c.make_label(
            "mm",
            module_name="Linear",
            layer_path="model.layers.0",
            dispatch_keys=[("CUDA", "FlagGems", False)],
        )
        assert "module=Linear" in label
        assert "layer=model.layers.0" in label
        assert "dispatch_keys=[(CUDA, FlagGems)]" in label
        # module inferred from args
        assert "module=Linear" in c.make_label("mm", (nn.Linear(2, 2),))
        # no module and no context
        assert c.make_label("mm") == "mm (module=)"

    def test_record_seen(self):
        c.record_seen("mm", module_name="Linear")
        _, seen_ops = c._reset_per_step_counters()
        assert seen_ops == {"mm"}
        c.record_seen("add", (nn.Linear(2, 2),))
        seen_mods, seen_ops = c._reset_per_step_counters()
        assert "add" in seen_ops
        assert "Linear" in seen_mods


class TestTorchFuncTags:
    def test_nested_reuse_then_regenerate(self):
        outer = c.acquire_torch_func_tags("mm")
        inner = c.acquire_torch_func_tags("mm")
        assert outer == inner
        c.release_torch_func_tags()
        c.release_torch_func_tags()
        after = c.acquire_torch_func_tags("mm")
        assert after[2] == outer[2] + 1
        c.release_torch_func_tags()

    def test_release_without_acquire(self):
        c.release_torch_func_tags()  # depth 0 -> no raise


# ── YAML config parsing ──


class TestYamlParsing:
    def test_missing_file(self, tmp_path):
        assert c.parse_io_config_from_yaml(str(tmp_path / "nope.yaml")) == {}

    def test_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("a: [unclosed\n", encoding="utf-8")
        assert c.parse_io_config_from_yaml(str(p)) == {}

    def test_no_io_dump_section(self, tmp_path):
        p = tmp_path / "plain.yaml"
        p.write_text("other: 1\n", encoding="utf-8")
        assert c.parse_io_config_from_yaml(str(p)) == {}

    def test_io_dump_not_a_dict(self, tmp_path):
        p = tmp_path / "scalar.yaml"
        p.write_text("io_dump: 5\n", encoding="utf-8")
        assert c.parse_io_config_from_yaml(str(p)) == {}

    def test_full_section(self, tmp_path):
        p = tmp_path / "full.yaml"
        p.write_text(
            "io_dump:\n"
            "  dir: /tmp/out\n"
            "  ops: [rms_norm, silu_and_mul]\n"
            "  modules: Linear\n"
            "  layers: ['0', '1-2', 'model.layers.*.self_attn']\n"
            "  max_calls: 5\n"
            "  step_range: '1-3'\n"
            "  with_torch_funcs: true\n"
            "  ranks: [0, 2]\n"
            "  with_values: true\n"
            "  with_metas: true\n"
            "  with_print: true\n",
            encoding="utf-8",
        )
        parsed = c.parse_io_config_from_yaml(str(p))["io_dump"]
        assert parsed["dir"] == "/tmp/out"
        assert parsed["ops"] == {"rms_norm", "silu_and_mul"}
        assert parsed["modules"] == {"Linear"}
        assert {"model.layers.0", "model.layers.1", "model.layers.2"} <= parsed["layers"]
        assert "model.layers.*.self_attn" in parsed["layers"]
        assert parsed["max_calls"] == 5
        assert parsed["step_range"] == (1, 4)
        assert parsed["with_torch_funcs"] == (True, set())
        assert parsed["ranks"] == {0, 2}
        assert parsed["with_values"] is True
        assert parsed["with_metas"] is True
        assert parsed["with_print"] is True

    @pytest.mark.parametrize(
        "cfg,expected",
        [
            ({"step_range": None}, None),
            ({"step_range": 3}, (3, 4)),
            ({"step_range": [2, 5]}, (2, 6)),
            ({"step_range": [1, 2, 3]}, None),
            ({"step_range": "bad"}, None),
        ],
    )
    def test_parse_step_range_yaml(self, cfg, expected):
        assert c._parse_step_range_yaml(cfg) == expected

    def test_parse_dump_section_defaults(self):
        parsed = c._parse_dump_section({"dir": 5, "max_calls": "bad"})
        assert parsed["dir"] == "5"
        assert parsed["max_calls"] == 0
        assert parsed["with_values"] is False
        assert "with_metas" not in parsed
        assert "with_print" not in parsed
        assert "with_torch_funcs" not in parsed
        assert parsed["layers"] == set()

    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, set()),
            ([], set()),
            (["a", "", " b "], {"a", "b"}),
            (("a", "b"), {"a", "b"}),
            ("a, b ,,", {"a", "b"}),
            (5, set()),
        ],
    )
    def test_parse_string_list(self, value, expected):
        assert c._parse_string_list(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, (False, set())),
            (False, (False, set())),
            (True, (True, set())),
            (["matmul", ""], (True, {"matmul"})),
            ([], (False, set())),
            ("1", (True, set())),
            ("matmul,softmax", (True, {"matmul", "softmax"})),
            (5, (False, set())),
        ],
    )
    def test_parse_torch_funcs_yaml(self, value, expected):
        assert c._parse_torch_funcs_yaml(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, None),
            ([0, "1", "x"], {0, 1}),
            (["x"], None),
            (3, {3}),
            ("0,2", {0, 2}),
            ("all", None),
            (5.5, None),
        ],
    )
    def test_parse_ranks_yaml(self, value, expected):
        assert c._parse_ranks_yaml(value) == expected


def test_is_glob():
    assert c._is_glob("a*") is True
    assert c._is_glob("a?b") is True
    assert c._is_glob("abc") is False
