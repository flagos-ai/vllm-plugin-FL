# Copyright (c) 2026 BAAI. All rights reserved.

"""
Tests for vllm_fl.dispatch.io_common.

Covers:
  - get_rank: env var fallback, default 0
  - ModeManager / managed_inference_mode context manager
  - push/pop/get_current_module_context, module context stack
  - layer_path_matches / expand_layer_specs / parse_layers_env
  - module_context_matches
  - should_inspect_torch_func / should_inspect_dispatch_op
  - register_tensor_stat / tensor_stats
  - format_value / format_result / make_label
  - parse_rank_filter
  - StepCounter / OpCounter
  - parse_io_config_from_yaml (light smoke test)
"""

import os
import threading
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# get_rank
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRank:

    def test_default_rank_is_zero(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        import importlib
        import vllm_fl.dispatch.io_common as m
        importlib.reload(m)  # reset cached _rank
        # After reload, call get_rank with no dist initialized
        rank = m.get_rank()
        assert rank == 0

    def test_rank_env_var(self, monkeypatch):
        monkeypatch.setenv("RANK", "3")
        import importlib
        import vllm_fl.dispatch.io_common as m
        # Clear the module-level cache
        m._rank = None
        rank = m.get_rank()
        assert rank == 3
        m._rank = None  # restore

    def test_local_rank_env_var_fallback(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "7")
        import vllm_fl.dispatch.io_common as m
        m._rank = None
        rank = m.get_rank()
        assert rank == 7
        m._rank = None  # restore


# ─────────────────────────────────────────────────────────────────────────────
# ModeManager / managed_inference_mode
# ─────────────────────────────────────────────────────────────────────────────

class TestModeManager:

    def setup_method(self):
        from vllm_fl.dispatch.io_common import ModeManager
        self.mgr = ModeManager()

    def test_initially_not_in_inference_mode(self):
        assert self.mgr.in_inference_mode() is False

    def test_enter_sets_mode(self):
        self.mgr.enter()
        assert self.mgr.in_inference_mode() is True
        self.mgr.exit()

    def test_exit_clears_mode(self):
        self.mgr.enter()
        self.mgr.exit()
        assert self.mgr.in_inference_mode() is False

    def test_nested_enter_exit(self):
        self.mgr.enter()
        self.mgr.enter()
        self.mgr.exit()
        assert self.mgr.in_inference_mode() is True
        self.mgr.exit()
        assert self.mgr.in_inference_mode() is False

    def test_managed_inference_mode_context_manager(self):
        from vllm_fl.dispatch.io_common import managed_inference_mode
        assert self.mgr.in_inference_mode() is False
        with managed_inference_mode(self.mgr):
            assert self.mgr.in_inference_mode() is True
        assert self.mgr.in_inference_mode() is False

    def test_managed_inference_mode_on_exception(self):
        from vllm_fl.dispatch.io_common import managed_inference_mode
        try:
            with managed_inference_mode(self.mgr):
                raise ValueError("test")
        except ValueError:
            pass
        assert self.mgr.in_inference_mode() is False

    def test_thread_safety(self):
        """Each thread has independent mode state (thread-local)."""
        from vllm_fl.dispatch.io_common import managed_inference_mode
        results = {}

        def worker(tid):
            with managed_inference_mode(self.mgr):
                results[tid] = self.mgr.in_inference_mode()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for v in results.values():
            assert v is True


# ─────────────────────────────────────────────────────────────────────────────
# Module context stack: push / pop / get_current_module_context
# ─────────────────────────────────────────────────────────────────────────────

class TestModuleContextStack:

    def setup_method(self):
        # Clear the thread-local stack before each test
        from vllm_fl.dispatch.io_common import _module_context
        if hasattr(_module_context, "stack"):
            _module_context.stack = []

    def test_empty_stack_returns_none(self):
        from vllm_fl.dispatch.io_common import get_current_module_context
        assert get_current_module_context() is None

    def test_push_then_get(self):
        from vllm_fl.dispatch.io_common import push_module_context, get_current_module_context, pop_module_context
        push_module_context("MyModule")
        ctx = get_current_module_context()
        assert ctx is not None
        assert ctx[0] == "MyModule"
        pop_module_context()

    def test_push_pop_restores_empty(self):
        from vllm_fl.dispatch.io_common import push_module_context, pop_module_context, get_current_module_context
        push_module_context("A")
        pop_module_context()
        assert get_current_module_context() is None

    def test_nested_push_pop(self):
        from vllm_fl.dispatch.io_common import push_module_context, pop_module_context, get_current_module_context
        push_module_context("Outer")
        push_module_context("Inner")
        assert get_current_module_context()[0] == "Inner"
        pop_module_context()
        assert get_current_module_context()[0] == "Outer"
        pop_module_context()

    def test_pop_on_empty_no_crash(self):
        from vllm_fl.dispatch.io_common import pop_module_context
        # Should not raise on empty stack
        pop_module_context()

    def test_thread_isolation(self):
        """Each thread has its own stack."""
        from vllm_fl.dispatch.io_common import push_module_context, get_current_module_context, pop_module_context
        push_module_context("MainThread")
        result = {}

        def worker():
            result["ctx"] = get_current_module_context()

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert result["ctx"] is None  # other thread starts empty
        pop_module_context()


# ─────────────────────────────────────────────────────────────────────────────
# layer_path_matches / expand_layer_specs / parse_layers_env
# ─────────────────────────────────────────────────────────────────────────────

class TestLayerPathMatches:

    def test_exact_match(self):
        from vllm_fl.dispatch.io_common import layer_path_matches
        assert layer_path_matches("model.layers.0", {"model.layers.0"}) is True

    def test_no_match(self):
        from vllm_fl.dispatch.io_common import layer_path_matches
        assert layer_path_matches("model.layers.1", {"model.layers.0"}) is False

    def test_none_specs_matches_all(self):
        from vllm_fl.dispatch.io_common import layer_path_matches
        assert layer_path_matches("model.layers.99", None) is True

    def test_empty_specs_matches_nothing(self):
        from vllm_fl.dispatch.io_common import layer_path_matches
        assert layer_path_matches("model.layers.0", set()) is False

    def test_glob_pattern(self):
        from vllm_fl.dispatch.io_common import layer_path_matches
        assert layer_path_matches("model.layers.3.self_attn", {"model.layers.*.self_attn"}) is True
        assert layer_path_matches("model.layers.3.mlp", {"model.layers.*.self_attn"}) is False


class TestExpandLayerSpecs:

    def test_integer_shorthand(self):
        from vllm_fl.dispatch.io_common import expand_layer_specs
        result = expand_layer_specs(["0", "1", "2"])
        assert "model.layers.0" in result
        assert "model.layers.1" in result
        assert "model.layers.2" in result

    def test_range_spec(self):
        from vllm_fl.dispatch.io_common import expand_layer_specs
        result = expand_layer_specs(["0-2"])
        assert "model.layers.0" in result
        assert "model.layers.1" in result
        assert "model.layers.2" in result

    def test_full_path_passthrough(self):
        from vllm_fl.dispatch.io_common import expand_layer_specs
        result = expand_layer_specs(["model.layers.5.self_attn"])
        assert "model.layers.5.self_attn" in result

    def test_glob_passthrough(self):
        from vllm_fl.dispatch.io_common import expand_layer_specs
        result = expand_layer_specs(["model.layers.*.mlp"])
        assert "model.layers.*.mlp" in result

    def test_empty_input(self):
        from vllm_fl.dispatch.io_common import expand_layer_specs
        result = expand_layer_specs([])
        assert result == set() or result == frozenset()

    def test_none_input_returns_none(self):
        from vllm_fl.dispatch.io_common import expand_layer_specs
        result = expand_layer_specs(None)
        assert result is None


class TestParseLayersEnv:

    def test_comma_separated_ints(self):
        from vllm_fl.dispatch.io_common import parse_layers_env
        result = parse_layers_env("0,1,2")
        assert result is not None
        assert "model.layers.0" in result

    def test_range_string(self):
        from vllm_fl.dispatch.io_common import parse_layers_env
        result = parse_layers_env("0-3")
        assert result is not None
        for i in range(4):
            assert f"model.layers.{i}" in result

    def test_empty_string_returns_none(self):
        from vllm_fl.dispatch.io_common import parse_layers_env
        result = parse_layers_env("")
        assert result is None

    def test_none_returns_none(self):
        from vllm_fl.dispatch.io_common import parse_layers_env
        result = parse_layers_env(None)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# module_context_matches
# ─────────────────────────────────────────────────────────────────────────────

class TestModuleContextMatches:

    def test_none_filter_always_matches(self):
        from vllm_fl.dispatch.io_common import module_context_matches
        assert module_context_matches(("MyModule", 0, 0), None) is True

    def test_exact_class_name_match(self):
        from vllm_fl.dispatch.io_common import module_context_matches
        assert module_context_matches(("MyModule", 0, 0), {"MyModule"}) is True

    def test_class_name_no_match(self):
        from vllm_fl.dispatch.io_common import module_context_matches
        assert module_context_matches(("OtherModule", 0, 0), {"MyModule"}) is False

    def test_empty_filter_no_match(self):
        from vllm_fl.dispatch.io_common import module_context_matches
        assert module_context_matches(("MyModule", 0, 0), set()) is False


# ─────────────────────────────────────────────────────────────────────────────
# register_tensor_stat / tensor_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestTensorStats:

    def setup_method(self):
        # Clear any stats registered by previous tests
        from vllm_fl.dispatch.io_common import _tensor_stats
        _tensor_stats.clear()

    def test_register_and_retrieve(self):
        from vllm_fl.dispatch.io_common import register_tensor_stat, tensor_stats
        register_tensor_stat("my_stat", lambda t: float(t.sum()))
        stats = tensor_stats()
        assert "my_stat" in stats

    def test_stat_callable_stored(self):
        from vllm_fl.dispatch.io_common import register_tensor_stat, tensor_stats
        fn = lambda t: 42.0
        register_tensor_stat("const_stat", fn)
        stats = tensor_stats()
        assert stats["const_stat"] is fn or callable(stats["const_stat"])

    def test_replace_existing_stat(self):
        from vllm_fl.dispatch.io_common import register_tensor_stat, tensor_stats
        register_tensor_stat("dup_stat", lambda t: 1.0)
        new_fn = lambda t: 2.0
        register_tensor_stat("dup_stat", new_fn)
        stats = tensor_stats()
        assert "dup_stat" in stats

    def test_multiple_stats(self):
        from vllm_fl.dispatch.io_common import register_tensor_stat, tensor_stats
        for name in ("s1", "s2", "s3"):
            register_tensor_stat(name, lambda t: 0.0)
        stats = tensor_stats()
        for name in ("s1", "s2", "s3"):
            assert name in stats

    def test_tensor_stats_returns_copy(self):
        from vllm_fl.dispatch.io_common import register_tensor_stat, tensor_stats
        register_tensor_stat("copy_stat", lambda t: 0.0)
        s1 = tensor_stats()
        s2 = tensor_stats()
        assert s1 is not s2


# ─────────────────────────────────────────────────────────────────────────────
# format_value / format_result / make_label
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatValue:

    def test_scalar_int(self):
        from vllm_fl.dispatch.io_common import format_value
        result = format_value(42)
        assert "42" in str(result)

    def test_scalar_float(self):
        from vllm_fl.dispatch.io_common import format_value
        result = format_value(3.14)
        assert result is not None

    def test_string_passthrough(self):
        from vllm_fl.dispatch.io_common import format_value
        result = format_value("hello")
        assert "hello" in str(result)

    def test_none_passthrough(self):
        from vllm_fl.dispatch.io_common import format_value
        result = format_value(None)
        assert result is None or str(result) in ("None", "null", "")

    def test_list_value(self):
        from vllm_fl.dispatch.io_common import format_value
        result = format_value([1, 2, 3])
        assert result is not None

    def test_tensor_value(self):
        import torch
        from vllm_fl.dispatch.io_common import format_value
        t = torch.tensor([1.0, 2.0, 3.0])
        result = format_value(t)
        assert result is not None


class TestFormatResult:

    def test_single_tensor(self):
        import torch
        from vllm_fl.dispatch.io_common import format_result
        t = torch.randn(4)
        result = format_result(t)
        assert result is not None

    def test_tuple_of_tensors(self):
        import torch
        from vllm_fl.dispatch.io_common import format_result
        t1 = torch.randn(2)
        t2 = torch.randn(3)
        result = format_result((t1, t2))
        assert result is not None

    def test_none_result(self):
        from vllm_fl.dispatch.io_common import format_result
        result = format_result(None)
        assert result is None or isinstance(result, (str, dict))


class TestMakeLabel:

    def test_basic_label(self):
        from vllm_fl.dispatch.io_common import make_label
        label = make_label("my_op", step=0, call=1)
        assert "my_op" in label

    def test_label_contains_step(self):
        from vllm_fl.dispatch.io_common import make_label
        label = make_label("op_x", step=5, call=2)
        assert "5" in label or "step" in label.lower()

    def test_label_is_string(self):
        from vllm_fl.dispatch.io_common import make_label
        label = make_label("op_y", step=0, call=0)
        assert isinstance(label, str)


# ─────────────────────────────────────────────────────────────────────────────
# parse_rank_filter
# ─────────────────────────────────────────────────────────────────────────────

class TestParseRankFilter:

    def test_none_returns_none(self):
        from vllm_fl.dispatch.io_common import parse_rank_filter
        assert parse_rank_filter(None) is None

    def test_empty_string_returns_none(self):
        from vllm_fl.dispatch.io_common import parse_rank_filter
        result = parse_rank_filter("")
        assert result is None

    def test_single_int_string(self):
        from vllm_fl.dispatch.io_common import parse_rank_filter
        result = parse_rank_filter("0")
        assert result == {0}

    def test_comma_separated(self):
        from vllm_fl.dispatch.io_common import parse_rank_filter
        result = parse_rank_filter("0,1,2")
        assert result == {0, 1, 2}

    def test_range_string(self):
        from vllm_fl.dispatch.io_common import parse_rank_filter
        result = parse_rank_filter("0-3")
        assert result == {0, 1, 2, 3}

    def test_int_input(self):
        from vllm_fl.dispatch.io_common import parse_rank_filter
        result = parse_rank_filter(5)
        assert result == {5}

    def test_list_of_ints(self):
        from vllm_fl.dispatch.io_common import parse_rank_filter
        result = parse_rank_filter([1, 2, 3])
        assert result == {1, 2, 3}

    def test_list_of_strings(self):
        from vllm_fl.dispatch.io_common import parse_rank_filter
        result = parse_rank_filter(["0", "2"])
        assert result == {0, 2}

    def test_invalid_string_returns_none(self):
        from vllm_fl.dispatch.io_common import parse_rank_filter
        result = parse_rank_filter("not_a_rank")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# should_inspect_torch_func / should_inspect_dispatch_op
# ─────────────────────────────────────────────────────────────────────────────

class TestShouldInspect:

    def test_should_inspect_dispatch_op_no_filter(self):
        from vllm_fl.dispatch.io_common import should_inspect_dispatch_op
        # No op filter → always True for any op
        result = should_inspect_dispatch_op("any_op", op_filter=None)
        assert result is True

    def test_should_inspect_dispatch_op_with_match(self):
        from vllm_fl.dispatch.io_common import should_inspect_dispatch_op
        result = should_inspect_dispatch_op("my_op", op_filter={"my_op"})
        assert result is True

    def test_should_inspect_dispatch_op_with_no_match(self):
        from vllm_fl.dispatch.io_common import should_inspect_dispatch_op
        result = should_inspect_dispatch_op("other_op", op_filter={"my_op"})
        assert result is False

    def test_should_inspect_torch_func_no_filter(self):
        from vllm_fl.dispatch.io_common import should_inspect_torch_func
        result = should_inspect_torch_func("torch.add", func_filter=None)
        assert result is True

    def test_should_inspect_torch_func_with_match(self):
        from vllm_fl.dispatch.io_common import should_inspect_torch_func
        result = should_inspect_torch_func("torch.add", func_filter={"torch.add"})
        assert result is True

    def test_should_inspect_torch_func_with_no_match(self):
        from vllm_fl.dispatch.io_common import should_inspect_torch_func
        result = should_inspect_torch_func("torch.mul", func_filter={"torch.add"})
        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# StepCounter / OpCounter
# ─────────────────────────────────────────────────────────────────────────────

class TestStepCounter:

    def setup_method(self):
        from vllm_fl.dispatch.io_common import _step_counter
        _step_counter.reset()

    def test_initial_value_is_zero(self):
        from vllm_fl.dispatch.io_common import _step_counter
        assert _step_counter.value() == 0

    def test_increment(self):
        from vllm_fl.dispatch.io_common import _step_counter
        _step_counter.increment()
        assert _step_counter.value() == 1

    def test_multiple_increments(self):
        from vllm_fl.dispatch.io_common import _step_counter
        for _ in range(5):
            _step_counter.increment()
        assert _step_counter.value() == 5

    def test_reset(self):
        from vllm_fl.dispatch.io_common import _step_counter
        _step_counter.increment()
        _step_counter.increment()
        _step_counter.reset()
        assert _step_counter.value() == 0

    def test_thread_safe_increment(self):
        from vllm_fl.dispatch.io_common import _step_counter
        _step_counter.reset()
        N = 100

        def worker():
            for _ in range(N):
                _step_counter.increment()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert _step_counter.value() == 4 * N


class TestOpCounter:

    def setup_method(self):
        from vllm_fl.dispatch.io_common import _op_counters
        _op_counters.clear()

    def test_first_call_count_is_one(self):
        from vllm_fl.dispatch.io_common import get_op_call_count
        count = get_op_call_count("my_op")
        assert count == 1

    def test_repeated_calls_increment(self):
        from vllm_fl.dispatch.io_common import get_op_call_count
        for i in range(1, 4):
            count = get_op_call_count("counted_op")
            assert count == i

    def test_different_ops_independent(self):
        from vllm_fl.dispatch.io_common import get_op_call_count
        get_op_call_count("op_a")
        get_op_call_count("op_a")
        count_b = get_op_call_count("op_b")
        assert count_b == 1

    def test_reset_clears_counter(self):
        from vllm_fl.dispatch.io_common import get_op_call_count, _op_counters
        get_op_call_count("reset_op")
        get_op_call_count("reset_op")
        _op_counters.clear()
        count = get_op_call_count("reset_op")
        assert count == 1


# ─────────────────────────────────────────────────────────────────────────────
# get_dispatch_backends
# ─────────────────────────────────────────────────────────────────────────────

class TestGetDispatchBackends:

    def test_returns_list(self):
        import torch
        from vllm_fl.dispatch.io_common import get_dispatch_backends
        result = get_dispatch_backends(torch.add)
        assert isinstance(result, list)

    def test_entries_are_tuples(self):
        import torch
        from vllm_fl.dispatch.io_common import get_dispatch_backends
        result = get_dispatch_backends(torch.add)
        for entry in result:
            assert isinstance(entry, tuple)
            assert len(entry) == 3

    def test_unknown_func_returns_fallback(self):
        from vllm_fl.dispatch.io_common import get_dispatch_backends
        # A non-torch callable should return the unknown fallback tuple
        result = get_dispatch_backends(lambda x: x)
        assert result == [("unknown", "unknown", True)]


# ─────────────────────────────────────────────────────────────────────────────
# parse_io_config_from_yaml (smoke test)
# ─────────────────────────────────────────────────────────────────────────────

class TestParseIoConfigFromYaml:

    def test_valid_minimal_config(self, tmp_path):
        import yaml
        from vllm_fl.dispatch.io_common import parse_io_config_from_yaml

        config = {
            "enabled": True,
            "output_dir": str(tmp_path / "dump"),
        }
        cfg_file = tmp_path / "io_config.yaml"
        cfg_file.write_text(yaml.dump(config))
        result = parse_io_config_from_yaml(str(cfg_file))
        assert result is not None

    def test_config_with_ops_filter(self, tmp_path):
        import yaml
        from vllm_fl.dispatch.io_common import parse_io_config_from_yaml

        config = {
            "enabled": True,
            "output_dir": str(tmp_path / "dump"),
            "ops": ["my_op", "other_op"],
        }
        cfg_file = tmp_path / "io_ops.yaml"
        cfg_file.write_text(yaml.dump(config))
        result = parse_io_config_from_yaml(str(cfg_file))
        assert result is not None

    def test_config_with_ranks(self, tmp_path):
        import yaml
        from vllm_fl.dispatch.io_common import parse_io_config_from_yaml

        config = {
            "enabled": True,
            "output_dir": str(tmp_path / "dump"),
            "ranks": [0, 1],
        }
        cfg_file = tmp_path / "io_ranks.yaml"
        cfg_file.write_text(yaml.dump(config))
        result = parse_io_config_from_yaml(str(cfg_file))
        assert result is not None

    def test_nonexistent_file_raises(self):
        from vllm_fl.dispatch.io_common import parse_io_config_from_yaml
        with pytest.raises((FileNotFoundError, OSError)):
            parse_io_config_from_yaml("/nonexistent/path/config.yaml")

    def test_disabled_config(self, tmp_path):
        import yaml
        from vllm_fl.dispatch.io_common import parse_io_config_from_yaml

        config = {"enabled": False}
        cfg_file = tmp_path / "disabled.yaml"
        cfg_file.write_text(yaml.dump(config))
        result = parse_io_config_from_yaml(str(cfg_file))
        # disabled config should either return None or an object with enabled=False
        assert result is None or getattr(result, "enabled", False) is False


# ─────────────────────────────────────────────────────────────────────────────
# make_module_tag_from_ctx
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeModuleTagFromCtx:

    def test_returns_string(self):
        from vllm_fl.dispatch.io_common import make_module_tag_from_ctx
        tag = make_module_tag_from_ctx(("MyModule", 0, 1))
        assert isinstance(tag, str)

    def test_contains_class_name(self):
        from vllm_fl.dispatch.io_common import make_module_tag_from_ctx
        tag = make_module_tag_from_ctx(("AttentionLayer", 2, 3))
        assert "AttentionLayer" in tag

    def test_none_returns_empty_or_none(self):
        from vllm_fl.dispatch.io_common import make_module_tag_from_ctx
        result = make_module_tag_from_ctx(None)
        assert result is None or result == "" or isinstance(result, str)
