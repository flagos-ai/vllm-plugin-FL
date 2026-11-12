# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Supplementary coverage tests for ``vllm_fl/dispatch/io_common.py`` (2/2).

D3 tranche of the flagcicd coverage campaign. Every test asserts an
independently computed behavior value. This file covers ATen op name
parsing, dispatch-table backend inference, torch-function config
parsing, io_dump YAML section parsing, execution-order/tag pairing and
per-step module counters.

Local collection note: importing ``vllm_fl`` pulls in ``flag_gems``
which needs triton (no Windows wheel). ``_flaggems_shim`` installs an
inert placeholder only when the real package cannot import.
"""

from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)
from ._dumper_helpers import restore_io_common_globals  # noqa: F401

import pytest
import torch

from vllm_fl.dispatch import io_common
from vllm_fl.dispatch.io_common import (
    _parse_dump_section,
    _parse_ranks_yaml,
    _parse_step_range_yaml,
    _parse_string_list,
    _parse_torch_funcs_yaml,
    acquire_torch_func_tags,
    advance_step,
    get_dispatch_keys,
    get_dispatch_op_name,
    get_dispatch_op_namespace,
    get_exec_order,
    get_step,
    get_current_module_counter,
    next_exec_order,
    parse_io_config_from_yaml,
    parse_torch_funcs_config,
    register_tensor_stat,
    release_torch_func_tags,
    reset_exec_order,
    reset_step,
    set_io_active,
    should_inspect_torch_func,
    tensor_stats,
)

pytestmark = pytest.mark.usefixtures("restore_io_common_globals")


class TestDispatchKeyHelpers:
    """ATen name parsing and dispatch-table backend inference."""

    def test_op_name_and_namespace(self):
        """OpOverload name is split into short name and namespace."""
        mm = torch.ops.aten.mm.default
        assert get_dispatch_op_name(mm) == "mm"
        assert get_dispatch_op_namespace(mm) == "aten"
        assert get_dispatch_op_name("plain-string") == "plain-string"
        assert get_dispatch_op_namespace("x") == "aten"  # default namespace

    def test_get_dispatch_keys_reports_cpu_kernel(self):
        """aten.mm on CPU reports a CPU dispatch key with a known backend."""
        mm = torch.ops.aten.mm.default
        keys = get_dispatch_keys(mm)
        cpu_entries = [k for k in keys if k[0] == "CPU"]
        assert cpu_entries and cpu_entries[0][2] is False  # dedicated kernel
        # Second call exercises the cache path (same result object).
        assert get_dispatch_keys(mm) is keys

    def test_backend_inference_from_registration_paths(self):
        """Registration-path pattern maps to backend labels."""
        inf = io_common._infer_backend_from_path
        assert inf("/pytorch/build/aten/src/ATen/RegisterCUDA_0.cpp:16060") == "CUDA"
        assert inf("C:/b/b/aten/src/ATen/RegisterCPU_0.cpp:324") == "CPU"
        assert inf("/site-packages/flag_gems/__init__.py:20") == "FlagGems"
        assert inf("/site-packages/triton/ops/matmul.py:42") == "Triton"
        assert inf("/torch/_meta_registrations.py:50") == "Meta"
        assert inf("/nowhere/special.py:1") == "unknown"


class TestTorchFuncConfig:
    """Torch-function config parsing and inspection gating."""

    def test_parse_torch_funcs_config(self):
        """Empty/0 disable, 1 enables all, lists parse to a filter set."""
        assert parse_torch_funcs_config("") == (False, set())
        assert parse_torch_funcs_config("0") == (False, set())
        assert parse_torch_funcs_config("1") == (True, set())
        assert parse_torch_funcs_config("matmul, softmax") == (
            True, {"matmul", "softmax"})

    def test_should_inspect_torch_func_gating(self):
        """Filter set membership, dunder/trivial skips and op_filter apply."""
        gate = should_inspect_torch_func
        assert gate("matmul", True, set(), True, set()) is True
        assert gate("matmul", False, set(), True, set()) is False
        assert gate("matmul", True, {"softmax"}, True, set()) is False
        assert gate("softmax", True, {"softmax"}, True, set()) is True
        assert gate("_private", True, set(), True, set()) is False
        assert gate("size", True, set(), True, set()) is False  # trivial
        assert gate("cat", True, set(), False, {"add"}) is False
        assert gate("add", True, set(), False, {"add"}) is True

    def test_parse_torch_funcs_yaml_variants(self):
        """YAML torch_funcs accepts bool, list, string forms."""
        assert _parse_torch_funcs_yaml(False) == (False, set())
        assert _parse_torch_funcs_yaml(True) == (True, set())
        assert _parse_torch_funcs_yaml(None) == (False, set())
        assert _parse_torch_funcs_yaml(["matmul", "softmax"]) == (
            True, {"matmul", "softmax"})
        assert _parse_torch_funcs_yaml([]) == (False, set())
        assert _parse_torch_funcs_yaml("matmul") == (True, {"matmul"})


class TestYamlParsing:
    """YAML section parsers against files on disk (tmp_path)."""

    def _write(self, path, text):
        path.write_text(text, encoding="utf-8")

    def test_parse_io_config_full_dump_section(self, tmp_path):
        """A full io_dump section parses every documented field."""
        cfg_file = tmp_path / "full.yaml"
        self._write(cfg_file, (
            "io_dump:\n"
            "  dir: /tmp/x\n"
            "  with_print: true\n"
            "  ops: [rms_norm, silu_and_mul]\n"
            "  modules: [Linear]\n"
            "  layers: [\"0-1\", model.layers.5]\n"
            "  max_calls: 100\n"
            "  step_range: \"5-15\"\n"
            "  with_torch_funcs: [matmul]\n"
            "  with_metas: true\n"
            "  with_values: true\n"
            "  ranks: [0, 2]\n"
        ))
        cfg = parse_io_config_from_yaml(str(cfg_file))
        dump = cfg["io_dump"]
        assert dump["dir"] == "/tmp/x"
        assert dump["ops"] == {"rms_norm", "silu_and_mul"}
        assert dump["modules"] == {"Linear"}
        assert dump["layers"] == {"model.layers.0", "model.layers.1",
                                  "model.layers.5"}
        assert dump["max_calls"] == 100
        assert dump["step_range"] == (5, 16)
        assert dump["with_torch_funcs"] == (True, {"matmul"})
        assert dump["ranks"] == {0, 2}
        assert dump["with_values"] is True
        assert dump["with_metas"] is True
        assert dump["with_print"] is True

    def test_parse_io_config_missing_and_empty(self, tmp_path):
        """Missing file, non-dict io_dump and empty files yield {}."""
        assert parse_io_config_from_yaml(str(tmp_path / "nope.yaml")) == {}
        blank = tmp_path / "blank.yaml"
        self._write(blank, "")
        assert parse_io_config_from_yaml(str(blank)) == {}
        other = tmp_path / "other.yaml"
        self._write(other, "unrelated: 1\n")
        assert parse_io_config_from_yaml(str(other)) == {}

    def test_parse_dump_section_defaults_and_invalid(self):
        """Absent fields default; invalid max_calls falls back to 0."""
        dump = _parse_dump_section({})
        assert dump["dir"] == ""
        assert dump["ops"] == set()
        assert dump["modules"] == set()
        assert dump["layers"] == set()
        assert dump["max_calls"] == 0
        assert dump["step_range"] is None
        assert dump["ranks"] is None
        assert dump["with_values"] is False
        assert "with_metas" not in dump
        assert "with_print" not in dump
        assert "with_torch_funcs" not in dump
        bad = _parse_dump_section({"max_calls": "not-a-number", "ops": "a,b",
                                   "modules": ["X"], "layers": "7"})
        assert bad["max_calls"] == 0
        assert bad["ops"] == {"a", "b"}
        assert bad["modules"] == {"X"}
        assert bad["layers"] == {"model.layers.7"}

    def test_step_range_yaml_accepts_int_and_list(self):
        """YAML accepts int (5→(5,6)), [s,e] lists, and dash strings."""
        assert _parse_step_range_yaml({"step_range": 5}) == (5, 6)
        assert _parse_step_range_yaml({"step_range": [1, 3]}) == (1, 4)
        assert _parse_step_range_yaml({"step_range": "1-3"}) == (1, 4)
        assert _parse_step_range_yaml({"step_range": [1, 2, 3]}) is None
        assert _parse_step_range_yaml({}) is None

    def test_parse_string_list_variants(self):
        """Lists, comma strings, None and scalars parse to string sets."""
        assert _parse_string_list(["a", "b "]) == {"a", "b"}
        assert _parse_string_list("a,b") == {"a", "b"}
        assert _parse_string_list(None) == set()
        assert _parse_string_list("") == set()
        assert _parse_string_list(7) == set()

    def test_parse_ranks_yaml_variants(self):
        """ranks accepts None, int, list-with-junk and string forms."""
        assert _parse_ranks_yaml(None) is None
        assert _parse_ranks_yaml("all") is None
        assert _parse_ranks_yaml(0) == {0}
        assert _parse_ranks_yaml([0, 2]) == {0, 2}
        assert _parse_ranks_yaml([0, "x", 4.5]) == {0, 4}  # int() coercion
        assert _parse_ranks_yaml("0,2") == {0, 2}
        assert _parse_ranks_yaml(["junk"]) is None
        assert _parse_ranks_yaml(3.5) is None  # float not accepted


def test_managed_inference_mode_downgrades_when_io_active():
    """With io active the wrapper runs under no_grad, not inference_mode."""
    from vllm_fl.dispatch.io_common import managed_inference_mode

    @managed_inference_mode()
    def probe():
        x = torch.zeros(2, requires_grad=True)
        return torch.is_inference(x), torch.is_grad_enabled()

    set_io_active(True)
    try:
        is_inference, grad_on = probe()
    finally:
        set_io_active(False)
    # no_grad: not inference tensors, autograd recording disabled.
    assert is_inference is False
    assert grad_on is False

    # Inactive: plain inference_mode semantics.
    is_inference, grad_on = probe()
    assert is_inference is True
    assert grad_on is False


class TestTensorStats:
    """Tensor statistics registry and computation."""

    def test_builtin_stats_values(self):
        """min/max/mean/std computed against closed-form references."""
        t = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
        stats = tensor_stats(t)
        assert stats["shape"] == [2, 2]
        assert stats["dtype"] == "torch.float32"
        assert stats["device"] == "cpu"
        assert stats["min"] == 1.0
        assert stats["max"] == 6.0
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["std"] == pytest.approx(
            torch.tensor([1.0, 2.0, 3.0, 6.0]).std().item())

    def test_integer_tensor_skips_float_only_stats(self):
        """Integer tensors get min/max but not mean/std (float_only=True)."""
        stats = tensor_stats(torch.tensor([[1, 4]]))
        assert stats["min"] == 1
        assert stats["max"] == 4
        assert "mean" not in stats
        assert "std" not in stats

    def test_empty_and_complex_tensors_return_meta_only(self):
        """Empty or complex tensors short-circuit to metadata only."""
        empty = tensor_stats(torch.zeros(0, 2))
        assert set(empty) == {"shape", "dtype", "device"}
        assert empty["shape"] == [0, 2]
        cplx = tensor_stats(torch.complex(torch.ones(2), torch.ones(2)))
        assert set(cplx) == {"shape", "dtype", "device"}
        assert cplx["dtype"] == "torch.complex64"

    def test_register_replace_and_rejection_paths(self):
        """register_tensor_stat replaces by name, appends new, rejects empty."""
        def _median(t):
            return t.flatten().median().item()

        register_tensor_stat("median", _median, float_only=False)
        stats = tensor_stats(torch.tensor([1.0, 5.0, 100.0]))
        assert stats["median"] == 5.0
        # Replace by same name with a different implementation.
        register_tensor_stat("median", lambda t: -1.0, float_only=False)
        assert tensor_stats(torch.tensor([1.0, 5.0, 100.0]))["median"] == -1.0
        # Empty name is rejected.
        with pytest.raises(ValueError):
            register_tensor_stat("  ", lambda t: 0.0)
        # float_only stat receives the float()-converted tensor.
        register_tensor_stat(
            "saw_float",
            lambda t: (t.dtype == torch.float32, t.sum().item()),
            float_only=False,
        )
        saw = tensor_stats(torch.tensor([2, 2]))["saw_float"]
        assert saw == (False, 4.0)  # float_only=False: original int tensor


class TestExecOrderAndTorchFuncTags:
    """Execution order counter and torch-func tag pairing."""

    def test_exec_order_increments_and_resets(self):
        """next_exec_order is monotonic; reset_exec_order restarts at 1."""
        reset_exec_order()
        assert get_exec_order() == 0
        assert [next_exec_order() for _ in range(3)] == [1, 2, 3]
        assert get_exec_order() == 3
        reset_exec_order()
        assert get_exec_order() == 0
        assert next_exec_order() == 1

    def test_acquire_release_torch_func_tags_nesting(self):
        """Outermost acquisition allocates; nested calls reuse the cache."""
        # Clean slate: drain any leaked depth from earlier tests.
        while getattr(io_common._tf_tags, "depth", 0) > 0:
            release_torch_func_tags()
        reset_exec_order()
        m1, o1, e1 = acquire_torch_func_tags("matmul")
        assert m1 == "" and o1.startswith("[op=") and e1 == 1
        m2, o2, e2 = acquire_torch_func_tags("matmul")  # nested: reuses
        assert (m2, o2, e2) == (m1, o1, e1)
        release_torch_func_tags()  # depth 2 -> 1: cache retained
        m3, o3, e3 = acquire_torch_func_tags("matmul")
        assert (m3, o3, e3) == (m1, o1, e1)  # still reuses
        release_torch_func_tags()  # depth 2 -> 1: cache retained
        release_torch_func_tags()  # depth 1 -> 0: cache cleared
        m4, o4, e4 = acquire_torch_func_tags("matmul")
        assert e4 == 2  # fresh allocation, order advanced by one
        assert o4 != o1
        release_torch_func_tags()
        reset_exec_order()

    def test_step_advances_and_fires_callbacks(self):
        """advance_step bumps the counter, resets exec order, fires callbacks."""
        reset_step()
        reset_exec_order()
        seen = []

        def _cb(step, mods, ops):
            seen.append((step, mods, ops))

        io_common.register_step_callback(_cb)
        try:
            assert get_step() == 0
            next_exec_order()
            assert advance_step() == 1
            assert get_step() == 1
            assert get_exec_order() == 0  # reset by advance
            assert seen == [(0, set(), set())]
        finally:
            io_common.unregister_step_callback(_cb)
        reset_step()

    def test_failing_step_callback_is_isolated(self):
        """A raising callback is logged and does not stop the advance."""
        reset_step()

        def _boom(step, mods, ops):
            raise RuntimeError("callback boom")

        io_common.register_step_callback(_boom)
        try:
            assert advance_step() == 1
        finally:
            io_common.unregister_step_callback(_boom)
        reset_step()


class TestModuleCounterHelpers:
    """Per-step module/op counters exposed through push/pop context."""

    def test_next_module_counter_indexes_and_counts(self):
        """First sighting assigns index 0; repeats bump only the count."""
        reset_step()
        io_common.push_module_context("Linear")
        assert get_current_module_counter() == (0, 1)
        io_common.pop_module_context()
        io_common.push_module_context("Linear")
        assert get_current_module_counter() == (0, 2)
        io_common.pop_module_context()
        io_common.push_module_context("ReLU")
        assert get_current_module_counter() == (1, 1)
        io_common.pop_module_context()
        reset_step()
