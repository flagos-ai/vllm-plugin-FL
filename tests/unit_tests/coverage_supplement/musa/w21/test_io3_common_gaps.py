# Copyright (c) 2026 BAAI. All rights reserved.

"""Residual branch coverage for ``vllm_fl.dispatch.io_common``.

Covers the helpers not reached by w2 (mode lifecycle) or w2b (rank, step,
dispatch table, formatting, YAML parsing): layer discovery, torch-func config
parsing, multi-variable layer env parsing, and the float-only tensor-stat
registry path.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch import io_common as c


@pytest.fixture(autouse=True)
def _restore_registry():
    saved = list(c._TENSOR_STAT_REGISTRY)
    c.reset_rank()
    c.reset_step()
    c.reset_exec_order()
    yield
    c._TENSOR_STAT_REGISTRY[:] = saved
    c.reset_rank()
    c.reset_step()
    c.reset_exec_order()


# 鈹€鈹€ list_model_layers 鈹€鈹€


class _Leaf:
    pass


class _Block:
    pass


class _FakeModel:
    def __init__(self, entries):
        self._entries = entries

    def named_modules(self):
        return iter(self._entries)


class TestListModelLayers:
    def _model(self):
        return _FakeModel(
            [
                ("", object()),
                ("block", _Block()),
                ("block.attn", _Leaf()),
                ("linear", _Leaf()),
            ]
        )

    def test_unlimited_depth(self, monkeypatch):
        logged = []
        monkeypatch.setattr(
            c,
            "_logger",
            SimpleNamespace(info=lambda *a, **k: logged.append(a)),
        )
        paths = c.list_model_layers(self._model())
        assert paths == [
            "block (_Block)",
            "block.attn (_Leaf)",
            "linear (_Leaf)",
        ]
        assert len(logged) == 1
        assert logged[0][1] == 3

    def test_max_depth_filters_and_skips_root(self, monkeypatch):
        logged = []
        monkeypatch.setattr(
            c,
            "_logger",
            SimpleNamespace(info=lambda *a, **k: logged.append(a)),
        )
        paths = c.list_model_layers(self._model(), max_depth=1)
        assert paths == ["block (_Block)", "linear (_Leaf)"]
        assert logged

    def test_max_depth_two(self):
        paths = c.list_model_layers(self._model(), max_depth=2)
        assert "block.attn (_Leaf)" in paths
        assert len(paths) == 3

    def test_root_only_does_not_log(self, monkeypatch):
        logged = []
        monkeypatch.setattr(
            c,
            "_logger",
            SimpleNamespace(info=lambda *a, **k: logged.append(a)),
        )
        assert c.list_model_layers(_FakeModel([("", object())])) == []
        assert logged == []


# 鈹€鈹€ parse_torch_funcs_config 鈹€鈹€


class TestParseTorchFuncsConfig:
    @pytest.mark.parametrize("value", ["", "   ", "0", " 0 "])
    def test_disabled_forms(self, value):
        assert c.parse_torch_funcs_config(value) == (False, set())

    @pytest.mark.parametrize("value", ["1", " 1 "])
    def test_enable_all(self, value):
        assert c.parse_torch_funcs_config(value) == (True, set())

    def test_explicit_list(self):
        assert c.parse_torch_funcs_config("matmul, softmax ,,") == (
            True,
            {"matmul", "softmax"},
        )

    def test_single_name(self):
        assert c.parse_torch_funcs_config("rms_norm") == (True, {"rms_norm"})


# 鈹€鈹€ parse_layers_env 鈹€鈹€


class TestParseLayersEnvMulti:
    def test_uses_first_non_empty(self, monkeypatch):
        monkeypatch.setenv("PRIMARY", "")
        monkeypatch.setenv("SECONDARY", " 1-2 , custom.path ")
        got = c.parse_layers_env("PRIMARY", "SECONDARY")
        assert got == {"model.layers.1", "model.layers.2", "custom.path"}

    def test_all_empty(self, monkeypatch):
        monkeypatch.delenv("PRIMARY", raising=False)
        monkeypatch.delenv("SECONDARY", raising=False)
        assert c.parse_layers_env("PRIMARY", "SECONDARY") == set()

    def test_none_set(self, monkeypatch):
        monkeypatch.delenv("ONLY", raising=False)
        assert c.parse_layers_env("ONLY") == set()


# 鈹€鈹€ tensor statistics registry 鈹€鈹€


class TestTensorStatFloatOnly:
    def test_float_only_false_receives_original_int_tensor(self):
        c.register_tensor_stat(
            "numel_raw", lambda t: int(t.numel()), float_only=False
        )
        stats = c.tensor_stats(torch.tensor([1, 2, 3], dtype=torch.int64))
        assert stats["numel_raw"] == 3
        assert "mean" not in stats  # int tensor skips float-only stats
        assert stats["min"] == 1 and stats["max"] == 3

    def test_float_only_true_casts_half_tensor(self):
        c.register_tensor_stat("dt_name", lambda t: str(t.dtype), float_only=True)
        stats = c.tensor_stats(torch.tensor([1.0, 2.0], dtype=torch.float16))
        assert stats["dt_name"] == "torch.float32"

    def test_replace_flips_float_only(self):
        c.register_tensor_stat("flip", lambda t: "orig", float_only=False)
        assert c.tensor_stats(torch.tensor([1.0, 2.0]))["flip"] == "orig"
        c.register_tensor_stat("flip", lambda t: "cast", float_only=True)
        got = c.tensor_stats(torch.tensor([1.0, 2.0]))
        assert got["flip"] == "cast"
        assert len([e for e in c._TENSOR_STAT_REGISTRY if e[0] == "flip"]) == 1

    def test_float_only_stat_cached_across_registry(self):
        calls = []

        def probe(t):
            calls.append(str(t.dtype))
            return 1

        c.register_tensor_stat("probe_a", probe, float_only=True)
        c.register_tensor_stat("probe_b", probe, float_only=True)
        c.tensor_stats(torch.tensor([1.0, 2.0, 3.0]))
        # The same cast float tensor is reused for both float-only stats.
        assert calls == ["torch.float32", "torch.float32"]


class TestFormatStatValueBoundary:
    def test_exactly_limit_has_no_ellipsis(self):
        out = c._format_stat_value([1.0, 2.0, 3.0, 4.0])
        assert out == "[1, 2, 3, 4]"
        assert "..." not in out

    def test_over_limit_truncates(self):
        out = c._format_stat_value([1, 2, 3, 4, 5])
        assert out.endswith(", ...]")


class TestRecordSeenExplicitModule:
    def test_explicit_module_overrides_args(self):
        c.record_seen("mm", module_name="Explicit")
        seen_mods, seen_ops = c._reset_per_step_counters()
        assert seen_mods == {"Explicit"}
        assert seen_ops == {"mm"}


class TestFormatValueSequenceBoundary:
    def test_four_items_inline(self):
        out = c.format_value([1, 2, 3, 4])
        assert out.startswith("list([")
        assert "len=" not in out

    def test_five_items_summary(self):
        assert c.format_value((1, 2, 3, 4, 5)) == "tuple(len=5)"


