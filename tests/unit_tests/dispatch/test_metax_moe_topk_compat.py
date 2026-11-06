# Copyright (c) 2026 BAAI. All rights reserved.

"""Restore top-k bridges after vLLM finishes a partially imported module."""

import ast
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

_PATCHES = (
    Path(__file__).resolve().parents[3]
    / "vllm_fl/dispatch/backends/vendor/metax/patches"
)


def _load(monkeypatch, name, path, *, package=False):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
        submodule_search_locations=[str(path.parent)] if package else None,
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _partially_imported_ops(monkeypatch):
    calls = []

    def native_six(weights, indices, token_indices, logits, renormalize, bias):
        calls.append((weights, indices, token_indices, logits, renormalize, bias))

    torch = types.ModuleType("torch")
    torch.ops = types.SimpleNamespace(
        _moe_C=types.SimpleNamespace(topk_softmax=native_six, topk_sigmoid=native_six)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    for name in ("vllm", "mcoplib"):
        package = types.ModuleType(name)
        package.__path__ = []
        package.__spec__ = importlib.util.spec_from_loader(
            name, loader=None, is_package=True
        )
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(
        sys.modules, "mcoplib._moe_C", types.ModuleType("mcoplib._moe_C")
    )
    ops = types.ModuleType("vllm._custom_ops")
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    compat = _load(monkeypatch, "metax_topk_test", _PATCHES / "moe_topk_compat.py")
    assert getattr(ops.topk_softmax, compat._PATCH_FLAG)
    return ops, compat, calls


def _finish_native_import(ops):
    # vLLM's remaining module body overwrites both functions after returning
    # from current_platform.import_kernels(), but keeps the early module flag.
    def upstream_topk_softmax(*args):
        raise AssertionError("the seven-argument vLLM route must be bridged")

    ops.topk_softmax = upstream_topk_softmax
    ops.topk_sigmoid = upstream_topk_softmax


def test_reapply_repairs_live_functions_despite_stale_module_flag(monkeypatch):
    ops, compat, calls = _partially_imported_ops(monkeypatch)
    _finish_native_import(ops)
    assert getattr(ops, compat._PATCH_FLAG) is True
    assert compat.apply() is True
    args = tuple(object() for _ in range(6))
    ops.topk_softmax(*args, is_padding=object())
    assert calls == [args]
    live_softmax, live_sigmoid = ops.topk_softmax, ops.topk_sigmoid
    assert compat.apply() is True
    assert (ops.topk_softmax, ops.topk_sigmoid) == (live_softmax, live_sigmoid)

    class Weights:
        def mul_(self, scale):
            self.scale = scale

    weights = Weights()
    ops.topk_sigmoid(weights, *args[1:], routed_scaling_factor=2.5)
    assert calls[-1] == (weights, *args[1:])
    assert weights.scale == 2.5


def test_stable_native_operator_survives_custom_ops_overwrite(monkeypatch):
    ops, compat, calls = _partially_imported_ops(monkeypatch)
    stable = compat.get_topk_operator("softmax")
    _finish_native_import(ops)
    args = tuple(object() for _ in range(6))
    stable(*args, is_padding=object())
    assert compat.get_topk_operator("softmax") is stable
    assert calls == [args]


def test_ensure_patches_reapplies_a_successful_early_topk_import(monkeypatch):
    ops, compat, calls = _partially_imported_ops(monkeypatch)
    parent = types.ModuleType("metax_topk_runtime_test")
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    version_gate = types.ModuleType(parent.__name__ + "._compat")
    version_gate.legacy_torch_runtime = lambda: False
    monkeypatch.setitem(sys.modules, version_gate.__name__, version_gate)
    utils = types.ModuleType("vllm.utils")
    iu = types.ModuleType("vllm.utils.import_utils")
    sys.modules["vllm"].utils = utils
    utils.import_utils = iu
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, iu.__name__, iu)

    def import_patch(name):
        return compat if name.endswith(".moe_topk_compat") else types.ModuleType(name)

    monkeypatch.setattr("importlib.import_module", import_patch)
    patches = _load(
        monkeypatch,
        parent.__name__ + ".patches",
        _PATCHES / "__init__.py",
        package=True,
    )
    assert patches.PATCH_ERRORS == {}
    _finish_native_import(ops)
    assert patches.ensure_patches() == {}
    ops.topk_softmax(None, None, None, None)
    assert calls == [(None, None, None, None, False, None)]


def test_missing_mcoplib_installs_torch_route_and_preserves_padding(monkeypatch):
    ops, compat, _ = _partially_imported_ops(monkeypatch)
    _finish_native_import(ops)
    monkeypatch.setitem(sys.modules, "mcoplib", None)
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    assert compat.apply() is True
    assert ops.topk_softmax is compat._topk_softmax_torch
    assert ops.topk_sigmoid is compat._topk_sigmoid_torch
    assert os.environ["VLLM_MOE_SKIP_PADDING"] == "1"


def test_installed_but_broken_mcoplib_surfaces_import_failure(monkeypatch):
    _, compat, _ = _partially_imported_ops(monkeypatch)
    monkeypatch.setitem(sys.modules, "mcoplib._moe_C", None)
    with pytest.raises(ImportError):
        compat.apply()


@pytest.mark.parametrize("sigmoid", [False, True])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_flagos_route_preserves_bias_padding_weights_and_expanded_rows(
    monkeypatch, sigmoid, renormalize, with_bias
):
    package = types.ModuleType("vllm")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm", package)
    monkeypatch.setitem(sys.modules, "mcoplib", None)
    monkeypatch.setitem(sys.modules, "torch", torch)
    ops = types.ModuleType("vllm._custom_ops")
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    _load(monkeypatch, "metax_pure_topk_test", _PATCHES / "moe_topk_compat.py")
    logits = torch.tensor(
        [[2.0, 1.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0], [float("nan")] * 4],
        device="cpu",
    )
    bias = torch.tensor([0.0, 0.0, 0.0, 10.0]) if with_bias else None
    weights = torch.empty((3, 2), dtype=torch.float32)
    ids = torch.empty((3, 2), dtype=torch.int32)
    expanded_rows = torch.empty((3, 2), dtype=torch.int32)
    padding = torch.tensor([False, False, True])
    args = (weights, ids, expanded_rows, logits, renormalize, bias)
    scale = 2.5 if sigmoid else 1.0
    if sigmoid:
        ops.topk_sigmoid(*args, routed_scaling_factor=scale, is_padding=padding)
    else:
        ops.topk_softmax(*args, is_padding=padding)

    # Large expert-3 bias changes expert selection, never its output weight.
    # The second row ties, so lower expert IDs must win in a stable order.
    expected_ids = torch.tensor(
        [[3, 0], [3, 0], [-1, -1]] if with_bias else [[0, 1], [0, 1], [-1, -1]],
        dtype=torch.int32,
    )
    torch.testing.assert_close(ids, expected_ids)
    probabilities = (
        torch.sigmoid(logits[:2]) if sigmoid else torch.softmax(logits[:2], -1)
    )
    expected_weights = probabilities.gather(1, expected_ids[:2].long())
    if renormalize:
        expected_weights = expected_weights / expected_weights.sum(-1, keepdim=True)
    expected_weights *= scale
    torch.testing.assert_close(weights[:2], expected_weights)
    assert torch.count_nonzero(weights[2]) == 0
    torch.testing.assert_close(
        expanded_rows, torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.int32)
    )
    assert os.environ["VLLM_MOE_SKIP_PADDING"] == "1"


def _fl_router(monkeypatch, compat, vendor, padding, upstream, cached):
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = types.SimpleNamespace(vendor_name=vendor)
    monkeypatch.setitem(sys.modules, platforms.__name__, platforms)
    bridge_name = "vllm_fl.dispatch.backends.vendor.metax.patches.moe_topk_compat"
    monkeypatch.setitem(sys.modules, bridge_name, compat)
    path = _PATCHES.parents[4] / "ops/fused_moe/router.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "fused_topk"
    )
    namespace = {
        "torch": torch,
        "_get_padding_mask": lambda _: padding,
        "upstream_fused_topk": upstream,
        "_topk_softmax": cached,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["fused_topk"]


@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
def test_fl_metax_router_uses_stable_torch_bridge_after_upstream_overwrite(
    monkeypatch, scoring_func
):
    package = types.ModuleType("vllm")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm", package)
    monkeypatch.setitem(sys.modules, "mcoplib", None)
    ops = types.ModuleType("vllm._custom_ops")
    monkeypatch.setitem(sys.modules, ops.__name__, ops)
    compat = _load(monkeypatch, "metax_fl_route_test", _PATCHES / "moe_topk_compat.py")
    _finish_native_import(ops)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "MetaX padded routing must bypass upstream/FlagGems aliases"
        )

    padding = torch.tensor([False, False, True])
    route = _fl_router(monkeypatch, compat, "metax", padding, forbidden, forbidden)
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [-1.0, 0.0, 3.0, 1.0], [1.0] * 4])
    weights, ids, expanded = route(
        torch.zeros(3, 8), logits, 2, True, torch.int64, scoring_func
    )
    probabilities = (
        torch.softmax(logits[:2], -1)
        if scoring_func == "softmax"
        else logits[:2].sigmoid()
    )
    expected_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    expected_weights = probabilities.gather(1, expected_ids)
    expected_weights /= expected_weights.sum(-1, keepdim=True)
    torch.testing.assert_close(ids[:2], expected_ids)
    torch.testing.assert_close(weights[:2], expected_weights)
    assert (ids[2] == -1).all() and (weights[2] == 0).all()
    torch.testing.assert_close(
        expanded, torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.int32)
    )
    assert not getattr(ops.topk_softmax, compat._PATCH_FLAG, False)


def test_non_metax_padded_router_keeps_upstream_route(monkeypatch):
    expected = object()
    captured = {}

    def upstream(**kwargs):
        captured.update(kwargs)
        return expected

    route = _fl_router(
        monkeypatch, object(), "nvidia", torch.tensor([True]), upstream, None
    )
    hidden, logits = torch.zeros(1, 8), torch.zeros(1, 4)
    assert route(hidden, logits, 2, True) is expected
    assert captured["hidden_states"] is hidden
    assert captured["gating_output"] is logits


def test_unpadded_metax_softmax_keeps_flaggems_route(monkeypatch):
    def cached(weights, ids, expanded, logits, renormalize):
        weights.fill_(0.5)
        ids.fill_(1)
        return weights, ids

    route = _fl_router(monkeypatch, object(), "metax", None, None, cached)
    weights, ids, _ = route(torch.zeros(1, 8), torch.zeros(1, 4), 2, True)
    torch.testing.assert_close(weights, torch.full((1, 2), 0.5))
    torch.testing.assert_close(ids, torch.ones((1, 2), dtype=torch.int32))
