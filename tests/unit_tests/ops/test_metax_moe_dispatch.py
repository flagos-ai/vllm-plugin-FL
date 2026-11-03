# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU routing and caller contracts, without executing hardware providers."""

import ast
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
ADAPTER = ROOT / "vllm_fl/dispatch/backends/flaggems/impl/fused_moe.py"
UTILS = ROOT / "vllm_fl/ops/fused_moe/fused_moe_utils.py"
LEGACY = ROOT / "vllm_fl/ops/fused_moe/fused_moe.py"
SAFE_MODULE = "vllm_fl.dispatch.backends.vendor.metax.impl.moe_align_block_size"


def _module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture
def adapter(monkeypatch):
    _module(monkeypatch, "vllm", __path__=[])
    _module(monkeypatch, "vllm.utils", __path__=[])
    _module(monkeypatch, "vllm.logger", init_logger=logging.getLogger)
    _module(
        monkeypatch,
        "vllm.triton_utils",
        triton=SimpleNamespace(cdiv=lambda n, b: (n + b - 1) // b),
    )
    _module(
        monkeypatch,
        "vllm.utils.math_utils",
        round_up=lambda n, b: ((n + b - 1) // b) * b,
    )
    platform = SimpleNamespace(vendor_name="metax")
    _module(monkeypatch, "vllm.platforms", current_platform=platform)
    spec = importlib.util.spec_from_file_location(
        "_metax_moe_adapter_contract", ADAPTER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Tests use CPU allocations while retaining accelerator device metadata.
    module.torch = SimpleNamespace(
        int32=torch.int32,
        empty=lambda shape, *, dtype, device: torch.empty(shape, dtype=dtype),
    )
    return module, platform


class _Ids:
    def __init__(self, device):
        self.device = SimpleNamespace(type=device)

    def numel(self):
        return 12


def _providers(monkeypatch, fail=None):
    calls = []
    marker = (object(), object(), object())

    def safe(*args):
        calls.append(("safe", args))
        if fail == "safe":
            raise RuntimeError("safe provider failed")
        return marker

    def provider(name):
        def old(*args):
            calls.append((name, args))
            if fail == name:
                raise RuntimeError(f"{name} provider failed")
            _, _, _, sorted_ids, expert_ids, postpad = args
            sorted_ids.fill_(12)
            expert_ids.fill_(0)
            postpad.zero_()

        return old

    _module(monkeypatch, SAFE_MODULE, moe_align_block_size=safe)
    _module(monkeypatch, "flag_gems", moe_align_block_size_triton=provider("gems"))
    _module(
        monkeypatch, "flaggems_vllm", moe_align_block_size_triton=provider("bridge")
    )
    return calls, marker


@pytest.mark.parametrize(
    "device,vendor,selected",
    [
        ("cuda", "metax", "safe"),
        ("cuda", "nvidia", "bridge"),
        ("gcu", "metax", "gems"),
        ("cpu", "metax", "bridge"),
    ],
)
def test_safe_align_requires_both_cuda_device_and_metax_vendor(
    monkeypatch, adapter, device, vendor, selected
):
    module, platform = adapter
    platform.vendor_name = vendor
    ids = _Ids(device)
    calls, marker = _providers(monkeypatch)
    result = module.moe_align_block_size_flaggems(
        ids, 4, 7, pad_sorted_ids=True, ignore_invalid_experts=True
    )
    assert len(calls) == 1
    name, args = calls[0]
    assert name == selected and args[0] is ids
    if selected == "safe":
        assert args[1:] == (4, 7, None, True, True)
        assert result is marker
    else:
        assert args[1:3] == (7, 4)
        assert len(args) == 6
        assert all(actual is passed for actual, passed in zip(result, args[3:]))
        assert result[2].item() == 0


@pytest.mark.parametrize(
    "device,vendor,selected",
    [("cuda", "metax", "safe"), ("cuda", "nvidia", "bridge"), ("gcu", "gcu", "gems")],
)
def test_alignment_provider_failures_propagate_without_fallback(
    monkeypatch, adapter, device, vendor, selected
):
    module, platform = adapter
    platform.vendor_name = vendor
    calls, _ = _providers(monkeypatch, fail=selected)
    with pytest.raises(RuntimeError, match=f"{selected} provider failed"):
        module.moe_align_block_size_flaggems(_Ids(device), 4, 7)
    assert len(calls) == 1 and calls[0][0] == selected


def _prepare_assignment(vendor, align):
    tree = ast.parse(UTILS.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_prepare_expert_assignment"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    namespace = {
        "torch": torch,
        "current_platform": SimpleNamespace(vendor_name=vendor),
        "_moe_align_block_size": align,
    }
    module = ast.fix_missing_locations(
        ast.Module(body=[future, function], type_ignores=[])
    )
    exec(compile(module, str(UTILS), "exec"), namespace)
    return namespace[function.name]


def _raw_ids(tokens, dtype):
    ids = (torch.arange(tokens * 8) % 256).reshape(tokens, 8)
    ids.reshape(-1)[:3] = torch.tensor([-1, 256, 263])
    if dtype == torch.int64:
        ids.reshape(-1)[3:5] = torch.tensor([2**32 + 1, -(2**32)])
    return ids.to(dtype)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("vendor", ["metax", "nvidia"])
def test_naive_boundary_normalizes_only_metax_invalid_ids_without_mutation(
    dtype, vendor
):
    ids = _raw_ids(8, dtype)
    original = ids.clone()
    align = Mock(side_effect=AssertionError("M8 must use naive assignment"))
    prepare = _prepare_assignment(vendor, align)
    sorted_ids, expert_ids, postpad = prepare(
        ids, {"BLOCK_SIZE_M": 16}, 8, 8, 256, None
    )
    expected = original.reshape(-1)
    if vendor == "metax":
        expected = torch.where((expected >= 0) & (expected < 256), expected, -1)
    assert sorted_ids is None
    assert expert_ids.dtype == dtype
    torch.testing.assert_close(expert_ids, expected, rtol=0, atol=0)
    torch.testing.assert_close(ids, original, rtol=0, atol=0)
    assert postpad.dtype == torch.int32 and postpad.tolist() == [64 * 16]
    align.assert_not_called()


@pytest.mark.parametrize("vendor", ["metax", "nvidia"])
def test_next_batch_size_delegates_original_routes_to_alignment(vendor):
    ids = _raw_ids(9, torch.int64)
    original = ids.clone()
    expected = (object(), object(), object())
    align = Mock(return_value=expected)
    prepare = _prepare_assignment(vendor, align)
    result = prepare(
        ids,
        {"BLOCK_SIZE_M": 16},
        9,
        8,
        256,
        None,
        ignore_invalid_experts=True,
    )
    assert result is expected
    args, kwargs = align.call_args
    assert len(args) == 4 and args[0] is ids and args[1:] == (16, 256, None)
    assert kwargs == {"ignore_invalid_experts": True}
    torch.testing.assert_close(ids, original, rtol=0, atol=0)


def _cleanup_statements(path, function_name, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scope = tree.body
    if class_name is not None:
        scope = next(
            node.body
            for node in scope
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    function = next(
        node
        for node in scope
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    # Extract and execute the actual cleanup branch and its immediately
    # following statement. Legacy's branch lives inside its chunk loop.
    for node in ast.walk(function):
        if not isinstance(node, (ast.FunctionDef, ast.For)):
            continue
        for index, statement in enumerate(node.body[:-1]):
            if isinstance(statement, ast.If) and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "zero_invalid_expert_outputs"
                for child in ast.walk(statement)
            ):
                return [statement, node.body[index + 1]]
    raise AssertionError("MoE implementation has no cleanup/sum call boundary")


@pytest.mark.parametrize("implementation", ["modular", "legacy"])
@pytest.mark.parametrize("vendor", ["metax", "nvidia", "gcu"])
def test_real_cleanup_call_boundary_clears_poison_before_sum_only_on_metax(
    monkeypatch, implementation, vendor
):
    ids = torch.tensor([[0, -1, 6, 7], [1, 2, 3, -2], [5, 4, -1, 0]])
    mapping = torch.tensor([0, -1, 1, 2, -1, 3, 4], dtype=torch.int32)
    valid = (ids >= 0) & (ids < 7)
    valid &= mapping[ids.clamp(0, 6)] >= 0
    finite = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5) + 1
    cache = torch.where(valid[..., None], finite, float("nan"))
    original = cache.clone()
    expected_cache = torch.where(valid[..., None], finite, 0.0)
    output = torch.full((3, 5), -101.0)
    backing_output = torch.full((9, 5), -101.0)
    events = []

    def cleanup(actual_cache, actual_ids, experts, actual_map):
        assert actual_cache is cache and actual_ids is ids
        assert experts == 7 and actual_map is mapping
        assert torch.isnan(actual_cache[~valid]).all().item()
        events.append("cleanup")
        actual_cache.copy_(torch.where(valid[..., None], actual_cache, 0.0))

    def reduce(actual_cache, actual_output):
        assert actual_cache.data_ptr() == cache.data_ptr()
        assert actual_cache.shape == cache.shape
        assert events == (["cleanup"] if vendor == "metax" else [])
        expected_output = output if implementation == "modular" else backing_output[2:5]
        assert actual_output.data_ptr() == expected_output.data_ptr()
        events.append("sum")
        actual_output.copy_(actual_cache.sum(1))

    _module(monkeypatch, SAFE_MODULE, zero_invalid_expert_outputs=cleanup)
    namespace = {
        "current_platform": SimpleNamespace(vendor_name=vendor),
        "intermediate_cache3": cache,
        "global_num_experts": 7,
        "expert_map": mapping,
        "output": output,
        "self": SimpleNamespace(moe_sum=reduce),
        "_moe_sum": reduce,
        "out_hidden_states": backing_output,
        "begin_chunk_idx": 2,
        "end_chunk_idx": 5,
        # The legacy caller must pass the current chunk's IDs.
        "topk_ids": ids if implementation == "modular" else torch.zeros_like(ids),
        "curr_topk_ids": ids,
    }
    statements = (
        _cleanup_statements(UTILS, "apply", "TritonExpertsFL")
        if implementation == "modular"
        else _cleanup_statements(LEGACY, "fused_experts_impl")
    )
    module = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
    exec(
        compile(module, str(UTILS if implementation == "modular" else LEGACY), "exec"),
        namespace,
    )
    assert events == (["cleanup", "sum"] if vendor == "metax" else ["sum"])
    actual_output = output if implementation == "modular" else backing_output[2:5]
    if vendor == "metax":
        torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
        torch.testing.assert_close(actual_output, expected_cache.sum(1), rtol=0, atol=0)
        torch.testing.assert_close(cache[valid], original[valid], rtol=0, atol=0)
    else:
        torch.testing.assert_close(cache, original, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(
            actual_output, original.sum(1), rtol=0, atol=0, equal_nan=True
        )
