"""CPU contracts for the 0.32 runner / Ascend baseline merge."""

import ast
import builtins
import math
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.unit_tests.worker.test_model_runner_032_contracts import _runner_class


class Mode(IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


@pytest.mark.parametrize(
    "starts,tokens,requests,padded_requests,expected",
    [
        ([0, 1, 2], 4, 2, 4, [0, 1, 2, 3, 4]),
        ([0, 2, 5], 8, 2, None, [0, 2, 5, 8]),
        ([0, 2, 5], 5, 2, None, [0, 2, 5]),
    ],
)
def test_fia_padding_and_expert_padding_keep_independent_buffers(
    starts, tokens, requests, padded_requests, expected
):
    runner = _runner_class(
        {"_pad_query_start_loc_for_fia", "_prepare_padding_mask"}, torch=torch
    )()
    runner.uniform_decode_query_len = 1
    runner.arange_np = np.arange(16)
    runner.is_padding = torch.ones(16, dtype=torch.bool)
    cpu = np.full(17, -99, dtype=np.int32)
    cpu[: len(starts)] = starts
    copies = []
    buffer = SimpleNamespace(np=cpu, copy_to_gpu=lambda: copies.append(cpu.copy()))
    count = runner._pad_query_start_loc_for_fia(
        buffer, tokens, requests, padded_requests
    )
    assert count == len(expected) - 1
    assert cpu[: count + 1].tolist() == expected
    assert len(copies) == 1 and copies[0][: count + 1].tolist() == expected
    assert runner._prepare_padding_mask(starts[-1], tokens).tolist() == (
        [False] * starts[-1] + [True] * (tokens - starts[-1])
    )
    assert cpu[: len(starts)].tolist() == starts


@pytest.mark.parametrize("dp", [False, True])
@pytest.mark.parametrize("reason", ["none", "explicit", "cascade", "encoder", "eager"])
def test_full_graph_exclusions_survive_dp_redispatch(dp, reason):
    calls = []

    def dispatch(**kwargs):
        calls.append(kwargs)
        if kwargs["valid_modes"] is not None:
            mode = next(iter(kwargs["valid_modes"]))
        elif kwargs["invalid_modes"]:
            mode = Mode.PIECEWISE
        else:
            mode = Mode.FULL
        return mode, SimpleNamespace(num_tokens=kwargs["num_tokens"])

    parallel = SimpleNamespace(data_parallel_size=2 if dp else 1, data_parallel_rank=0)

    def coordinate(**kwargs):
        return False, torch.tensor([8, 8]), kwargs["cudagraph_mode"]

    runner = _runner_class(
        {"_determine_batch_execution_and_padding"},
        CUDAGraphMode=Mode,
        coordinate_batch_across_dp=coordinate,
    )()
    runner.uniform_decode_query_len = 1
    runner._is_uniform_decode = lambda **kwargs: True
    runner._pad_for_sequence_parallelism = lambda value: value
    runner.model_config = SimpleNamespace(is_encoder_decoder=reason == "encoder")
    runner.input_batch = SimpleNamespace(lora_id_to_lora_request={})
    runner.compilation_config = SimpleNamespace(
        pass_config=SimpleNamespace(enable_sp=False)
    )
    runner.parallel_config = parallel
    runner.vllm_config = SimpleNamespace(
        parallel_config=parallel,
        observability_config=SimpleNamespace(cudagraph_metrics=False),
    )
    runner.cudagraph_dispatcher = SimpleNamespace(dispatch=dispatch)
    mode, descriptor, *_ = runner._determine_batch_execution_and_padding(
        4,
        4,
        np.ones(4, dtype=np.int32),
        1,
        use_cascade_attn=reason == "cascade",
        disable_full=reason == "explicit",
        num_encoder_reqs=1 if reason == "encoder" else 0,
        force_eager=reason == "eager",
    )
    expected_invalid = (
        {Mode.FULL} if reason in ("explicit", "cascade", "encoder") else None
    )
    assert len(calls) == (2 if dp else 1)
    assert all(call["invalid_modes"] == expected_invalid for call in calls)
    assert mode is (
        Mode.NONE
        if reason == "eager"
        else Mode.PIECEWISE
        if expected_invalid
        else Mode.FULL
    )
    assert descriptor.num_tokens == (8 if dp else 4)
    if reason == "eager":
        assert all(call["valid_modes"] == {Mode.NONE} for call in calls)


def _ascend_binding_class(hybrid=None):
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm_fl/dispatch/backends/vendor/ascend/impl/gdn.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    layer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "AscendQwenGatedDeltaNetAttention"
    )
    layer.body = [
        node
        for node in layer.body
        if isinstance(node, ast.FunctionDef) and node.name == "bind_kv_cache"
    ]

    class NativeLayer:
        def bind_kv_cache(self, cache):
            pytest.fail("Native state tuples must not be unpacked or flattened again")

    def restricted_import(name, *args, **kwargs):
        if name == "vllm_fl.kv_cache.ascend.hybrid" and hybrid is not None:
            return hybrid
        return builtins.__import__(name, *args, **kwargs)

    namespace = {
        "QwenGatedDeltaNetAttention": NativeLayer,
        "__builtins__": {**vars(builtins), "__import__": restricted_import},
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[layer], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace["AscendQwenGatedDeltaNetAttention"]


@pytest.mark.parametrize("container", [tuple, list])
def test_native_gdn_state_views_preserve_page_offsets_and_storage_alias(container):
    raw = torch.zeros(16 + 4 * 96, dtype=torch.uint8)
    conv = raw[16:].view(torch.bfloat16).as_strided((4, 2, 3), (48, 3, 1))
    state = raw[28:].view(torch.float32).as_strided((4, 4, 4), (24, 4, 1))
    layer = _ascend_binding_class()()
    layer.bind_kv_cache(container([conv, state]))
    assert layer.kv_cache == (conv, state)
    assert layer.kv_cache[0] is conv and layer.kv_cache[1] is state
    assert conv.stride(0) == 48 and state.stride(0) == 24
    layer.kv_cache[0][1].fill_(3)
    layer.kv_cache[1][2].fill_(7)
    expected = torch.zeros_like(raw)
    expected[16 + 96 : 16 + 96 + 12].copy_(
        torch.full((6,), 3, dtype=torch.bfloat16).view(torch.uint8)
    )
    expected[28 + 2 * 96 : 28 + 2 * 96 + 64].copy_(
        torch.full((16,), 7, dtype=torch.float32).view(torch.uint8)
    )
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)


def test_legacy_raw_dense_binding_remains_unchanged():
    path = Path(__file__).resolve().parents[3] / "vllm_fl/kv_cache/ascend/hybrid.py"
    names = {"dense_mamba_views", "uses_dense_hybrid_cache"}
    functions = [
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "math": math,
        "get_dtype_size": lambda dtype: torch.empty((), dtype=dtype).element_size(),
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    layer = _ascend_binding_class(SimpleNamespace(**namespace))()
    layer.model_config = SimpleNamespace(
        is_hybrid=True, hf_text_config=SimpleNamespace(model_type="qwen3_5_text")
    )
    layer.get_state_shape = lambda: ((2, 3), (4, 4))
    layer.get_state_dtype = lambda: (torch.float16, torch.float32)
    pages = torch.zeros(3, 1, 1, 140, dtype=torch.uint8)
    layer.bind_kv_cache(pages)
    conv, state = layer.kv_cache
    assert conv.shape == (3, 2, 3) and state.shape == (3, 4, 4)
    assert conv.data_ptr() == pages.data_ptr()
    assert state.data_ptr() == pages.data_ptr() + 36
    state[1].fill_(5)
    expected = torch.zeros_like(pages).flatten()
    expected[100:164].copy_(torch.full((16,), 5, dtype=torch.float32).view(torch.uint8))
    torch.testing.assert_close(pages.flatten(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("ascend", [False, True])
def test_ascend_adapter_only_changes_layer_binding_views(ascend):
    backing = torch.arange(40, dtype=torch.float32)
    manager = backing[4:36].view(4, 1, 4, 2)
    kernel = manager.view(8, 1, 2, 2)
    states = (manager[:, 0, :2], manager[:, 0, 2:])
    group = object()
    events = []

    def adapt(caches, groups):
        assert caches == {"layer.0": kernel} and groups == [group]
        events.append("adapt")
        return {"layer.0": states}

    def selected_import(name, *args, **kwargs):
        if name == "vllm_fl.kv_cache.ascend.native_032":
            assert ascend
            return SimpleNamespace(adapt_layer_kv_caches=adapt)
        return builtins.__import__(name, *args, **kwargs)

    def bind(
        caches, context, runner_caches, count, *, kv_cache_groups, layer_kv_caches
    ):
        assert caches["layer.0"] is manager
        expected = states if ascend else kernel
        assert layer_kv_caches["layer.0"] is expected
        runner_caches.append(caches["layer.0"])
        events.append("bind")

    runner = _runner_class(
        {"initialize_kv_cache_tensors"},
        __builtins__={**vars(builtins), "__import__": selected_import},
        allocate_kv_cache=lambda *args: {"layer.0": manager},
        map_kv_caches_to_kernel_blocks=lambda caches, groups: {"layer.0": kernel},
        bind_kv_cache=bind,
    )()
    runner.use_ascend_attention = ascend
    runner.device = torch.device("cpu")
    runner.cache_config = SimpleNamespace(get_resolved_kv_cache_layout=lambda: "LBNHC")
    runner.shared_kv_cache_layers = {}
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen3_5")
    )
    runner.attn_groups = [[group]]
    runner.speculative_config = None
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.kv_caches = []
    caches = runner.initialize_kv_cache_tensors(
        SimpleNamespace(kv_cache_groups=[]), [2]
    )
    assert events == (["adapt", "bind"] if ascend else ["bind"])
    assert caches["layer.0"] is manager and runner.kv_caches[0] is manager
    torch.testing.assert_close(
        backing, torch.arange(40, dtype=torch.float32), rtol=0, atol=0
    )
