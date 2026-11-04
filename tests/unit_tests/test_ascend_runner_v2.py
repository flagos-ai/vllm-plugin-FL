# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the V2 adapter's phase and physical block operations."""

import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _load_definitions(relative_path, names, namespace):
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(selected) == len(names)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *selected], type_ignores=[])
    )
    exec(compile(module, relative_path, "exec"), namespace)
    return namespace


@pytest.fixture
def adapter():
    namespace = {"torch": torch, "np": np, "Enum": Enum, "GPUModelRunner": object}
    _load_definitions(
        "vllm_fl/dispatch/backends/vendor/ascend/impl/attention.py",
        {
            "AscendAttentionState",
            "classify_ascend_attention_state",
            "get_ascend_v2_attention_state",
        },
        namespace,
    )
    _load_definitions(
        "vllm_fl/kv_cache/ascend/hybrid.py",
        {"_deduplicate_block_planes", "copy_block_planes", "zero_block_planes"},
        namespace,
    )
    _load_definitions(
        "vllm_fl/worker/model_runner_v2.py",
        {"_AscendBlockZeroer", "GPUModelRunnerFLV2"},
        namespace,
    )
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize(
    "query_lens,seq_lens,chunked,method,expected",
    [
        ([1], [1], False, None, "PrefillNoCache"),
        ([7, 3], [7, 3], True, None, "PrefillNoCache"),
        ([1, 1], [6, 13], True, None, "DecodeOnly"),
        ([1, 4], [6, 12], True, None, "ChunkedPrefill"),
        ([4], [12], False, None, "PrefillCacheHit"),
        ([1], [7], True, "mtp", "SpecDecoding"),
    ],
)
def test_v2_phase_uses_exact_lengths(
    adapter, query_lens, seq_lens, chunked, method, expected
):
    common = SimpleNamespace(
        num_reqs=len(query_lens),
        query_start_loc_cpu=torch.tensor(
            [0, *np.cumsum(query_lens)], dtype=torch.int32
        ),
        # V2's optimistic bound intentionally disagrees with its real batch.
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens) + 100,
    )
    actual = adapter.get_ascend_v2_attention_state(
        common, torch.tensor(seq_lens), chunked, method
    )
    assert actual.name == expected


def test_v2_request_update_zeros_then_copies_all_dense_planes(adapter):
    runner = object.__new__(adapter.GPUModelRunnerFLV2)
    planes = [
        torch.arange(4 * width, dtype=torch.int32).reshape(4, width) + 1
        for width in (3, 8, 8)
    ]
    before = [plane.clone() for plane in planes]
    runner._ascend_block_planes = planes
    computed = np.array([0, 0, 0], dtype=np.int32)
    prefill = np.array([3, 8, 2], dtype=np.int32)
    computed_prefill = np.zeros(3, dtype=np.int32)
    runner.req_states = SimpleNamespace(
        req_id_to_index={"r1": 1},
        num_computed_tokens_np=computed,
        prefill_len=SimpleNamespace(np=prefill),
        num_computed_prefill_tokens=computed_prefill,
    )
    appends = []
    runner.block_tables = SimpleNamespace(
        append_block_ids=lambda *args, **kwargs: appends.append((args, kwargs))
    )
    output = SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r1"], num_computed_tokens=[9], new_block_ids=[([2],)]
        ),
        new_block_ids_to_zero=[3],
        kv_cache_block_copies=[(1, 2), (2, 3)],
    )
    runner.update_requests(output)
    assert computed.tolist() == [0, 9, 0]
    assert computed_prefill.tolist() == [0, 8, 0]
    assert appends == [((1, ([2],)), {"overwrite": False})]
    for actual, original in zip(planes, before):
        torch.testing.assert_close(actual[0], original[0])
        torch.testing.assert_close(actual[1], original[1])
        torch.testing.assert_close(actual[2], original[1])
        torch.testing.assert_close(actual[3], original[2])


def test_v2_zeroer_warmup_preserves_dense_plane_addresses(adapter):
    planes = [torch.ones(4, width, dtype=torch.uint8) for width in (12, 32, 32)]
    pointers = [plane.data_ptr() for plane in planes]
    zeroer = adapter._AscendBlockZeroer(planes)
    zeroer.warmup(4)
    zeroer.zero_block_ids([2])
    for plane, pointer in zip(planes, pointers):
        assert plane.data_ptr() == pointer
        assert torch.count_nonzero(plane[[0, 2]]).item() == 0
        assert torch.all(plane[[1, 3]] == 1).item()


def test_v2_runner_uses_real_upstream_v2_mro():
    pytest.importorskip("vllm")
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as UpstreamV2

    from vllm_fl.worker.model_runner_v2 import GPUModelRunnerFLV2

    assert GPUModelRunnerFLV2.__mro__[1] is UpstreamV2
    modules = {base.__module__ for base in GPUModelRunnerFLV2.__mro__}
    assert "vllm_fl.worker.model_runner" not in modules
    assert "vllm.v1.worker.gpu_model_runner" not in modules
