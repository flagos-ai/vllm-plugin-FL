"""CPU contract tests for the native sparse decode startup gate."""

from dataclasses import replace

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

from vllm_fl.ops import native_sparse as sparse

_NHD = (0, 1, 2, 3, 4)
_NATIVE_BACKEND = type(
    "MacaFlashAttentionBackend",
    (),
    {"__module__": "vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn"},
)


def _metadata_planner(blocks=600, dense_limit=8192):
    return sparse.MetadataRefreshPlanner(
        blocks, dense_limit, device="cpu", pin_memory=False
    )


def test_metadata_planner_tracks_reordered_requests_and_deduplicates_prefix_pages():
    planner = _metadata_planner(blocks=16, dense_limit=0)
    first_table = torch.tensor([[1, 2, 0, 0], [3, 4, 0, 0]], dtype=torch.int32)
    assert (
        planner.plan(["a", "b"], np.array([16, 32]), np.array([1, 1]), first_table) == 3
    )
    assert planner.jobs_cpu[:3].tolist() == [1, 3, 4]

    reordered_table = torch.tensor([[3, 4, 5, 0], [1, 2, 0, 0]], dtype=torch.int32)
    assert (
        planner.plan(["b", "a"], np.array([48, 32]), np.array([1, 1]), reordered_table)
        == 2
    )
    assert planner.jobs_cpu[:2].tolist() == [2, 5]

    shared = _metadata_planner(blocks=8, dense_limit=0)
    shared_table = torch.tensor([[1, 2, 0], [1, 2, 0]], dtype=torch.int32)
    assert (
        shared.plan(
            ["left", "right"], np.array([32, 32]), np.array([1, 1]), shared_table
        )
        == 2
    )
    assert shared.jobs_cpu[:2].tolist() == [1, 2]


def test_metadata_planner_does_not_mark_q_gt_1_prefill_pages_ready():
    planner = _metadata_planner(blocks=600, dense_limit=8192)
    table = torch.arange(1, 601, dtype=torch.int32).view(1, -1)

    assert planner.plan(["r"], np.array([8192]), np.array([16]), table) == 0
    assert planner._progress == {}
    assert planner.plan(["r"], np.array([8208]), np.array([1]), table) == 513
    assert planner.jobs_cpu[:513].tolist() == list(range(1, 514))


def test_sparse_graph_key_preserves_fields_and_separates_routes():
    base = BatchDescriptor(
        num_tokens=128,
        num_reqs=64,
        uniform=True,
        has_lora=True,
        num_active_loras=3,
    )
    dense = sparse.route_batch_descriptor(base, 8192, dense_limit=8192)
    sparse_decode = sparse.route_batch_descriptor(base, 8193, dense_limit=8192)

    assert dense == sparse.sparse_decode_batch_descriptor(base, False)
    assert sparse_decode == sparse.sparse_decode_batch_descriptor(base, True)
    assert dense is not sparse_decode
    assert dense != base and sparse_decode != base
    for field in ("num_tokens", "num_reqs", "uniform", "has_lora", "num_active_loras"):
        assert getattr(dense, field) == getattr(base, field)
        assert getattr(sparse_decode, field) == getattr(base, field)
    assert dense.sparse_decode is False
    assert sparse_decode.sparse_decode is True
    assert sparse.route_batch_descriptor(base, 8192, dense_limit=8192) is dense


def test_dummy_length_one_still_captures_both_explicit_full_decode_routes():
    base = BatchDescriptor(num_tokens=64, num_reqs=64, uniform=True)

    modes = sparse.full_decode_capture_modes(True, CUDAGraphMode.FULL, base)
    dense_key = sparse.sparse_decode_batch_descriptor(base, modes[0])
    sparse_key = sparse.sparse_decode_batch_descriptor(base, modes[1])

    assert modes == (False, True)
    assert not sparse.should_use_sparse_route(
        dense_key, CUDAGraphMode.NONE, max_seq_len=1, dense_limit=8192
    )
    assert sparse.should_use_sparse_route(
        sparse_key, CUDAGraphMode.NONE, max_seq_len=1, dense_limit=8192
    )


def _page_score_reference(query, kmin, kmax):
    q = query.detach().float().cpu().numpy().reshape(query.shape[0], 2, 8, 128)
    lo = kmin.detach().float().cpu().numpy()
    hi = kmax.detach().float().cpu().numpy()
    result = np.empty((query.shape[0], kmin.shape[0]), dtype=np.float32)
    for row in range(query.shape[0]):
        for page in range(kmin.shape[0]):
            kv_scores = []
            for kv_head in range(2):
                qh = q[row, kv_head]
                low_product = qh * lo[page, kv_head]
                high_product = qh * hi[page, kv_head]
                query_head_scores = np.maximum(low_product, high_product).sum(axis=-1)
                kv_scores.append(float(query_head_scores.max()))
            result[row, page] = max(kv_scores)
    return torch.from_numpy(result)


def test_cpu_page_scores_match_independent_fp32_reference():
    generator = torch.Generator().manual_seed(4871)
    query = torch.rand((2, 16, 128), generator=generator)
    query[1].neg_()
    query = query.to(torch.bfloat16)
    kmin = torch.randn((5, 2, 128), generator=generator).to(torch.bfloat16)
    kmax = (kmin.float() + torch.rand((5, 2, 128), generator=generator)).to(
        torch.bfloat16
    )
    expected = _page_score_reference(query, kmin, kmax)

    actual = sparse._cpu_page_scores(query, kmin, kmax)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)


def test_cpu_expected_pages_boundary_ties_sink_and_recent():
    cases = (
        (4096, 4096, list(range(256))),
        (4097, 4096, list(range(124)) + list(range(253, 257))),
        (2049, 1024, list(range(124)) + list(range(125, 129))),
    )
    for seq_len, threshold, expected_pages in cases:
        num_pages = (seq_len + 15) // 16
        selected = sparse._cpu_expected_pages(
            torch.zeros(num_pages), num_pages, seq_len, threshold, budget_tokens=2048
        )
        assert selected == expected_pages


def _supported_spec(**changes):
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=128,
        head_size_v=128,
        dtype=torch.bfloat16,
        kv_quant_mode=KVQuantMode.NONE,
    )
    return replace(spec, **changes) if changes else spec


def test_spec_gate_accepts_target_contract_and_rejects_incompatible_specs():
    spec = _supported_spec()
    assert type(spec) is FullAttentionSpec
    assert sparse.spec_is_supported(spec, _NATIVE_BACKEND, _NHD)
    for changes in (
        {"dtype": torch.float16},
        {"block_size": 32},
        {"num_kv_heads": 4},
        {"kv_quant_mode": KVQuantMode.FP8_PER_TENSOR},
    ):
        assert not sparse.spec_is_supported(
            _supported_spec(**changes), _NATIVE_BACKEND, _NHD
        )
