"""Correctness and CUDA-graph tests for the selfdev QSA split path."""

from __future__ import annotations

import importlib
import math

import pytest
import torch


def _load_qsa_ops():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; QSA split validation requires a GPU")
    try:
        return importlib.import_module(
            "vllm_fl.models.qwen3_8_flash_next.gpu.ops.qsa"
        )
    except Exception as exc:  # pragma: no cover - target-GPU import guard
        pytest.fail(f"vLLM QSA plugin import failed: {type(exc).__name__}: {exc}")


def _case(
    device: torch.device,
    *,
    rows: int = 8,
    topk: int = 513,
    head_dim: int = 256,
) -> dict[str, torch.Tensor]:
    page_size = 16
    pages = math.ceil(topk / page_size)
    physical_blocks = rows * pages
    # Padding the final dimension preserves stride-one dim access while making
    # page/head strides non-contiguous, matching allocator-owned cache views.
    q_storage = torch.randn(
        rows, 3, head_dim + 8, dtype=torch.bfloat16, device=device
    )
    k_storage = torch.randn(
        physical_blocks,
        page_size,
        1,
        head_dim + 8,
        dtype=torch.bfloat16,
        device=device,
    )
    v_storage = torch.randn_like(k_storage)
    q = q_storage[..., :head_dim]
    k_cache = k_storage[..., :head_dim]
    v_cache = v_storage[..., :head_dim]
    # Reverse each request's page order to exercise the page-table indirection.
    table = torch.arange(
        physical_blocks, dtype=torch.int32, device=device
    ).reshape(rows, pages).flip(1)
    indices = torch.arange(topk, dtype=torch.int32, device=device).expand(
        rows, -1
    ).clone()
    # One sentinel and one out-of-table token per row must be ignored by both
    # the split and single kernels.
    indices[:, -1] = -1
    indices[:, -2] = pages * page_size + 5
    token_to_req = torch.arange(rows, dtype=torch.int32, device=device)
    gate = torch.randn_like(q)
    return {
        "q": q,
        "k": k_cache,
        "v": v_cache,
        "indices": indices,
        "table": table,
        "token_to_req": token_to_req,
        "gate": gate,
    }


def _workspace(case: dict[str, torch.Tensor], splits: int):
    q = case["q"]
    return (
        torch.empty(
            q.shape[0], q.shape[1], splits, q.shape[2],
            dtype=torch.float32,
            device=q.device,
        ),
        torch.empty(
            q.shape[0], q.shape[1], splits,
            dtype=torch.float32,
            device=q.device,
        ),
        torch.empty(
            q.shape[0], q.shape[1], splits,
            dtype=torch.float32,
            device=q.device,
        ),
    )


def _run(ops, case, out, *, workspace=None):
    return ops.qsa_sparse_paged_attention(
        case["q"],
        case["k"],
        case["v"],
        case["indices"],
        case["table"],
        case["token_to_req"],
        256**-0.5,
        out,
        gate=case["gate"],
        split_workspace=workspace,
    )


def _assert_split_error(candidate: torch.Tensor, baseline: torch.Tensor) -> None:
    """Use explicit BF16 error gates instead of a broad relative tolerance."""

    diff = (candidate.float() - baseline.float()).abs()
    flat = diff.flatten()
    max_abs = float(flat.max().item())
    rmse = float(torch.sqrt(torch.mean(diff.square())).item())
    p99 = float(torch.quantile(flat, 0.99).item())
    print(
        f"qsa split error: max_abs={max_abs:.8g} rmse={rmse:.8g} "
        f"p99={p99:.8g}"
    )
    assert math.isfinite(max_abs) and math.isfinite(rmse) and math.isfinite(p99)
    # The merge is FP32 and both paths round the gated result to BF16.  These
    # gates leave room for reduction-order ULPs while catching a real ABI or
    # page-layout mismatch (the measured rows64/topk2051 max is <1e-3).
    assert max_abs <= 1.0e-2
    assert rmse <= 1.0e-3
    assert p99 <= 2.0e-3


@pytest.mark.gpu
def test_qsa_split8_matches_single_with_invalid_pages_and_strides(monkeypatch):
    ops = _load_qsa_ops()
    device = torch.device("cuda")
    case = _case(device)
    assert case["q"].shape == (8, 3, 256)
    assert case["k"].stride(-1) == 1
    assert case["k"].stride(1) != case["k"].shape[2] * case["k"].shape[3]

    baseline = torch.empty_like(case["q"])
    monkeypatch.setenv("QWEN4_QSA_SPLIT_TOPK", "1")
    _run(ops, case, baseline)
    torch.cuda.synchronize()

    candidate = torch.empty_like(case["q"])
    monkeypatch.setenv("QWEN4_QSA_SPLIT_TOPK", "8")
    workspace = _workspace(case, 8)
    _run(ops, case, candidate, workspace=workspace)
    torch.cuda.synchronize()
    assert torch.isfinite(candidate.float()).all()
    _assert_split_error(candidate, baseline)

    # The reduction has no atomics and should be repeatable for the same input.
    repeat = torch.empty_like(case["q"])
    _run(ops, case, repeat, workspace=workspace)
    torch.cuda.synchronize()
    assert torch.equal(candidate, repeat)


@pytest.mark.gpu
def test_qsa_split_dispatch_topk_boundaries(monkeypatch):
    ops = _load_qsa_ops()
    device = torch.device("cuda")
    case = _case(device, rows=2, topk=512)
    monkeypatch.setenv("QWEN4_QSA_SPLIT_TOPK", "8")
    assert (
        ops.qsa_sparse_split_count(case["q"], case["k"], 511) == 1
    )
    assert (
        ops.qsa_sparse_split_count(case["q"], case["k"], 512) == 8
    )
    assert (
        ops.qsa_sparse_split_count(case["q"], case["k"], 513) == 8
    )

    # Execute both sides of the dispatch boundary.  511 remains the explicit
    # single-kernel fallback, while 512 enters split=8.
    case_fallback = _case(device, rows=2, topk=511)
    fallback_baseline = torch.empty_like(case_fallback["q"])
    monkeypatch.setenv("QWEN4_QSA_SPLIT_TOPK", "1")
    _run(ops, case_fallback, fallback_baseline)
    torch.cuda.synchronize()
    fallback_candidate = torch.empty_like(case_fallback["q"])
    monkeypatch.setenv("QWEN4_QSA_SPLIT_TOPK", "8")
    _run(ops, case_fallback, fallback_candidate)
    torch.cuda.synchronize()
    _assert_split_error(fallback_candidate, fallback_baseline)

    baseline = torch.empty_like(case["q"])
    monkeypatch.setenv("QWEN4_QSA_SPLIT_TOPK", "1")
    _run(ops, case, baseline)
    torch.cuda.synchronize()
    candidate = torch.empty_like(case["q"])
    monkeypatch.setenv("QWEN4_QSA_SPLIT_TOPK", "8")
    _run(ops, case, candidate, workspace=_workspace(case, 8))
    torch.cuda.synchronize()
    _assert_split_error(candidate, baseline)


@pytest.mark.gpu
def test_qsa_split_rows64_invalid_requests_pages_and_gate_extremes(monkeypatch):
    """Exercise the measured bucket plus every invalid-output gate."""

    ops = _load_qsa_ops()
    device = torch.device("cuda")
    case = _case(device, rows=64, topk=2051)
    case["token_to_req"][0] = -1
    case["token_to_req"][1] = 64
    case["table"][2, 0] = -1
    case["table"][3, -1] = case["k"].shape[0] + 7
    case["indices"][4].fill_(-1)  # all-invalid selection
    case["gate"][6].fill_(-80)
    case["gate"][7].fill_(80)

    baseline = torch.empty_like(case["q"])
    monkeypatch.setenv("QWEN4_QSA_SPLIT_TOPK", "1")
    _run(ops, case, baseline)
    torch.cuda.synchronize()

    candidate = torch.empty_like(case["q"])
    monkeypatch.setenv("QWEN4_QSA_SPLIT_TOPK", "8")
    workspace = _workspace(case, 8)
    _run(ops, case, candidate, workspace=workspace)
    torch.cuda.synchronize()
    assert torch.isfinite(candidate.float()).all()
    _assert_split_error(candidate, baseline)
    for row in (0, 1, 4):
        assert torch.equal(candidate[row], torch.zeros_like(candidate[row]))
    assert float(candidate[6].float().abs().max()) < 1.0e-3

    # +80 rounds sigmoid to one, so the gated output must equal the same
    # kernel's ungated result for that valid row.
    ungated = torch.empty_like(case["q"])
    ops.qsa_sparse_paged_attention(
        case["q"],
        case["k"],
        case["v"],
        case["indices"],
        case["table"],
        case["token_to_req"],
        256**-0.5,
        ungated,
        gate=None,
        split_workspace=workspace,
    )
    torch.cuda.synchronize()
    assert float((candidate[7].float() - ungated[7].float()).abs().max()) <= 1.0e-2


@pytest.mark.gpu
def test_qsa_split8_graph_replay_with_changed_inputs(monkeypatch):
    ops = _load_qsa_ops()
    device = torch.device("cuda")
    case = _case(device)
    monkeypatch.setenv("QWEN4_QSA_SPLIT_TOPK", "8")
    # Require mode makes a successful capture evidence that the split/merge
    # path received a warmed fixed workspace rather than falling back to the
    # single CTA kernel on a capture miss.
    monkeypatch.setenv("QWEN4_QSA_SPLIT_REQUIRE", "1")
    cache = {}
    splits, workspace = ops.qsa_prepare_split_workspace(
        case["q"], case["k"], case["indices"].shape[1], cache
    )
    assert splits == 8
    assert workspace is not None
    captured_out = torch.empty_like(case["q"])

    # Eager warmup compiles both kernels before capture.
    for _ in range(3):
        _run(ops, case, captured_out, workspace=workspace)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        _run(ops, case, captured_out, workspace=workspace)

    new_case = _case(device)
    case["q"].copy_(new_case["q"])
    case["gate"].copy_(new_case["gate"])
    case["indices"].copy_(new_case["indices"])
    graph.replay()
    torch.cuda.synchronize()

    expected = torch.empty_like(case["q"])
    monkeypatch.setenv("QWEN4_QSA_SPLIT_TOPK", "1")
    _run(ops, case, expected)
    torch.cuda.synchronize()
    _assert_split_error(captured_out, expected)

    # Capture a second layer/bucket without resizing the first workspace.  A
    # real input mutation and replay here catches accidental cross-bucket
    # aliasing that a pointer-only cache test cannot see.
    monkeypatch.setenv("QWEN4_QSA_SPLIT_TOPK", "8")
    case_b = _case(device, rows=16, topk=1025)
    splits_b, workspace_b = ops.qsa_prepare_split_workspace(
        case_b["q"], case_b["k"], case_b["indices"].shape[1], cache
    )
    assert splits_b == 8
    assert workspace_b is not None
    assert set(t.data_ptr() for t in workspace_b).isdisjoint(
        t.data_ptr() for t in workspace
    )
    captured_out_b = torch.empty_like(case_b["q"])
    for _ in range(3):
        _run(ops, case_b, captured_out_b, workspace=workspace_b)
    torch.cuda.synchronize()
    graph_b = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_b, capture_error_mode="thread_local"):
        _run(ops, case_b, captured_out_b, workspace=workspace_b)
    new_case_b = _case(device, rows=16, topk=1025)
    case_b["q"].copy_(new_case_b["q"])
    case_b["gate"].copy_(new_case_b["gate"])
    case_b["indices"].copy_(new_case_b["indices"])
    graph_b.replay()
    torch.cuda.synchronize()
    expected_b = torch.empty_like(case_b["q"])
    monkeypatch.setenv("QWEN4_QSA_SPLIT_TOPK", "1")
    _run(ops, case_b, expected_b)
    torch.cuda.synchronize()
    _assert_split_error(captured_out_b, expected_b)


@pytest.mark.gpu
def test_qsa_split_workspace_is_per_layer_and_bucket(monkeypatch):
    """Old captures keep their partial addresses across bucket changes."""

    ops = _load_qsa_ops()
    device = torch.device("cuda")
    monkeypatch.setenv("QWEN4_QSA_SPLIT_TOPK", "8")
    layer_a = {}
    layer_b = {}

    case_a = _case(device, rows=8, topk=513)
    splits_a, workspace_a = ops.qsa_prepare_split_workspace(
        case_a["q"], case_a["k"], case_a["indices"].shape[1], layer_a
    )
    assert splits_a == 8
    assert workspace_a is not None
    # Same layer and exact bucket must reuse the same allocations.
    _, workspace_a_repeat = ops.qsa_prepare_split_workspace(
        case_a["q"], case_a["k"], case_a["indices"].shape[1], layer_a
    )
    assert workspace_a_repeat is not None
    assert tuple(t.data_ptr() for t in workspace_a_repeat) == tuple(
        t.data_ptr() for t in workspace_a
    )

    # A different TopK/row bucket gets new addresses, so it cannot resize or
    # invalidate the graph captured with the first bucket.
    case_a_next = _case(device, rows=16, topk=1025)
    splits_next, workspace_next = ops.qsa_prepare_split_workspace(
        case_a_next["q"],
        case_a_next["k"],
        case_a_next["indices"].shape[1],
        layer_a,
    )
    assert splits_next == 8
    assert workspace_next is not None
    assert set(t.data_ptr() for t in workspace_next).isdisjoint(
        t.data_ptr() for t in workspace_a
    )

    # Distinct QSA layers must not alias even when their bucket shapes match.
    _, workspace_b = ops.qsa_prepare_split_workspace(
        case_a["q"], case_a["k"], case_a["indices"].shape[1], layer_b
    )
    assert workspace_b is not None
    assert set(t.data_ptr() for t in workspace_b).isdisjoint(
        t.data_ptr() for t in workspace_a
    )
