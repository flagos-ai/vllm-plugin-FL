"""Repetition incident: deterministic membership AND sparse reduction order."""

import pytest
import torch

from vllm_fl.models.qwen3_8_flash_next.gpu.ops import qsa as ops


def reference(scores, visible, k):
    result = torch.full((scores.shape[0], k), -1, dtype=torch.int32)
    for row, n in enumerate(visible.tolist()):
        candidates = sorted(range(n), key=lambda i: (-float(scores[row, i]), i))[:k]
        result[row, : len(candidates)] = torch.tensor(
            sorted(candidates), dtype=torch.int32
        )
    return result


@pytest.mark.parametrize(
    "columns,k", [(0, 4), (3, 4), (4, 4), (5, 4), (513, 512), (1024, 512)]
)
def test_selection_exact_ties_padding_and_small_capacity(columns, k):
    scores = torch.zeros(3, columns)
    visible = torch.tensor([0, min(columns, 3), columns], dtype=torch.int32)
    if columns:
        scores[-1, -1] = 1
    for i, n in enumerate(visible):
        scores[i, int(n) :] = -torch.inf
    actual = ops._qsa_deterministic_block_topk(scores, visible, k)
    assert torch.equal(actual, reference(scores, visible, k))


def test_selection_does_not_perturb_close_scores():
    # Adding an index epsilon to FP32 scores would incorrectly reverse these.
    high = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
    scores = torch.tensor([[1.0, 1.0, high, 1.0]])
    selected = ops._qsa_deterministic_block_topk(scores, torch.tensor([4]), 2)
    assert selected.tolist() == [[0, 2]]


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8, 16, 33])
def test_qsa_selection_graph_replay_budget_boundary(rows):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(54)
    # Real indexer geometry; 513 groups is the first step past token budget.
    device = "cuda"
    key = torch.randn(40, 16, 1, 128, dtype=torch.bfloat16, device=device)
    q = torch.randn(rows, 4, 128, dtype=torch.bfloat16, device=device)
    table = torch.randperm(40, device=device).to(torch.int32)[None]
    req = torch.zeros(rows, dtype=torch.int32, device=device)
    positions = torch.full((rows,), 2051, dtype=torch.int64, device=device)
    lengths = torch.tensor([2052], dtype=torch.int32, device=device)
    out = torch.empty(rows, 2051, dtype=torch.int32, device=device)
    attn_q = torch.randn(rows, 3, 256, dtype=torch.bfloat16, device=device)
    attn_k = torch.randn(40, 64, 1, 256, dtype=torch.bfloat16, device=device)
    attn_v = torch.randn_like(attn_k)
    attn_out = torch.empty_like(attn_q)
    # Match the model-owned workspace contract in both modes. Direct eager
    # allocation vs capture fallback would compare split vs unsplit kernels.
    _, workspace = ops.qsa_prepare_split_workspace(attn_q, attn_k, 2051, {})

    def select():
        return ops.qsa_select_paged_tokens(
            q, key, table, req, positions, lengths, 2048, 4, out
        )

    def forward():
        select()
        ops.qsa_sparse_paged_attention(
            attn_q, attn_k, attn_v, out, table, req,
            out=attn_out, split_workspace=workspace,
        )

    forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    for length in (2047, 2048, 2051, 2052, 2053, 2304):
        positions.fill_(length - 1)
        lengths.fill_(length)
        for tied in (False, True):
            if tied:
                q.zero_()
            else:
                q.normal_()
            scores, visible = ops.qsa_mqa_paged(q, key, table, req, positions, lengths, 4)
            expected_blocks = reference(scores.cpu(), visible.cpu(), 512).to(device)
            expected = ops.expand_qsa_block_indices(
                expected_blocks, positions, lengths, req, 4, 2048
            )
            forward()
            assert torch.equal(out, expected)
            expected_attention = attn_out.clone()
            for _ in range(10):
                graph.replay()
                assert torch.equal(out, expected)
                assert torch.equal(attn_out, expected_attention)
