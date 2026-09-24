"""Exact short-context selection without data-dependent graph branches."""

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.models.qwen3_8_flash_next.gpu import indexer_qsa
from vllm_fl.models.qwen3_8_flash_next.gpu.ops import qsa as ops


@pytest.fixture
def make_indexer(monkeypatch):
    monkeypatch.setattr(
        indexer_qsa, "ReplicatedLinear", lambda *a, **k: torch.nn.Identity()
    )
    monkeypatch.setattr(
        indexer_qsa, "GemmaRMSNorm", lambda *a, **k: torch.nn.Identity()
    )
    for name in ["QSAKeyStateCache", "QSACompressedKeyCache"]:
        monkeypatch.setattr(
            indexer_qsa, name, lambda **k: SimpleNamespace(prefix=k["prefix"])
        )

    def build(max_len, transfer=None):
        config = SimpleNamespace(
            indexer_n_heads=4,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=2048,
            indexer_compress_ratio=4,
            hidden_size=2560,
        )
        runtime = SimpleNamespace(
            cache_config=SimpleNamespace(),
            kv_transfer_config=transfer,
            model_config=SimpleNamespace(
                dtype=torch.bfloat16, max_model_len=max_len, uses_mrope=False
            ),
        )
        rotary = torch.nn.Identity()
        rotary.rotary_dim, rotary.is_neox_style = 64, True
        return indexer_qsa.QSAIndexer(
            vllm_config=runtime,
            config=config,
            layer_id=0,
            rotary_emb=rotary,
        )

    return build


@pytest.mark.parametrize("limit", [1, 2047, 2048, 2049, 131072])
def test_selection_gate_uses_worker_context_limit(make_indexer, limit):
    indexer = make_indexer(limit)
    assert indexer.select_all_tokens is (limit <= 2048)
    status = indexer.runtime_status()
    assert status["stages"]["selection"]["callable"].endswith(
        ".qsa_select_all_paged_tokens" if limit <= 2048 else ".qsa_select_paged_tokens"
    )
    if limit <= 2048:
        assert set(status["stages"]) == {"metadata", "selection", "attention"}
        assert status["compression_candidates"] == []


def test_kv_export_keeps_side_cache_state(make_indexer):
    assert make_indexer(2048, transfer=object()).select_all_tokens is False


def test_short_batch_in_long_context_worker_retains_scoring(make_indexer, monkeypatch):
    indexer = make_indexer(131072)
    calls = []
    metadata = SimpleNamespace(num_actual_tokens=1)
    monkeypatch.setattr(indexer, "_metadata", lambda: (metadata, metadata))

    def project(*a):
        calls.append("project")
        return torch.zeros(1, 4, 128), torch.zeros(1, 1, 128)

    monkeypatch.setattr(indexer, "project_qk", project)
    monkeypatch.setattr(
        indexer, "_update_and_compress", lambda *a: calls.append("compress")
    )
    expected = torch.full((1, 2051), -1, dtype=torch.int32)
    monkeypatch.setattr(
        indexer, "_select", lambda *a: (calls.append("score"), expected)[1]
    )
    assert indexer(torch.zeros(1, 2560), torch.zeros(1, dtype=torch.long)) is expected
    assert calls == ["project", "compress", "score"]


@pytest.mark.parametrize("limit", [0, 2049, 131072])
def test_all_visible_rejects_context_outside_budget(limit):
    with pytest.raises(ValueError, match="within budget"):
        ops.qsa_select_all_paged_tokens(
            torch.tensor([0]), torch.tensor([1]), torch.tensor([0]), 2048, 4, limit
        )


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 8, 16, 24, 32, 33, 40, 48, 56, 64, 128])
def test_graph_indices_and_attention_equal_scored_path(rows):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(171)
    device = "cuda"
    # Deliberately larger physical score capacity than logical context.
    key = torch.randn(400, 16, 1, 128, dtype=torch.bfloat16, device=device)
    q = torch.randn(rows, 4, 128, dtype=torch.bfloat16, device=device)
    table = torch.arange(400, device=device, dtype=torch.int32).reshape(4, 100)
    req = torch.arange(rows, device=device, dtype=torch.int32) % 4
    positions = torch.zeros(rows, dtype=torch.int64, device=device)
    lengths = torch.full((4,), 2048, dtype=torch.int32, device=device)
    out = torch.empty(rows, 2051, dtype=torch.int32, device=device)
    attn_q = torch.randn(rows, 3, 256, dtype=torch.bfloat16, device=device)
    attn_k = torch.randn(400, 64, 1, 256, dtype=torch.bfloat16, device=device)
    attn_v = torch.randn_like(attn_k)
    attn_out = torch.empty_like(attn_q)
    _, workspace = ops.qsa_prepare_split_workspace(attn_q, attn_k, 2051, {})

    def fast():
        ops.qsa_select_all_paged_tokens(positions, lengths, req, 2048, 4, 2048, out)
        ops.qsa_sparse_paged_attention(
            attn_q,
            attn_k,
            attn_v,
            out,
            table,
            req,
            out=attn_out,
            split_workspace=workspace,
        )

    fast()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fast()
    for length in [1, 3, 4, 5, 511, 512, 513, 2047, 2048]:
        lengths.copy_(
            torch.tensor(
                [0, min(7, length), length, 2048], dtype=torch.int32, device=device
            )
        )
        req.copy_((torch.arange(rows, device=device, dtype=torch.int32) + length) % 4)
        positions.copy_(torch.clamp(lengths[req].to(torch.int64) - 1, min=-1))
        if rows > 1:
            req[-1] = -1  # Graph padding has no request or visible token.
        for tied in [False, True]:
            q.zero_() if tied else q.normal_()
            expected = ops.qsa_select_paged_tokens(
                q, key, table, req, positions, lengths, 2048, 4
            )
            # Independent semantic reference: within this lifetime bound every
            # causal token is selected in ascending order, with -1 padding.
            columns = torch.arange(2051, device=device)[None, :]
            visible = torch.minimum(
                positions + 1, lengths[req.clamp_min(0)].to(torch.int64)
            ).clamp_min(0)
            reference = torch.where(
                (req[:, None] >= 0) & (columns < visible[:, None]), columns, -1
            ).to(torch.int32)
            assert torch.equal(expected, reference)
            expected_attn = ops.qsa_sparse_paged_attention(
                attn_q,
                attn_k,
                attn_v,
                expected,
                table,
                req,
                split_workspace=workspace,
            )
            for _ in range(10):
                graph.replay()
                assert torch.equal(out, expected)
                assert torch.equal(attn_out, expected_attn)


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [0, 1024, 16384])
def test_all_visible_prefill_and_empty_batches(rows):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    positions = torch.arange(rows, device="cuda") % 2048
    lengths = torch.tensor([2048], device="cuda", dtype=torch.int32)
    req = torch.zeros(rows, device="cuda", dtype=torch.int32)
    result = ops.qsa_select_all_paged_tokens(positions, lengths, req, 2048, 4, 2048)
    columns = torch.arange(2051, device="cuda")[None, :]
    expected = torch.where(columns <= positions[:, None], columns, -1).to(torch.int32)
    assert torch.equal(result, expected)


@pytest.mark.gpu
def test_fast_indexer_prepares_request_map_before_skipping_projection(
    make_indexer, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    indexer = make_indexer(2048)
    metadata = SimpleNamespace(
        num_actual_tokens=2,
        logical_positions=torch.tensor([3, 0], device="cuda"),
        seq_lens=torch.tensor([1, 4], dtype=torch.int32, device="cuda"),
        token_to_req=torch.zeros(2, dtype=torch.int32, device="cuda"),
    )

    def prepare():
        metadata.token_to_req.copy_(
            torch.tensor([1, 0], dtype=torch.int32, device="cuda")
        )
        return metadata, metadata

    def unwanted(*a):
        raise AssertionError("projection/compression/scoring must be skipped")

    monkeypatch.setattr(indexer, "_metadata", prepare)
    for name in ["project_qk", "_update_and_compress", "_select"]:
        monkeypatch.setattr(indexer, name, unwanted)
    out = torch.empty(2, 2051, dtype=torch.int32, device="cuda")
    assert (
        indexer(torch.zeros(2, 2560, device="cuda"), metadata.logical_positions, out)
        is out
    )
    assert out[0, :5].tolist() == [0, 1, 2, 3, -1]
    assert out[1, :2].tolist() == [0, -1]
