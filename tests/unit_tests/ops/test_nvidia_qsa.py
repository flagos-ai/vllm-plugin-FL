# Copyright (c) 2026 BAAI. All rights reserved.

"""Numerical, paging and graph contracts for Hopper QSA scores."""

from types import SimpleNamespace

import pytest
import torch

from tests.utils.device_utils import get_backend, requires_cuda_fp8

pytestmark = pytest.mark.skipif(get_backend() != "nvidia", reason="NVIDIA QSA")


def _qwen_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(architectures=["Qwen4ExpForConditionalGeneration"])
    )


def _inputs(rows, strided=False, dtype=torch.bfloat16, requests=1, width=128):
    gen = torch.Generator(device="cpu").manual_seed(618)
    q = torch.randn(rows, 4 + int(strided), 128, generator=gen).to(dtype).cuda()[:, :4]
    pages = width // 16
    cache = (
        torch.randn(requests * pages, 16, 1 + int(strided), 128, generator=gen)
        .to(dtype)
        .cuda()[:, :, :1]
    )
    # Reversed pages make a contiguous-cache assumption observably incorrect.
    table = (
        torch.arange(requests * pages - 1, -1, -1, dtype=torch.int32)
        .view(requests, pages)
        .cuda()
    )
    return q, cache, table


def _reference(q, cache, table, starts, lengths, offset=0, count=None):
    q, cache, table = q.cpu().float(), cache.cpu().float(), table.cpu().long()
    starts, lengths = starts.cpu().tolist(), lengths.cpu().tolist()
    count = q.shape[0] - offset if count is None else count
    width = table.shape[1] * cache.shape[1]
    out = torch.full((count, width), -777.0, dtype=torch.float32)
    base = starts[0]
    for req in range(table.shape[0]):
        keys = cache[table[req]].reshape(width, cache.shape[-1])
        begin, end = (
            max(starts[req] - base, offset),
            min(starts[req + 1] - base, offset + count),
        )
        if end > begin:
            scores = torch.einsum("rhd,kd->rkh", q[begin:end], keys).relu().sum(-1)
            for i, row in enumerate(range(begin, end)):
                n = max(0, min(lengths[row], width))
                out[row - offset, :n] = scores[i, :n]
    return out


def _check(actual, expected, lengths):
    actual = actual.cpu()
    for row, n in enumerate(lengths.cpu().tolist()):
        n = max(0, min(n, actual.shape[1]))
        torch.testing.assert_close(
            actual[row, :n], expected[row, :n], rtol=2e-5, atol=2e-4
        )


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [16, 64, 512])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float16,
        pytest.param(torch.float8_e4m3fn, marks=requires_cuda_fp8),
    ],
)
def test_prefill_scores_match_cpu(rows, strided, dtype):
    from vllm_fl.ops.nvidia_qsa import prefill_logits

    q, cache, table = _inputs(rows, strided, dtype)
    starts = torch.tensor([17, 17 + rows], dtype=torch.int32, device="cuda")
    lengths = (torch.arange(1, rows + 1, dtype=torch.int32) // 4).cuda()
    workspace = torch.full((rows * 128 + 32,), -777.0, device="cuda")
    logits = prefill_logits(
        q, cache, table, starts, lengths, workspace, rows, 128, 0, rows
    )
    torch.cuda.synchronize()
    expected = _reference(q, cache, table, starts, lengths)
    _check(logits, expected, lengths)
    assert logits.data_ptr() == workspace.data_ptr()
    # No writes outside visible columns or the requested workspace slice.
    mask = torch.arange(128)[None, :] >= lengths.cpu()[:, None]
    assert torch.all(logits.cpu()[mask] == -777.0)
    assert torch.all(workspace[-32:] == -777.0)


@pytest.mark.gpu
def test_prefill_multiple_requests_and_partial_chunk():
    from vllm_fl.ops.nvidia_qsa import prefill_logits

    q, cache, table = _inputs(17, True, requests=4)
    starts = torch.tensor([37, 40, 40, 49, 54], dtype=torch.int32, device="cuda")
    lengths = torch.tensor(
        [0, 1, 7, 3, 22, 31, 63, 65, 79, 95, 110, 127, 2, 5, 9, 13, 17],
        dtype=torch.int32,
        device="cuda",
    )
    workspace = torch.full((12 * 128,), -777.0, device="cuda")
    logits = prefill_logits(q, cache, table, starts, lengths, workspace, 9, 128, 2, 12)
    _check(logits, _reference(q, cache, table, starts, lengths, 2, 12), lengths[2:14])


@pytest.mark.gpu
@pytest.mark.parametrize("decode_len", [1, 2, 4])
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, pytest.param(torch.float8_e4m3fn, marks=requires_cuda_fp8)],
)
def test_decode_scores_match_cpu(decode_len, dtype):
    from vllm_fl.ops.nvidia_qsa import decode_logits

    rows = 3 * decode_len
    q, cache, table = _inputs(rows, True, dtype, requests=3)
    starts = torch.arange(0, rows + 1, decode_len, dtype=torch.int32, device="cuda")
    lengths = torch.arange(rows, dtype=torch.int32).mul(13).add(1).clamp(max=128).cuda()
    logits = decode_logits(q, cache, table, lengths, decode_len)
    _check(logits, _reference(q, cache, table, starts, lengths), lengths)


def _check_indices(ids, scores, lengths, k=512):
    ids, scores = ids.cpu().long(), scores.cpu()
    for row, n in enumerate(lengths.cpu().tolist()):
        count = min(n, k)
        actual = ids[row, :count]
        assert actual.unique().numel() == count
        assert torch.all((actual >= 0) & (actual < n))
        assert torch.all(ids[row, count:] == -1)
        if n > k:
            torch.testing.assert_close(
                scores[row, actual].sort().values,
                scores[row, :n].topk(k).values.sort().values,
                rtol=2e-5,
                atol=2e-4,
            )


@pytest.mark.gpu
def test_decode_topk_and_graph_updates_inputs():
    from vllm_fl.ops.nvidia_qsa import select_paged_decode
    from vllm_fl.ops.nvidia_topk_ops import register_nvidia_topk_ops

    assert register_nvidia_topk_ops()
    q, cache, table = _inputs(2, True, requests=2, width=1024)
    lengths = torch.tensor([17, 1024], dtype=torch.int32, device="cuda")
    ids = torch.empty(2, 512, dtype=torch.int32, device="cuda")

    def run():
        select_paged_decode(q, cache, table, lengths, 2048, 4, 1, ids)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    starts = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    _check_indices(ids, _reference(q, cache, table, starts, lengths), lengths)
    lengths.copy_(torch.tensor([1024, 0], dtype=torch.int32, device="cuda"))
    q.neg_()
    table.copy_(table.flip(1))
    graph.replay()
    _check_indices(ids, _reference(q, cache, table, starts, lengths), lengths)


@pytest.mark.gpu
def test_prefill_installed_upstream_chunking_and_graph(monkeypatch):
    from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer

    from vllm_fl.nvidia_compat import has_native_vllm_kernels
    from vllm_fl.ops.nvidia_qsa import install_nvidia_qsa, prefill_logits
    from vllm_fl.ops.nvidia_topk_ops import register_nvidia_topk_ops

    if has_native_vllm_kernels() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("QSA compatibility installation is specific to EMPTY Hopper")
    assert register_nvidia_topk_ops()
    # Restore both attributes after exercising the real production installer.
    monkeypatch.setattr(qsa_indexer, "_prefill_logits", qsa_indexer._prefill_logits)
    monkeypatch.setattr(
        qsa_indexer, "qsa_select_paged_decode", qsa_indexer.qsa_select_paged_decode
    )
    assert install_nvidia_qsa(_qwen_config())
    assert install_nvidia_qsa(_qwen_config())
    assert qsa_indexer._prefill_logits is prefill_logits
    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "1")
    q, cache, table = _inputs(270, True, width=1024)
    lengths = torch.full((270,), 1024, dtype=torch.int32, device="cuda")
    starts = torch.tensor([19, 289], dtype=torch.int32, device="cuda")
    ids = torch.empty(270, 512, dtype=torch.int32, device="cuda")

    def run():
        qsa_indexer.qsa_select_paged_prefill(
            q, cache, table, starts, lengths, 2048, 4, 270, ids, 4096
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    lengths[::2] = 0
    q.neg_()
    table.copy_(table.flip(1))
    graph.replay()
    _check_indices(ids, _reference(q, cache, table, starts, lengths), lengths)


@pytest.mark.parametrize("boundary", ["other_model", "native", "triton37"])
def test_installer_leaves_unaffected_paths_unchanged(monkeypatch, boundary):
    from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer

    from vllm_fl import nvidia_compat
    from vllm_fl.ops import nvidia_qsa

    cfg = _qwen_config()
    if boundary == "other_model":
        cfg.model_config.architectures = ["Qwen3ForCausalLM"]
    elif boundary == "native":
        monkeypatch.setattr(nvidia_compat, "has_native_vllm_kernels", lambda: True)
    else:
        monkeypatch.setattr(nvidia_qsa.triton, "__version__", "3.7.1")
    before = (qsa_indexer._prefill_logits, qsa_indexer.qsa_select_paged_decode)
    assert not nvidia_qsa.install_nvidia_qsa(cfg)
    assert (qsa_indexer._prefill_logits, qsa_indexer.qsa_select_paged_decode) == before
