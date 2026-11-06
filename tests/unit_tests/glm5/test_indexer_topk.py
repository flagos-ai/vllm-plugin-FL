# SPDX-License-Identifier: Apache-2.0
"""Independent range/selection checks for the existing-Torch TopK adapter."""

import os

import pytest
import torch

from vllm_fl.dispatch.policy import SelectionPolicy, policy_context
from vllm_fl.models.glm5 import indexer_backend, indexer_topk


def _check(logits, starts, ends, output, k):
    values = logits.cpu()
    for row, (start, end) in enumerate(zip(starts.cpu().tolist(), ends.cpu().tolist())):
        end = min(end, logits.shape[1])
        valid = list(range(max(start, 0), max(end, 0)))
        selected = output[row].cpu()
        selected = selected[selected >= 0].long()
        assert selected.numel() == min(k, len(valid))
        assert selected.unique().numel() == selected.numel()
        assert selected.tolist() == sorted(selected.tolist())
        assert all(start <= index + start < end for index in selected.tolist())
        expected = sorted((float(values[row, index]) for index in valid), reverse=True)[
            :k
        ]
        actual = sorted(
            (float(values[row, index + start]) for index in selected), reverse=True
        )
        assert actual == expected
        assert (output[row, selected.numel() :] == -1).all()


@pytest.mark.parametrize("k", [17, 97])
@pytest.mark.parametrize("all_infinite", [False, True])
def test_prefill_ranges_padding_strides_and_infinite_ties(k, all_infinite):
    torch.manual_seed(51)
    backing = torch.randn(35, 166)
    logits = backing[:, ::2]
    if all_infinite:
        logits.fill_(-torch.inf)
    before = backing.clone()
    rows = 33
    starts = torch.arange(rows) % 7
    ends = starts + torch.tensor([0, 1, 10, 42, 80] * 7)[:rows]
    storage = torch.full((35, k + 3), -9, dtype=torch.int32)
    output = storage[:, :k]
    indexer_topk.topk_prefill(logits, starts, ends, output, rows, *logits.stride(), k)
    _check(logits, starts, ends, output[:rows], k)
    assert (storage[rows:] == -9).all()
    assert (storage[:rows, k:] == -9).all()
    torch.testing.assert_close(backing, before)


@pytest.mark.parametrize("next_n", [1, 3])
def test_decode_request_lengths_include_each_mtp_offset(next_n):
    torch.manual_seed(13)
    lengths = torch.tensor([[0], [1], [13], [58]], dtype=torch.int32)
    rows = lengths.numel() * next_n
    logits = torch.randn(rows, 64)
    logits[2].fill_(-torch.inf)
    output = torch.empty(rows, 17, dtype=torch.int32)
    indexer_topk.topk_decode(
        logits, next_n, lengths, output, rows, *logits.stride(), 17
    )
    ends = torch.tensor(
        [
            int(length) - next_n + offset + 1
            for length in lengths.flatten()
            for offset in range(next_n)
        ]
    )
    _check(logits, torch.zeros_like(ends), ends, output, 17)


def test_topk_policy_rejection_precedes_execution(monkeypatch):
    backend = indexer_backend.Glm5NextIndexerBackend()
    backend.use_torch_topk = True
    monkeypatch.setattr(
        backend, "_flag", lambda *a: pytest.fail("resolved broken Triton TopK")
    )
    with (
        policy_context(
            SelectionPolicy.from_dict(per_op_order={"top_k_per_row_decode": ["flagos"]})
        ),
        pytest.raises(RuntimeError, match="No implementation selected"),
    ):
        backend.topk_decode(*([None] * 8), _preflight=True)


@pytest.mark.parametrize(
    "vendor,fp8,expected",
    [
        ("kunlunxin", False, True),
        ("metax", False, True),
        ("nvidia", False, False),
        ("nvidia", True, False),
        ("kunlunxin", True, False),
    ],
)
def test_topk_compiler_capability_selects_before_execution(
    monkeypatch, vendor, fp8, expected
):
    monkeypatch.setattr(indexer_backend.current_platform, "vendor_name", vendor)
    monkeypatch.setattr(indexer_backend.current_platform, "supports_fp8", lambda: fp8)
    backend = indexer_backend.Glm5NextIndexerBackend()
    assert backend.use_torch_topk is expected
    if expected:
        monkeypatch.setattr(
            backend, "_flag", lambda *args: pytest.fail("resolved unsupported radix")
        )
        with policy_context(SelectionPolicy()):
            for method in (backend.topk_prefill, backend.topk_decode):
                description = method(*([None] * 8), _preflight=True)
                assert description["selected"] == "glm5.torch"


def test_topk_execution_error_propagates_once(monkeypatch):
    backend = indexer_backend.Glm5NextIndexerBackend()
    backend.use_torch_topk = True
    error = RuntimeError("existing TopK operation failed")
    calls = []

    def broken(*args, **kwargs):
        calls.append(True)
        raise error

    monkeypatch.setattr(torch, "topk", broken)
    logits = torch.randn(2, 13)
    output = torch.empty(2, 3, dtype=torch.int32)
    with pytest.raises(RuntimeError) as caught:
        backend.topk_decode(
            logits, 1, torch.tensor([5, 13]), output, 2, *logits.stride(), 3
        )
    assert caught.value is error
    assert calls == [True]


@pytest.mark.gpu
@pytest.mark.parametrize("decode", [False, True])
def test_topk_backend_eager_and_refreshed_graph(monkeypatch, decode, request):
    device_env = request.config.getoption("test_device_env", default="")
    if device_env:
        assert len(os.environ[device_env].split(",")) == 1
        assert indexer_backend.current_platform.torch_device_fn.device_count() == 1
    backend = indexer_backend.Glm5NextIndexerBackend()
    backend.use_torch_topk = True
    monkeypatch.setattr(
        backend, "_flag", lambda *a: pytest.fail("resolved broken Triton TopK")
    )
    torch.manual_seed(39)
    logits = torch.randn(3, 64, device="cuda")
    starts = torch.tensor([0, 7, 3], device="cuda", dtype=torch.int32)
    ends = torch.tensor([0, 11, 64], device="cuda", dtype=torch.int32)
    output = torch.empty(3, 17, device="cuda", dtype=torch.int32)

    def run():
        if decode:
            backend.topk_decode(
                logits, 1, ends[:, None], output, 3, *logits.stride(), 17
            )
        else:
            backend.topk_prefill(logits, starts, ends, output, 3, *logits.stride(), 17)

    for _ in range(3):
        run()
    reference_starts = torch.zeros_like(starts) if decode else starts
    _check(logits, reference_starts, ends, output, 17)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    logits.mul_(-1)
    ends.copy_(torch.tensor([1, 40, 15], device="cuda", dtype=torch.int32))
    output.fill_(-77)
    graph.replay()
    _check(logits, reference_starts, ends, output, 17)
    description = next(iter(backend._bindings.values())).describe()
    assert description["selected"] == "glm5.torch"
    assert description["graph_capabilities"]["glm5.torch"] is True
