# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Independent numerical regressions for the TP-local attention/indexer kernels."""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires an accelerator"),
]
ROOT = Path(__file__).resolve().parents[2]
IMPL = ROOT / "vllm_fl/dispatch/backends/vendor/metax/impl/dsv4"


@pytest.fixture(scope="module")
def kernels():
    # Load the actual numerical modules without installing model startup hooks.
    name = "_dsv4_numeric_test"
    package = types.ModuleType(name)
    package.__path__ = [str(IMPL)]
    sys.modules[name] = package
    attention = {
        part: importlib.import_module(f"{name}.attention.{part}")
        for part in ("prefill", "staged", "decode")
    }
    spec = importlib.util.spec_from_file_location(
        name + "_topk", IMPL / "indexer/tree_topk.py"
    )
    topk = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = topk
    spec.loader.exec_module(topk)
    yield attention, topk
    for key in list(sys.modules):
        if key.startswith(name):
            del sys.modules[key]


def assert_topk(logits, lengths, result, k):
    """Compare the score multiset; torch.topk does not specify tie index order."""
    for row, length in enumerate(lengths.tolist()):
        ids = result[row]
        count = min(length, k)
        valid_ids = ids[ids >= 0].long()
        assert len(valid_ids) == count
        assert len(valid_ids.unique()) == count
        assert (valid_ids < length).all()
        expected = logits[row, :length].topk(count).values.sort().values
        actual = logits[row, valid_ids].sort().values
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("width", [513, 1024, 2049, 8192])
@pytest.mark.parametrize(
    "pattern", ["ties", "reduce_ties", "equal", "random", "minus_inf"]
)
def test_tree_topk_ties_and_lengths(kernels, width, pattern):
    _, topk = kernels
    torch.manual_seed(541)
    logits = torch.full((6, width + 7), -1.0, device="cuda")[:, :width]
    if pattern == "random":
        logits.copy_(torch.randn_like(logits))
    elif pattern == "minus_inf":
        logits.fill_(-torch.inf)
    else:
        logits[:, :512] = 0
        if pattern == "reduce_ties":
            logits[:, 1024 : min(width, 1536)] = 0
            logits[:, -1] = 1
        elif pattern == "ties":
            logits[:, 512] = 1
        else:
            logits.fill_(0)
    lengths = torch.tensor(
        [0, 1, 511, 512, 513, width], device="cuda", dtype=torch.int32
    )
    out = torch.full((6, 512), -99, device="cuda", dtype=torch.int32)
    topk.decode_topk_tree(logits, lengths, out)
    assert_topk(logits, lengths, out, 512)


def test_tree_topk_graph_refresh(kernels):
    _, topk = kernels
    x = torch.zeros((4, 4096), device="cuda")
    lengths = torch.tensor([0, 513, 2049, 4096], device="cuda", dtype=torch.int32)
    out = torch.empty((4, 512), device="cuda", dtype=torch.int32)

    def call():
        topk.decode_topk_tree(x, lengths, out)

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for offset in (512, 3072):
        x.fill_(-1)
        x[:, :512] = 0
        x[:, offset] = 10
        lengths.copy_(torch.tensor([4096, 0, 511, 2049], device="cuda"))
        out.fill_(-99)
        graph.replay()
        torch.cuda.synchronize()
        assert_topk(x, lengths, out, 512)


@pytest.mark.parametrize(
    "score,sink",
    [
        (100, 0),
        (-100, -100),
        (-100, 0),
        (10000, 10000),
        (-10000, -10000),
        (2, float("inf")),
        (2, -float("inf")),
        (2, None),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_decode_reduce_stable_sink(kernels, score, sink, empty):
    attention, _ = kernels
    device = "cuda"
    logits = torch.full((4,), score, dtype=torch.float64, device=device)
    v = torch.linspace(-1, 1, 4 * 512, device=device).reshape(4, 512).double()
    if empty:
        pm = torch.full((1, 4, 16), -torch.inf, device=device)
        ps = torch.zeros_like(pm)
        pa = torch.zeros((1, 4, 16, 512), device=device)
        expected = torch.zeros((512,), device=device)
    else:
        pm = logits.float()[None, :, None].expand(1, 4, 16).contiguous()
        ps = torch.ones_like(pm)
        pa = v.float()[None, :, None, :].expand(1, 4, 16, 512).contiguous()
        if sink is None:
            weights = logits.softmax(0)
        elif sink == float("inf"):
            weights = torch.zeros_like(logits)
        else:
            weights = torch.cat((logits, logits.new_tensor([sink]))).softmax(0)[:4]
        expected = (weights[:, None] * v).sum(0).float()
    s = torch.full((16,), sink or 0, device=device, dtype=torch.float32)
    out = torch.full((1, 1, 16, 512), torch.nan, device=device, dtype=torch.bfloat16)
    lse = torch.empty((1, 16, 1), device=device)
    attention["decode"]._reduce_split_kernel[(16, 2)](
        pm,
        ps,
        pa,
        s,
        out,
        lse,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        1,
        16,
        4,
        sink is not None,
        256,
        num_warps=4,
        num_stages=1,
    )
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        out[0, 0].float(), expected.expand(16, -1), rtol=0.004, atol=0.004
    )
    expected_lse = -torch.inf if empty else float(torch.logsumexp(logits, 0))
    torch.testing.assert_close(
        lse, torch.full_like(lse, expected_lse), rtol=1e-6, atol=1e-5
    )


def attention_reference(q, kv, ids, lens, sink):
    result = torch.zeros((len(q), 8, 512), device=q.device, dtype=torch.float64)
    for t in range(len(q)):
        valid = ids[t, 0, : int(lens[t])].long()
        valid = valid[(valid >= 0) & (valid < len(kv))]
        if not len(valid):
            continue
        values = kv[valid, 0].double()
        scores = q[t, :8].double() @ values.T
        if sink is not None:
            extra = sink[:8, None].double()
            if torch.isposinf(extra).any():
                finite = torch.where(torch.isposinf(extra), 0, extra)
                p = torch.cat((scores, finite), 1).softmax(1)[:, :-1]
                p = torch.where(torch.isposinf(extra), 0, p)
            else:
                p = torch.cat((scores, extra), 1).softmax(1)[:, :-1]
        else:
            p = scores.softmax(1)
        result[t] = p @ values
    return result.float()


@pytest.mark.parametrize("variant", ["prefill", "staged"])
@pytest.mark.parametrize(
    "score,sink_value",
    [
        (100, 0),
        (-100, -100),
        (-100, 0),
        (10000, 10000),
        (-10000, -10000),
        (2, float("inf")),
        (2, -float("inf")),
        (2, None),
    ],
)
def test_prefill_sink_empty_blocks_and_graph(kernels, variant, score, sink_value):
    attention, _ = kernels
    q = torch.zeros((4, 16, 512), device="cuda", dtype=torch.bfloat16)
    q[:, :, 0] = score
    kv = torch.linspace(-1, 1, 40 * 512, device="cuda").reshape(40, 1, 512).bfloat16()
    kv[:, :, 0] = 1
    ids = torch.arange(33, device="cuda", dtype=torch.int32).repeat(4, 1, 1)
    ids[1, :, :16] = -1
    ids[2] = -1
    lens = torch.tensor([33, 33, 33, 0], device="cuda", dtype=torch.int32)
    sink = (
        None
        if sink_value is None
        else torch.full((16,), sink_value, device="cuda", dtype=torch.float32)
    )
    out = torch.empty((4, 16, 512), device="cuda", dtype=torch.bfloat16)
    ml = torch.empty((4, 16), device="cuda")
    lse = torch.empty_like(ml)
    work = attention["staged"].buffers(q, 33)

    def call():
        if variant == "staged":
            attention["staged"].run(
                q, kv, ids, sink, lens, out, ml, lse, work, sm_scale=1.0
            )
        else:
            attention["prefill"].flash_mla_sparse_fwd(
                q,
                kv,
                ids,
                1.0,
                attn_sink=sink,
                topk_length=lens,
                out=out,
                max_logits_out=ml,
                lse_out=lse,
                real_num_heads=8,
            )

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for scale in (1, -1):
        q[:, :, 0] = score * scale
        lens.copy_(torch.tensor([33, 33, 33, 0], device="cuda"))
        out.fill_(torch.nan)
        ml.fill_(torch.nan)
        lse.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        expected = attention_reference(q, kv, ids, lens, sink)
        assert torch.isfinite(out[:, :8]).all()
        torch.testing.assert_close(out[:, :8].float(), expected, rtol=0.008, atol=0.008)
        assert torch.isposinf(lse[2:]).all()


@pytest.mark.parametrize("score", [100.0, -100.0])
def test_decode_packed_cache_split_and_graph(kernels, score):
    attention, _ = kernels
    decode = attention["decode"]
    q = torch.zeros((4, 1, 16, 512), device="cuda", dtype=torch.bfloat16)
    q[:, :, :, 0] = score
    values = torch.ones((256, 1, 512), device="cuda", dtype=torch.bfloat16)
    # Independent byte packing: 448 FP8 NOPE bytes + 64 BF16 ROPE values per
    # token, followed by the page's 8 exponent-scale bytes per token.
    cache = torch.zeros((1, 256, 584), device="cuda", dtype=torch.uint8)
    data = cache.flatten()[: 256 * 576].view(256, 576)
    data[:, :448].copy_(
        values[:, 0, :448].contiguous().to(torch.float8_e4m3fn).view(torch.uint8)
    )
    data[:, 448:].copy_(values[:, 0, 448:].contiguous().view(torch.uint8))
    cache.flatten()[256 * 576 :].fill_(127)
    ids = torch.arange(33, device="cuda", dtype=torch.int32).repeat(4, 1, 1)
    ids[1, :, :16] = -1
    ids[2] = -1
    lens = torch.tensor([33, 33, 33, 0], device="cuda", dtype=torch.int32)
    extra_ids = torch.zeros((4, 1, 1), device="cuda", dtype=torch.int32)
    extra_lens = torch.tensor([1, 0, 0, 0], device="cuda", dtype=torch.int32)
    sink = torch.zeros(16, device="cuda")
    out = torch.empty_like(q)
    lse = torch.empty((4, 16, 1), device="cuda")

    def call():
        assert decode._try_sparse_decode_model1_split(
            q,
            cache,
            ids,
            out,
            lse,
            sink,
            lens,
            cache,
            extra_ids,
            extra_lens,
            4,
            1,
            16,
            512,
            512,
            33,
            256,
            1.0,
            real_num_heads=8,
        )

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    all_ids = torch.cat((ids, extra_ids), -1)
    # The reference concatenates only valid prefixes, rather than treating an
    # unused extra slot as another token.
    all_ids[1:, :, -1] = -1
    reference_lens = torch.tensor([34, 34, 34, 0], device="cuda")
    for updated_score in (score, -score):
        q[:, :, :, 0] = updated_score
        out.fill_(torch.nan)
        lse.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        expected = attention_reference(q[:, 0], values, all_ids, reference_lens, sink)
        assert torch.isfinite(out[:, 0, :8]).all()
        torch.testing.assert_close(
            out[:, 0, :8].float(), expected, rtol=0.004, atol=0.004
        )


@pytest.mark.parametrize("setting", [None, False, True])
def test_deepgemm_mode_selection(kernels, monkeypatch, setting):
    import deep_gemm

    module = importlib.import_module("_dsv4_numeric_test.indexer.deep_gemm")
    monkeypatch.setattr(module, "_selected_backend", None)
    module._backend.cache_clear()
    try:
        backend = module._backend(bool(setting))
        module.configure(bool(setting))
        assert module._selected_backend is backend
        module.configure(bool(setting))
        with pytest.raises(RuntimeError):
            module.configure(not bool(setting))
        if setting:
            assert backend.func is module._eager_logits
            assert (
                backend.args[0] is not deep_gemm.bf16_attention.bf16_mqa_logits_kernel
            )
        else:
            assert backend is deep_gemm.bf16_mqa_logits
    finally:
        module._backend.cache_clear()


@pytest.mark.parametrize("rows,columns", [(31, 7), (3, 1025), (65, 17)])
def test_deepgemm_eager_reference_and_replay(kernels, monkeypatch, rows, columns):
    module = importlib.import_module("_dsv4_numeric_test.indexer.deep_gemm")
    monkeypatch.setattr(module, "_selected_backend", None)
    module._backend.cache_clear()
    module.configure(True)
    torch.manual_seed(541)
    q = torch.randn((rows, 64, 128), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((columns, 128), device="cuda", dtype=torch.bfloat16)
    weights = torch.rand((rows, 64), device="cuda")
    ks = torch.arange(rows, device="cuda", dtype=torch.int32) % columns
    ke = torch.full_like(ks, columns)
    ks[0] = columns  # Empty range; masked logits must be -inf.
    call = lambda: module.bf16_mqa_logits(q, kv, weights, ks, ke)
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    try:
        for scale in (0.5, -1.0):
            q.mul_(scale)
            ke[1] = 0
            output.fill_(torch.nan)
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.zeros((rows, columns), device="cuda", dtype=torch.float64)
            for head in range(64):
                expected += (q[:, head].double() @ kv.double().T).relu() * weights[
                    :, head, None
                ].double()
            ids = torch.arange(columns, device="cuda")
            expected.masked_fill_(
                (ids[None] < ks[:, None]) | (ids[None] >= ke[:, None]), -torch.inf
            )
            # BF16 inputs, FP32 dot/reduction; the reference accumulates in FP64.
            torch.testing.assert_close(output.double(), expected, rtol=1e-5, atol=1e-4)
    finally:
        module._backend.cache_clear()
