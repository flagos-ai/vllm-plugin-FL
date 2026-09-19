# SPDX-License-Identifier: Apache-2.0
"""FP8 page addressing, numerical parity and changing-input graph replay."""

import importlib

import pytest
import torch

from vllm_fl.kernels.glm5_next.indexer_backend import (
    _graph_safe_flaggems_paged_mqa_logits,
)

pytestmark = pytest.mark.gpu


def make_case(page_size, padding=0, pool_pages=128, batch=3):
    torch.manual_seed(11)
    device = "cuda"
    width, heads, max_len = 128, 32, 2051
    stride = page_size * (width + 4) + padding
    raw = torch.zeros(pool_pages, stride, dtype=torch.uint8, device=device)
    cache = torch.as_strided(
        raw, (pool_pages, page_size, 1, width + 4), (stride, width + 4, width + 4, 1)
    )
    keys = torch.randn(pool_pages, page_size, width, device=device).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(pool_pages, page_size, device=device) * 0.05 + 0.01
    raw[:, : page_size * width].copy_(keys.view(torch.uint8).reshape(pool_pages, -1))
    raw[:, page_size * width : page_size * (width + 4)].copy_(scales.view(torch.uint8))
    query = torch.randn(batch, 1, heads, width, device=device).to(torch.float8_e4m3fn)
    weights = torch.randn(batch, heads, device=device)
    table = torch.stack(
        [torch.randperm(pool_pages, device=device)[:65] for _ in range(batch)]
    ).int()
    lens = torch.full((batch,), 65, dtype=torch.int32, device=device)
    return query, cache, weights, lens, table, keys, scales, max_len


def candidate(case):
    q, cache, weights, lens, table, _, _, max_len = case
    loaded = importlib.import_module("flag_gems.fused.fp8_fp4_paged_mqa_logits")
    return _graph_safe_flaggems_paged_mqa_logits(
        loaded,
        (q, None),
        cache,
        weights,
        lens,
        table,
        None,
        max_len,
        True,
    )


def reference(case):
    q, _, weights, lens, table, keys, scales, max_len = case
    result = torch.full((len(lens), max_len), -torch.inf, device=q.device)
    for row, length in enumerate(lens.tolist()):
        idx = table[row].long()
        k = keys.float()[idx].reshape(-1, 128)[:length]
        s = scales[idx].reshape(-1)[:length]
        dots = q[row, 0].float() @ k.T
        result[row, :length] = (torch.relu(dots * s) * weights[row, :, None]).sum(0)
    return result


@pytest.mark.parametrize("page_size,padding", [(32, 0), (32, 256), (64, 0), (64, 256)])
def test_page_stride_eager_and_changing_graph_inputs(page_size, padding):
    case = make_case(page_size, padding)
    for length in (0, 1, page_size - 1, page_size, page_size + 1, 2049):
        case[3].copy_(torch.tensor([length, length // 2, 0], device="cuda"))
        torch.testing.assert_close(
            candidate(case), reference(case), atol=2e-5, rtol=2e-5
        )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = candidate(case)
    # Rewrite metadata in place: empty/padded row becomes live, and logical
    # pages resolve to different physical pages after the capture.
    case[3].copy_(torch.tensor([33, 2051, 64], device="cuda"))
    case[4].copy_(case[4].flip(1))
    expected = reference(case)
    for _ in range(10):
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_private_vision_attention_eager_and_replay(dtype):
    from vllm_fl.kernels.glm5_next.vision_attention import Glm5VisionAttention

    torch.manual_seed(17)
    query, key, value = [
        torch.randn(1, 33, 4, 64, device="cuda", dtype=dtype) for _ in range(3)
    ]
    cu = torch.tensor([0, 16, 33], dtype=torch.int32, device="cuda")
    layer = Glm5VisionAttention(4, 64, 64**-0.5)

    def ref(boundary):
        chunks = []
        for start, end in ((0, boundary), (boundary, 33)):
            chunks.append(
                torch.nn.functional.scaled_dot_product_attention(
                    query[:, start:end].transpose(1, 2).float(),
                    key[:, start:end].transpose(1, 2).float(),
                    value[:, start:end].transpose(1, 2).float(),
                ).transpose(1, 2)
            )
        return torch.cat(chunks, 1).to(dtype)

    tol = 3e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(
        layer(query, key, value, cu), ref(16), atol=tol, rtol=tol
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = layer(query, key, value, cu)
    cu.copy_(torch.tensor([0, 1, 33], dtype=torch.int32, device="cuda"))
    query.copy_(query * 0.5)
    graph.replay()
    torch.testing.assert_close(out, ref(1), atol=tol, rtol=tol)
