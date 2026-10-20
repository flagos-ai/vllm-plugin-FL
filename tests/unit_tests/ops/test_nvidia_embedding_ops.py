# Copyright (c) 2026 BAAI. All rights reserved.

"""Check the real EMPTY TP embedding CUDA op against independent CPU gathers."""

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(get_backend() != "nvidia", reason="NVIDIA TP embedding")


@pytest.fixture
def embedding_op():
    from vllm_fl.nvidia_compat import has_native_vllm_kernels
    from vllm_fl.ops.nvidia_embedding_ops import register_nvidia_embedding_ops

    if has_native_vllm_kernels():
        pytest.skip("Native wheel supplies its own TP embedding operation")
    assert register_nvidia_embedding_ops()
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "_C::vocab_parallel_embedding", "CUDA"
    )
    return torch.ops._C.vocab_parallel_embedding


def _expected(ids, weight, start, end, padding, added_start, added_end):
    ids, weight = ids.cpu().tolist(), weight.cpu()
    output = torch.zeros(len(ids), weight.shape[1], dtype=weight.dtype)
    for row, token in enumerate(ids):
        if start <= token < end:
            output[row] = weight[token - start]
        elif added_start <= token < added_end:
            output[row] = weight[token - added_start + end - start + padding]
    return output


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_tp_embedding_shards_padding_and_added_tokens(embedding_op, dtype, id_dtype):
    ids = torch.tensor(
        [-1, 0, 16, 17, 33, 34, 35, 36, 37, 38, 100], device="cuda", dtype=id_dtype
    )
    partials, references = [], []
    for rank in range(2):
        indices = (rank * 17, (rank + 1) * 17, 3, 34 + rank * 2, 36 + rank * 2)
        weight = (torch.arange(22 * 72, device="cuda").reshape(22, 72) + rank).to(dtype)
        output = torch.full((len(ids), 72), float("nan"), device="cuda", dtype=dtype)
        embedding_op(output, ids, weight, *indices)
        expected = _expected(ids, weight, *indices)
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        partials.append(output)
        references.append(expected)
    torch.testing.assert_close(
        (partials[0] + partials[1]).cpu(), references[0] + references[1], rtol=0, atol=0
    )


@pytest.mark.gpu
def test_tp_embedding_graph_replay_observes_new_ids(embedding_op):
    ids = torch.tensor([0, 16, 17, 34, 35, 38], device="cuda", dtype=torch.int32)
    weight = torch.randn(22, 72, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(6, 72, device="cuda", dtype=weight.dtype)
    indices = (0, 17, 3, 34, 36)

    def run():
        embedding_op(output, ids, weight, *indices)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    ids.copy_(torch.tensor([35, 34, 17, 16, -1, 0], device="cuda", dtype=ids.dtype))
    graph.replay()
    torch.testing.assert_close(output.cpu(), _expected(ids, weight, *indices), rtol=0, atol=0)
