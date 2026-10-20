# Copyright (c) 2026 BAAI. All rights reserved.

"""Exercise actual registered sparse-indexer kernels, including graph replay."""

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(get_backend() != "nvidia", reason="NVIDIA top-k bridge")


@pytest.fixture
def empty_topk_ops():
    from vllm_fl.nvidia_compat import has_native_vllm_kernels
    from vllm_fl.ops.nvidia_topk_ops import register_nvidia_topk_ops

    if has_native_vllm_kernels():
        pytest.skip("Native wheel supplies its own top-k operations")
    assert register_nvidia_topk_ops()
    return torch.ops._C


def _assert_selection(logits, lengths, output, k, bound):
    logits, lengths, output = logits.cpu(), lengths.cpu().reshape(-1), output.cpu()
    for row, raw_length in enumerate(lengths.tolist()):
        length = max(0, min(raw_length, bound, logits.shape[1]))
        count = min(length, k)
        selected = output[row, :count].long()
        assert torch.all(output[row, count:] == -1)
        if length <= k:
            torch.testing.assert_close(selected, torch.arange(count))
        else:
            assert selected.unique().numel() == k
            assert torch.all((selected >= 0) & (selected < length))
            # Compare scores, not indices: equal boundary scores have no
            # stable selection/ordering guarantee in native CUDA radix top-k.
            actual = logits[row, selected].sort().values
            expected = logits[row, :length].topk(k).values.sort().values
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("op_name", ["cooperative_topk", "persistent_topk"])
@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("width", [4096, 16387])
def test_indexer_topk_lengths_strides_ties(empty_topk_ops, op_name, k, width):
    generator = torch.Generator(device="cuda").manual_seed(132)
    storage = torch.randn(8, width + 16, generator=generator, device="cuda")
    logits = storage[:, :width]
    lengths = torch.tensor(
        [-3, 0, 17, k, k + 19, width - 7, width, width + 100],
        dtype=torch.int32,
        device="cuda",
    ).reshape(4, 2)
    # Two distinct tie cases, plus FP32 values that collapse into one FP16
    # histogram bin. The latter must be resolved using the original FP32.
    logits[4].round_()
    logits[5] = 1 + torch.arange(width, device="cuda") * 1e-7
    logits[6].fill_(0)
    untouched_padding = storage[:, width:].clone()
    output = torch.full((8, k), -77, dtype=torch.int32, device="cuda")
    workspace = torch.empty(1024, dtype=torch.uint8, device="cuda")
    getattr(empty_topk_ops, op_name)(logits, lengths, output, workspace, k, width)
    _assert_selection(logits, lengths, output, k, width)
    torch.testing.assert_close(storage[:, width:], untouched_padding)


@pytest.mark.gpu
@pytest.mark.parametrize("op_name", ["cooperative_topk", "persistent_topk"])
def test_indexer_topk_cuda_graph_updates_lengths(empty_topk_ops, op_name):
    width, k = 4096, 512
    logits = torch.arange(width, device="cuda", dtype=torch.float32).repeat(2, 1)
    lengths = torch.tensor([width, 17], dtype=torch.int32, device="cuda")
    output = torch.empty(2, k, dtype=torch.int32, device="cuda")
    workspace = torch.empty(1024, dtype=torch.uint8, device="cuda")
    op = getattr(empty_topk_ops, op_name)

    def run():
        op(logits, lengths, output, workspace, k, width - 4)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    lengths.copy_(torch.tensor([0, width], dtype=torch.int32, device="cuda"))
    logits.neg_()
    graph.replay()
    _assert_selection(logits, lengths, output, k, width - 4)


def test_topk_registration_is_idempotent(empty_topk_ops):
    from vllm_fl.ops import nvidia_topk_ops

    previous = nvidia_topk_ops._lib
    assert nvidia_topk_ops.register_nvidia_topk_ops()
    assert nvidia_topk_ops._lib is previous
    for name in ("cooperative_topk", "persistent_topk"):
        assert torch._C._dispatch_has_kernel_for_dispatch_key(f"_C::{name}", "CUDA")


def test_topk_preserves_native_wheel(monkeypatch):
    from vllm_fl.ops import nvidia_topk_ops

    previous = nvidia_topk_ops._lib
    monkeypatch.setattr(nvidia_topk_ops, "has_native_vllm_kernels", lambda: True)
    assert not nvidia_topk_ops.register_nvidia_topk_ops()
    assert nvidia_topk_ops._lib is previous
