# Copyright (c) 2026 BAAI. All rights reserved.

"""Real CUDA coverage for invalid-route alignment and padded expert execution."""

from contextlib import nullcontext

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(
    get_backend() != "nvidia", reason="NVIDIA MoE alignment"
)


@pytest.fixture
def align():
    from vllm_fl.ops.nvidia_moe_align import moe_align_block_size_nvidia

    return moe_align_block_size_nvidia


def _assert_alignment(ids, block, experts, mapping, pad, result):
    flat = ids.cpu().long().reshape(-1).tolist()
    mapping = mapping.cpu().tolist() if mapping is not None else list(range(experts))
    groups = [[] for _ in range(experts)]
    for row, expert in enumerate(flat):
        if 0 <= expert < experts and 0 <= mapping[expert] < experts:
            groups[mapping[expert]].append(row)
    capacity = len(flat) + experts * (block - 1)
    if pad:
        capacity = ((capacity + block - 1) // block) * block
    if len(flat) < experts:
        capacity = min(len(flat) * block, capacity)
    sorted_ids, expert_ids, post_pad = [x.cpu() for x in result]
    assert all(x.dtype == torch.int32 for x in result)
    assert sorted_ids.shape == (capacity,)
    assert expert_ids.shape == ((capacity + block - 1) // block,)
    assert post_pad.shape == (1,)
    cursor = 0
    for expert, routes in enumerate(groups):
        size = ((len(routes) + block - 1) // block) * block
        actual = sorted_ids[cursor : cursor + size].tolist()
        assert sorted(row for row in actual if row != len(flat)) == routes
        assert actual.count(len(flat)) == size - len(routes)
        assert expert_ids[cursor // block : (cursor + size) // block].tolist() == [
            expert
        ] * (size // block)
        cursor += size
    assert post_pad.item() == cursor
    assert sorted_ids[cursor:].tolist() == [len(flat)] * (capacity - cursor)
    assert expert_ids[cursor // block :].tolist() == [-1] * (
        len(expert_ids) - cursor // block
    )


def _inputs(rows, experts, dtype, mapped):
    generator = torch.Generator().manual_seed(271)
    ids = torch.randint(-2, experts + 2, (rows, 3), generator=generator)
    if rows:
        ids[0] = torch.tensor([-1, experts, experts + 1])
    ids = ids.to(device="cuda", dtype=dtype)
    # Deliberately non-contiguous inputs, including the expert map.
    ids = torch.stack((ids, ids), dim=-1)[..., 0]
    mapping = None
    if mapped:
        mapping = torch.arange(experts, device="cuda", dtype=torch.int32).flip(0) // 2
        mapping[::3] = -1
        mapping = torch.stack((mapping, mapping), dim=-1)[:, 0]
    return ids, mapping


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.uint32])
@pytest.mark.parametrize("pad", [False, True])
@pytest.mark.parametrize(
    "rows,experts,block,mapped",
    [
        (0, 8, 16, False),
        (1, 8, 16, True),
        (17, 8, 16, False),
        (17, 8, 16, True),
        (137, 512, 32, True),
        (2049, 512, 64, False),
        (19, 1536, 1, True),
    ],
)
def test_alignment_matches_reference(align, dtype, pad, rows, experts, block, mapped):
    ids, mapping = _inputs(rows, experts, dtype, mapped)
    result = align(ids, block, experts, mapping, pad)
    _assert_alignment(ids, block, experts, mapping, pad, result)


@pytest.mark.gpu
@pytest.mark.parametrize("mapped", [False, True])
def test_all_invalid_routes(align, mapped):
    ids = torch.full((8, 3), -1, dtype=torch.int32, device="cuda")
    mapping = None
    if mapped:
        ids.fill_(0)
        mapping = torch.full((8,), -7, dtype=torch.int32, device="cuda")
    _assert_alignment(ids, 16, 8, mapping, False, align(ids, 16, 8, mapping))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.uint32])
@pytest.mark.parametrize("use_gems", [False, True])
def test_graph_replays_changed_routes_and_map(align, dtype, use_gems):
    import flag_gems

    from vllm_fl.utils import get_flag_gems_whitelist_blacklist

    _, blacklist = get_flag_gems_whitelist_blacklist()
    context = flag_gems.use_gems(exclude=blacklist) if use_gems else nullcontext()
    ids = torch.zeros((33, 3), dtype=dtype, device="cuda")
    mapping = torch.arange(8, dtype=torch.int32, device="cuda")
    with context:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            align(ids, 16, 8, mapping, True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = align(ids, 16, 8, mapping, True)
    new_ids, new_map = _inputs(33, 8, dtype, True)
    for all_invalid in (False, True, False):
        ids.copy_(new_ids)
        mapping.copy_(new_map)
        if all_invalid:
            mapping.fill_(-1)
        graph.replay()
        _assert_alignment(ids, 16, 8, mapping, True, result)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_padded_router_alignment_expert_gemm(dtype):
    from vllm.triton_utils import tl

    from vllm_fl.dispatch.backends.flaggems.impl.fused_moe import (
        invoke_fused_moe_triton_kernel_flaggems,
        moe_align_block_size_flaggems,
    )
    from vllm_fl.nvidia_compat import has_native_vllm_kernels
    from vllm_fl.ops.nvidia_moe_ops import register_nvidia_moe_ops

    if has_native_vllm_kernels():
        pytest.skip("Exercise the EMPTY provider and FlagGems expert GEMM")
    assert register_nvidia_moe_ops()
    torch.manual_seed(23)
    rows, experts, topk, width = 17, 8, 2, 32
    logits = torch.randn(rows, experts, device="cuda", dtype=dtype)
    padding = torch.arange(rows, device="cuda") % 3 == 0
    weights = torch.empty(rows, topk, dtype=torch.float32, device="cuda")
    ids = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
    source = torch.empty_like(ids)
    torch.ops._moe_C.topk_softmax(weights, ids, source, logits, True, None, padding)
    mapping = torch.tensor([2, -1, 0, 1, -1, 2, 1, 0], dtype=torch.int32, device="cuda")
    aligned = moe_align_block_size_flaggems(ids, 16, experts, mapping, True, True)
    _assert_alignment(ids, 16, experts, mapping, True, aligned)
    a = torch.randn(rows, width, device="cuda", dtype=dtype)
    b = torch.randn(3, width, width, device="cuda", dtype=dtype)
    out = torch.zeros(rows, topk, width, device="cuda", dtype=dtype)
    invoke_fused_moe_triton_kernel_flaggems(
        a,
        b,
        out,
        None,
        None,
        weights,
        *aligned,
        True,
        topk,
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 1},
        tl.float16 if dtype == torch.float16 else tl.bfloat16,
        False,
        False,
        False,
        False,
        False,
    )
    expected = torch.zeros_like(out, device="cpu")
    cpu_a, cpu_b, cpu_weights = a.cpu().float(), b.cpu().float(), weights.cpu()
    cpu_map = mapping.cpu().tolist()
    for row, routes in enumerate(ids.cpu().tolist()):
        for k, expert in enumerate(routes):
            if expert >= 0 and cpu_map[expert] >= 0:
                expected[row, k] = (
                    cpu_a[row] @ cpu_b[cpu_map[expert]].T
                ) * cpu_weights[row, k]
    torch.testing.assert_close(out.cpu(), expected, rtol=0.02, atol=0.04)
    assert torch.count_nonzero(out[padding]).item() == 0


@pytest.mark.gpu
@pytest.mark.parametrize(
    "ignore,is_cuda,native",
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_other_alignment_paths_still_call_gems(monkeypatch, ignore, is_cuda, native):
    import flag_gems

    from vllm.platforms import current_platform

    from vllm_fl import nvidia_compat
    from vllm_fl.dispatch.backends.flaggems.impl.fused_moe import (
        moe_align_block_size_flaggems,
    )

    called = []

    def existing_gems(ids, experts, block, sorted_ids, expert_ids, post_pad):
        called.append((ids, experts, block))

    monkeypatch.setattr(current_platform, "is_cuda", lambda: is_cuda)
    monkeypatch.setattr(nvidia_compat, "has_native_vllm_kernels", lambda: native)
    monkeypatch.setattr(flag_gems, "moe_align_block_size_triton", existing_gems)
    ids = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    moe_align_block_size_flaggems(ids, 16, 8, ignore_invalid_experts=ignore)
    assert len(called) == 1 and called[0][0] is ids
