# Copyright 2025 BAAI. All rights reserved.
"""Real-BI-V150 coverage for the FlagGems MoE block-size alignment kernel.

``vllm_fl/dispatch/backends/flaggems/impl/fused_moe.py::moe_align_block_size_flaggems``
runs ``flaggems_vllm.moe_align_block_size_triton`` — a Triton kernel that
only executes on a real device. The CPU baseline cannot compile it, leaving
the function's device orchestration unmeasured.

These tests run the real kernel on the BI-V150 against an independent,
pure-PyTorch bucketing oracle (the same route-invariant reasoning as the
GCU functional suite, implemented inline so nothing is imported from the
code under test):

* random routes with -1 padding sentinels (upstream's top-k kernel writes
  -1 for padded rows) at balanced and skewed distributions;
* every routed slot must appear exactly once in ``sorted_ids[:keep]``, the
  remaining entries must be the ``num_routes`` padding sentinel;
* each expert's region is padded up to a multiple of ``block_size`` —
  ``num_tokens_post_pad`` equals the sum of padded per-expert blocks;
* ``expert_ids`` names the owning expert for every active block;
* the ``expert_map`` remap branch rewrites block owner ids through the map;
* the FlagGems adapter's allocation sizing (small-batch tightening,
  ``pad_sorted_ids`` rounding) is pinned through the returned shapes.

Integer tensors use exact equality (rtol=0, atol=0) — there is no floating
point in this op. Vendor-gated to iluvatar; carries ``gpu``.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

pytestmark = [
    pytest.mark.supplementary,
    pytest.mark.platforms("iluvatar"),
    pytest.mark.gpu,
]

# tests/platforms/iluvatar.yaml: integers are exact; the yaml defaults
# (rtol/atol 1e-3) apply to float comparisons, none of which occur here.
RTOL = 0
ATOL = 0


def _device():
    from tests.utils.device_utils import get_device

    return get_device()


def _routes(num_tokens, num_experts, topk, *, skewed=False, generator=None):
    """Random routing ids with -1 padding sentinels, distinct within a token."""
    routes = torch.randint(
        0, num_experts, (num_tokens, topk), generator=generator, dtype=torch.int64
    )
    # de-duplicate within each token (top-k always returns distinct experts)
    for row in range(num_tokens):
        seen, out = set(), []
        for col in range(topk):
            v = int(routes[row, col])
            while v in seen:
                v = (v + 1) % num_experts
            seen.add(v)
            out.append(v)
        routes[row] = torch.tensor(out, dtype=torch.int64)
    if skewed:
        routes = routes % max(1, num_experts // 4)
    # sprinkle -1 padding sentinels on the last column of some rows
    mask = torch.rand(num_tokens, generator=generator) < 0.2
    routes[mask, -1] = -1
    return routes


def _expected_alignment(routes_cpu, block_size, num_experts):
    """Pure-PyTorch oracle: per-expert padded block structure.

    Returns (counts, padded_blocks, total_post_pad) computed the same way
    the reference implementation buckets: valid ids only, each expert's slot
    count rounded up to ``block_size``.
    """
    flat = routes_cpu.reshape(-1).to(torch.int64)
    valid = (flat >= 0) & (flat < num_experts)
    counts = Counter(flat[valid].tolist())
    blocks = {
        expert: (count + block_size - 1) // block_size
        for expert, count in counts.items()
    }
    total = sum(blocks.values()) * block_size
    return counts, blocks, total


def _assert_alignment(outputs, routes_cpu, block_size, num_experts):
    sorted_ids, expert_ids, num_post_pad = outputs
    for output in outputs:
        assert output.device.type == "cuda"
        assert output.dtype == torch.int32
        assert output.ndim == 1

    keep = int(num_post_pad.cpu().item())
    _, expected_blocks, expected_total = _expected_alignment(
        routes_cpu, block_size, num_experts
    )
    assert keep == expected_total, (
        f"num_tokens_post_pad {keep} != oracle {expected_total}"
    )
    assert 0 < keep <= sorted_ids.numel()

    num_routes = routes_cpu.numel()
    active_ids = sorted_ids[:keep].cpu().to(torch.int64)
    valid = active_ids < num_routes
    # every routed slot appears exactly once
    torch.testing.assert_close(
        torch.sort(active_ids[valid]).values,
        torch.arange(int(valid.sum()), dtype=torch.int64),
        rtol=RTOL,
        atol=ATOL,
    )
    assert int(valid.sum()) == int((routes_cpu.reshape(-1) >= 0).sum()) - int(
        (routes_cpu.reshape(-1) >= num_experts).sum()
    ), "valid routed count must match the oracle's valid id count"
    # the remaining region is the padding sentinel
    assert int((active_ids == num_routes).sum()) == keep - int(valid.sum())

    # block ownership
    num_blocks = keep // block_size
    assert num_blocks <= expert_ids.numel()
    active_experts = expert_ids[:num_blocks].cpu().to(torch.int64)
    assert Counter(active_experts.tolist()) == expected_blocks
    for block_index, expert in enumerate(active_experts.tolist()):
        block = active_ids[block_index * block_size : (block_index + 1) * block_size]
        block_valid = block[block < num_routes]
        assert block_valid.numel() > 0, "an active block cannot be all padding"
        flat = routes_cpu.reshape(-1).to(torch.int64)
        assert bool((flat[block_valid] == expert).all())


def _invoke(routes_cpu, block_size, num_experts, **kwargs):
    from vllm_fl.dispatch.backends.flaggems.impl.fused_moe import (
        moe_align_block_size_flaggems,
    )

    device = _device()
    topk_ids = routes_cpu.to(torch.int32).to(device)
    before = topk_ids.clone()
    outputs = moe_align_block_size_flaggems(
        topk_ids, block_size, num_experts, **kwargs
    )
    torch.cuda.synchronize(device)
    # the adapter must not mutate its input
    torch.testing.assert_close(
        topk_ids.cpu(), before.cpu(), rtol=RTOL, atol=ATOL
    )
    return outputs


class TestMoeAlignKernelVsOracle:
    """Real flaggems_vllm kernel vs the pure-torch bucketing oracle."""

    @pytest.mark.parametrize(
        "num_tokens,block_size,num_experts,topk",
        [
            (1, 4, 8, 2),  # single decode route
            (17, 8, 16, 4),  # small prefill, tokens < experts (tightening)
            (128, 16, 32, 8),  # prefill
            (513, 32, 64, 8),  # larger prefill with padding sentinels
        ],
        ids=["decode", "small", "prefill", "large"],
    )
    def test_random_routes_match_oracle(
        self, num_tokens, block_size, num_experts, topk
    ):
        generator = torch.Generator().manual_seed(20261019 + num_tokens)
        routes = _routes(
            num_tokens, num_experts, topk, generator=generator
        )
        outputs = _invoke(routes, block_size, num_experts)
        _assert_alignment(outputs, routes, block_size, num_experts)

    def test_skewed_routing_matches_oracle(self):
        """All tokens concentrate on a quarter of the experts."""
        generator = torch.Generator().manual_seed(7)
        routes = _routes(64, 32, 64, 4, skewed=True, generator=generator)
        outputs = _invoke(routes, 16, 64)
        _assert_alignment(outputs, routes, 16, 64)

    def test_all_padding_sentinels(self):
        """A fully padded batch degenerates to zero post-pad tokens."""
        routes = torch.full((4, 2), -1, dtype=torch.int64)
        outputs = _invoke(routes, 4, 4)
        _, _, num_post_pad = outputs
        assert int(num_post_pad.cpu().item()) == 0

    def test_int64_topk_ids_accepted(self):
        """The kernel contract accepts int64 routing ids too."""
        generator = torch.Generator().manual_seed(11)
        routes = _routes(33, 16, 8, 2, generator=generator)
        device = _device()
        from vllm_fl.dispatch.backends.flaggems.impl.fused_moe import (
            moe_align_block_size_flaggems,
        )

        topk_ids = routes.to(torch.int64).to(device)
        outputs = moe_align_block_size_flaggems(topk_ids, 8, 16)
        torch.cuda.synchronize(device)
        _assert_alignment(outputs, routes, 8, 16)


class TestMoeAlignAllocationContract:
    """The adapter's buffer sizing rules, observed through returned shapes."""

    def test_small_batch_tightened_to_numel_times_block(self):
        # numel (2) < num_experts (8): max padded = min(2 * bs, numel + E*(bs-1))
        routes = torch.tensor([[0, 1]], dtype=torch.int32)
        outputs = _invoke(routes, 4, 8)
        sorted_ids, expert_ids, num_post_pad = outputs
        assert sorted_ids.numel() <= 2 * 4
        keep = int(num_post_pad.cpu().item())
        assert keep == 8  # both experts pad one slot each: 4 + 4

    def test_pad_sorted_ids_rounds_allocation_to_block_multiple(self):
        routes = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        outputs = _invoke(routes, 4, 4, pad_sorted_ids=True)
        sorted_ids, _, _ = outputs
        assert sorted_ids.numel() % 4 == 0

    def test_expert_map_remaps_block_owners(self):
        """``expert_map`` is applied to the returned expert_ids."""
        routes = torch.tensor([[0, 1], [2, 1]], dtype=torch.int32)
        # reverse the expert order: global e -> local (num_experts-1-e)
        num_experts = 4
        expert_map = torch.arange(num_experts - 1, -1, -1, dtype=torch.int32)
        device = _device()
        from vllm_fl.dispatch.backends.flaggems.impl.fused_moe import (
            moe_align_block_size_flaggems,
        )

        topk_ids = routes.to(device)
        mapped, unmapped = (
            moe_align_block_size_flaggems(
                topk_ids, 4, num_experts, expert_map=expert_map.to(device)
            ),
            moe_align_block_size_flaggems(topk_ids, 4, num_experts),
        )
        torch.cuda.synchronize(device)
        keep = int(mapped[2].cpu().item())
        assert keep == int(unmapped[2].cpu().item())
        num_blocks = keep // 4
        mapped_experts = mapped[1][:num_blocks].cpu()
        unmapped_experts = unmapped[1][:num_blocks].cpu()
        torch.testing.assert_close(
            mapped_experts,
            expert_map[unmapped_experts],
            rtol=RTOL,
            atol=ATOL,
        )

    def test_outputs_are_device_int32(self):
        routes = torch.tensor([[3, 1]], dtype=torch.int32)
        sorted_ids, expert_ids, num_post_pad = _invoke(routes, 2, 4)
        assert sorted_ids.dtype == torch.int32
        assert expert_ids.dtype == torch.int32
        assert num_post_pad.dtype == torch.int32
        assert num_post_pad.numel() == 1
        assert sorted_ids.device.type == "cuda"
        assert expert_ids.device.type == "cuda"
        assert num_post_pad.device.type == "cuda"


class TestMoeAlignDeterminism:
    """Re-running the kernel on identical routes is byte-identical."""

    def test_two_runs_identical(self):
        generator = torch.Generator().manual_seed(3)
        routes = _routes(37, 16, 16, 4, generator=generator)
        first = _invoke(routes, 8, 16)
        second = _invoke(routes, 8, 16)
        for a, b in zip(first, second, strict=True):
            torch.testing.assert_close(a.cpu(), b.cpu(), rtol=RTOL, atol=ATOL)
