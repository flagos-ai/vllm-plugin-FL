"""Numerical and boundary checks for CP layout fusion."""

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.cp.layout import ZigZagPlan


@pytest.mark.parametrize(
    "starts,lengths",
    [
        ([0, 32], [32]),
        ([0, 31], [47]),
        ([0, 1], [5]),
        ([0, 0], [0]),
        ([0, 9, 24, 56], [25, 17, 96]),
    ],
)
def test_plans_partition_and_restore_causal_rows(starts, lengths):
    plans = ZigZagPlan.build_all(starts, lengths, 4)
    full = torch.arange(starts[-1])
    packed = torch.full((4 * plans[0].padded_rows,), -1)
    for rank, plan in enumerate(plans):
        local = plan.select(full)
        offset = rank * plan.padded_rows
        packed[offset : offset + len(local)] = local
        for i, req in enumerate(plan.request_ids):
            rows = local[plan.query_start_loc[i] : plan.query_start_loc[i + 1]]
            assert torch.all(rows[1:] == rows[:-1] + 1)
            prefix = lengths[req] - (starts[req + 1] - starts[req])
            assert plan.seq_lens[i] == prefix + rows[-1].item() - starts[req] + 1
    assert sorted(row for plan in plans for row in plan.rows) == full.tolist()
    assert torch.equal(
        packed[torch.tensor(plans[0].restore_rows, dtype=torch.long)], full
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("starts", [[0, 32], [0, 31], [0, 1], [0, 0], [0, 9, 25]])
@pytest.mark.parametrize("heads,local_heads", [(4, 1), (8, 2), (2, 1)])
def test_fused_kv_exactly_matches_ordered_reference(device, starts, heads, local_heads):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    plans = ZigZagPlan.build_all(starts, starts[1:], 4)
    n, dim = starts[-1], 7
    # A QKV projection produces views with row/head gaps. Also exercise a
    # strided last dimension and a non-power-of-two head width.
    source = torch.arange(
        n * heads * 4 * dim, device=device, dtype=torch.float32
    ).reshape(n, heads, 4 * dim)
    source = (source % 101).to(torch.bfloat16)
    k, v = source[..., : 2 * dim : 2], source[..., 2 * dim :: 2]
    packed = torch.zeros(
        (4 * plans[0].padded_rows, 2, heads, dim), dtype=k.dtype, device=device
    )
    locals_ = []
    for rank, plan in enumerate(plans):
        # Select the backing storage first, preserving the strided K/V views.
        local_source = plan.select(source)
        lk = local_source[..., : 2 * dim : 2]
        lv = local_source[..., 2 * dim :: 2]
        locals_.append((lk, lv))
        begin = rank * plan.padded_rows
        packed[begin : begin + len(plan.rows), 0] = lk
        packed[begin : begin + len(plan.rows), 1] = lv
    for rank, plan in enumerate(plans):
        calls = []

        def gather(value, dim=0, *, calls=calls, rank=rank, plan=plan):
            calls.append(value)
            begin = rank * plan.padded_rows
            assert torch.equal(value, packed[begin : begin + plan.padded_rows])
            return packed

        first = rank * local_heads if heads >= 4 else rank // (4 // heads)
        nk, nv = plan.gather_native_kv(*locals_[rank], first, local_heads, gather)
        assert len(calls) == 1
        assert nk.is_contiguous() and nv.is_contiguous()
        assert torch.equal(nk, k[:, first : first + local_heads])
        assert torch.equal(nv, v[:, first : first + local_heads])


@pytest.mark.parametrize("starts", [[0, 32], [0, 31], [0, 8, 24], [0, 31, 32], [0, 1]])
@pytest.mark.parametrize("already_reduced", [False, True])
def test_moe_preserves_local_tokens_and_single_reduction(starts, already_reduced):
    from vllm_fl.cp.prstyle import _pr_moe

    plans = ZigZagPlan.build_all(starts, starts[1:], 4)
    full = torch.randn(starts[-1], 5)
    expected = (full.square() + full.sigmoid()) * sum(range(1, 5))
    for rank, plan in enumerate(plans):
        collectives = []
        runner = SimpleNamespace(
            _fused_output_is_reduced=already_reduced,
            _maybe_reduce_final_output=object(),
        )
        original = runner._maybe_reduce_final_output

        def gather(value, dim=0, *, rank=rank):
            rows = []
            for owner in plans:
                padded = full.new_zeros((owner.padded_rows, full.shape[1]))
                padded[: len(owner.rows)] = owner.select(full)
                rows.append(padded)
            assert torch.equal(value, rows[rank])
            return torch.cat(rows)

        def scatter(value, dim=0, *, collectives=collectives, rank=rank):
            collectives.append("reduce_scatter")
            return (value * (10 / (rank + 1))).chunk(4)[rank]

        class Experts:
            is_internal_router = False

            def __call__(
                self, hidden_states, router_logits, *, rank=rank, runner=runner
            ):
                assert torch.equal(router_logits, hidden_states + 1)
                partial = (hidden_states.square() + hidden_states.sigmoid()) * (
                    rank + 1
                )
                return runner._maybe_reduce_final_output(partial, full.shape[1])

        experts = Experts()
        experts.runner = runner

        class MLP:
            def __call__(self, x):
                return (x.square() + x.sigmoid()) * 10

        mlp = MLP()
        mlp.experts = experts
        mlp.gate = lambda x: (x + 1, None)
        group = SimpleNamespace(
            world_size=4, rank_in_group=rank, all_gather=gather, reduce_scatter=scatter
        )
        actual = _pr_moe(mlp, plan.select(full), plan, plans, group)
        torch.testing.assert_close(actual, plan.select(expected))
        assert collectives == ([] if already_reduced else ["reduce_scatter"])
        assert runner._maybe_reduce_final_output is original


def test_prefill_cache_is_scoped_to_forward_and_prefix(monkeypatch):
    from vllm_fl.cp import prstyle

    class HostCopy:
        def __init__(self, values):
            self.values = values
            self.copies = 0

        def cpu(self):
            self.copies += 1
            return torch.tensor(self.values)

    context = SimpleNamespace(additional_kwargs={}, no_compile_layers={})
    group = SimpleNamespace(world_size=4)
    metadata = SimpleNamespace(
        max_query_len=32, query_start_loc=HostCopy([0, 32]), seq_lens=HostCopy([32])
    )
    monkeypatch.setattr(prstyle, "get_forward_context", lambda: context)
    monkeypatch.setattr(prstyle, "get_tp_group", lambda: group)
    prstyle._build_plans.cache_clear()
    for _ in range(40):
        assert prstyle._is_sequence_prefill(metadata)
        plans = prstyle._plans(context, metadata, group)
    assert metadata.query_start_loc.copies == metadata.seq_lens.copies == 1
    context.additional_kwargs.clear()
    metadata.seq_lens.values = [64]
    prefix_plans = prstyle._plans(context, metadata, group)
    assert prefix_plans[0].seq_lens == tuple(n + 32 for n in plans[0].seq_lens)
    context.additional_kwargs.clear()
    metadata.query_start_loc.values = [0, 31, 32]
    assert not prstyle._is_sequence_prefill(metadata)


@pytest.mark.parametrize(
    "role,expected",
    [
        (None, False),
        ("kv_producer", True),
        ("kv_consumer", False),
        ("kv_both", False),
    ],
)
@pytest.mark.parametrize("lengths", [(65535, 1), (1, 2, 65533), (7, 8)])
def test_pd_producer_admits_short_chunks_without_changing_other_roles(
    monkeypatch, role, expected, lengths
):
    from itertools import accumulate

    from vllm_fl.cp import prstyle

    context = SimpleNamespace(
        additional_kwargs={},
        no_compile_layers={
            "test": SimpleNamespace(_fl_prstyle_kv_producer=role == "kv_producer")
        },
    )
    metadata = SimpleNamespace(
        max_query_len=max(lengths),
        query_start_loc=torch.tensor(tuple(accumulate(lengths, initial=0))),
    )
    monkeypatch.setattr(prstyle, "get_forward_context", lambda: context)
    monkeypatch.setattr(prstyle, "get_tp_group", lambda: SimpleNamespace(world_size=4))
    monkeypatch.setattr(
        prstyle,
        "get_current_vllm_config",
        lambda: pytest.fail("Forward must not read the initialization context"),
    )
    assert prstyle._is_sequence_prefill(metadata) == expected
    # A subsequent all-single-token batch must retain the native TP path.
    context.additional_kwargs.clear()
    metadata.max_query_len = 1
    assert not prstyle._is_sequence_prefill(metadata)
