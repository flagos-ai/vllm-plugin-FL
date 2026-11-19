"""Exact BF16 gate/add semantics and scoped instance overrides."""

from types import SimpleNamespace

import pytest
import torch

from vllm_fl.cp.fused_moe import (
    _fused_moe_sum,
    _ungated_no_copy,
    gate_add,
    sum_gate_add,
    try_prefill_moe,
)


@pytest.mark.gpu
@pytest.mark.parametrize("rows,width", [(0, 2048), (1, 2048), (31, 127), (2048, 2048)])
def test_gate_add_keeps_intermediate_bf16_rounding(rows, width):
    torch.manual_seed(10)
    shared = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    routed = torch.randn_like(shared)
    gate = torch.sigmoid(torch.randn(rows, 1, device="cuda", dtype=shared.dtype))
    before = shared.clone()
    assert torch.equal(gate_add(gate, shared, routed), gate * shared + routed)
    assert torch.equal(shared, before)


def test_gate_add_noncontiguous_fallback():
    shared = torch.randn(3, 14)[:, ::2]
    routed = torch.randn_like(shared)
    gate = torch.randn(3, 1)
    assert torch.equal(gate_add(gate, shared, routed), gate * shared + routed)


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [1, 31, 1024, 4104])
def test_sum_gate_add_matches_native_reduction(rows):
    from vllm import _custom_ops as ops

    torch.manual_seed(11)
    # Wide exponents exercise FP32 summation order and cancellation, unlike
    # small BF16 random values whose sum is frequently exact in FP32.
    experts = (
        torch.randn(rows, 8, 2048, device="cuda")
        * torch.exp2(torch.randint(-20, 20, (rows, 8, 2048), device="cuda"))
    ).bfloat16()
    gate = torch.sigmoid(torch.randn(rows, 1, device="cuda", dtype=torch.bfloat16))
    shared = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16)
    actual = torch.empty_like(shared)
    assert sum_gate_add(experts, gate, shared, actual)
    routed = torch.empty_like(shared)
    ops.moe_sum(experts, routed)
    expected = gate * shared + routed
    assert torch.equal(actual, expected)


def test_sum_gate_add_rejects_unsupported_layout():
    experts = torch.zeros(2, 4, 2048)
    gate = torch.zeros(2, 1)
    shared = torch.zeros(2, 2048)
    actual = torch.full_like(shared, 7)
    assert not sum_gate_add(experts, gate, shared, actual)
    assert torch.equal(actual, torch.full_like(shared, 7))


@pytest.mark.parametrize("has_override", [False, True])
def test_sum_scope_preserves_shared_and_restores_override(monkeypatch, has_override):
    from vllm_fl.cp import fused_moe

    class Experts:
        def moe_sum(self, inputs, output):
            output.copy_(inputs.sum(dim=1))

    experts = Experts()
    if has_override:
        experts.moe_sum = lambda inputs, output: output.fill_(9)
    previous = dict(experts.__dict__)
    shared = torch.full((2, 3), 7.0)
    runner = SimpleNamespace(
        quant_method=SimpleNamespace(
            moe_kernel=SimpleNamespace(impl=SimpleNamespace(fused_experts=experts))
        ),
        _shared_experts=SimpleNamespace(_output=[shared], _output_idx=0),
    )
    mlp = SimpleNamespace(experts=SimpleNamespace(runner=runner))
    monkeypatch.setattr(fused_moe, "sum_gate_add", lambda *args: True)
    with (
        pytest.raises(RuntimeError, match="injected"),
        _fused_moe_sum(mlp, None) as state,
    ):
        experts.moe_sum(None, None)
        assert state["used"]
        assert torch.equal(shared, torch.full_like(shared, 7))
        raise RuntimeError("injected")
    assert experts.__dict__ == previous


@pytest.mark.parametrize("has_override", [False, True])
def test_moe_scope_restores_state_on_exception(has_override):
    class Impl:
        def _finalize(self, *args):
            return "original"

    impl = Impl()
    if has_override:
        impl._finalize = lambda *args: "override"
    expected = dict(impl.__dict__)
    gate = object()
    shared = SimpleNamespace(expert_gate=gate)
    with (
        pytest.raises(RuntimeError, match="injected"),
        _ungated_no_copy(shared, impl) as saved,
    ):
        assert saved is gate and shared.expert_gate is None
        tensor = torch.ones(2, 3)
        assert impl._finalize(None, tensor) is tensor
        raise RuntimeError("injected")
    assert shared.expert_gate is gate
    assert impl.__dict__ == expected


def test_non_cuda_moe_does_not_touch_runner():
    assert try_prefill_moe(None, torch.zeros(3, 4)) is None


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [5, 9, 16, 17])
@pytest.mark.parametrize("owned", [True, False])
def test_overlap_keeps_partials_alive_during_workspace_reuse(monkeypatch, rows, owned):
    from vllm_fl.cp import prstyle

    chunk_rows, width = 4, 16
    local = torch.arange(
        rows * width * 2,
        device="cuda",
        dtype=torch.float32,
    ).reshape(rows, width * 2)[:, ::2]
    workspace = torch.empty(2 * chunk_rows, width, device="cuda")
    pointers, computed, reduced, destinations = [], [], [], []
    # Load the small CUDA kernels before deliberately delaying another stream;
    # CUDA's lazy module loading may otherwise synchronize the first launch.
    warm = (local.contiguous() + 3).repeat(2, 1)
    warm[:rows].copy_(warm[:rows] + warm[rows:])
    torch.cuda._sleep(1)
    torch.cuda.synchronize()

    def moe(_mlp, gathered, *, out):
        partial = workspace[: gathered.shape[0]]
        partial.copy_(gathered + 3)
        # False is a negative control: recreate the old borrowed-workspace bug.
        result = out.copy_(partial) if owned else partial
        pointers.append(result.data_ptr())
        done = torch.cuda.Event(enable_timing=True)
        done.record()
        computed.append(done)
        return result

    class Communicator:
        def all_gather(self, output, value, stream):
            with torch.cuda.stream(stream):
                output.copy_(value.repeat(2, 1))

        def reduce_scatter(self, output, partial, stream):
            with torch.cuda.stream(stream):
                # Let the compute stream reach workspace reuse before this read.
                torch.cuda._sleep(200_000_000)
                size = output.shape[0]
                output.copy_(partial[:size] + partial[size:])
                destinations.append(output.data_ptr())
                done = torch.cuda.Event(enable_timing=True)
                done.record(stream)
                reduced.append(done)

    comm = Communicator()
    monkeypatch.setattr(prstyle, "_eligible", lambda *_args: True)
    monkeypatch.setattr(prstyle, "_flagcx_comm", lambda _group: comm)
    monkeypatch.setattr(prstyle, "try_prefill_moe", moe)
    actual = prstyle._overlap_pr_moe(
        None,
        local,
        SimpleNamespace(uniform_rows=True),
        SimpleNamespace(world_size=2),
        chunk_rows=chunk_rows,
    )
    assert len(pointers) == (rows + chunk_rows - 1) // chunk_rows
    # Prove that chunk 1 actually computes before reduction 0 finishes.
    reduced[0].synchronize()
    computed[1].synchronize()
    assert computed[1].elapsed_time(reduced[0]) > 0
    assert len(set(pointers)) == (2 if owned else 1)
    assert all(pointers[i] == pointers[i % 2] for i in range(len(pointers)))
    assert destinations == [
        actual[start:].data_ptr() for start in range(0, rows, chunk_rows)
    ]
    assert torch.equal(actual, 2 * (local + 3)) == owned


@pytest.mark.gpu
def test_overlap_rejects_unsupported_tail_before_collectives(monkeypatch):
    from vllm_fl.cp import prstyle

    local = torch.ones(5, 16, device="cuda")
    monkeypatch.setattr(prstyle, "_eligible", lambda _, value: value.shape[0] > 2)
    # Any attempted collective would fail on this object.
    monkeypatch.setattr(prstyle, "_flagcx_comm", lambda _: object())
    assert (
        prstyle._overlap_pr_moe(
            None,
            local,
            SimpleNamespace(uniform_rows=True),
            SimpleNamespace(world_size=2),
            chunk_rows=4,
        )
        is None
    )


@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("owned", [False, True])
def test_moe_returns_owned_output_even_when_sum_fusion_falls_back(
    monkeypatch, fuse, owned
):
    from vllm_fl.cp import fused_moe

    inputs = torch.ones(2, 3)
    expert_values = torch.arange(48.0).reshape(2, 8, 3)
    shared_values = torch.full_like(inputs, 7)
    workspace = torch.full_like(inputs, -9)
    out = torch.empty_like(inputs) if owned else None

    class Experts:
        def moe_sum(self, values, output):
            output.copy_(values.sum(dim=1))

    experts = Experts()
    impl = SimpleNamespace(fused_experts=experts)
    gate = lambda value: (torch.zeros(value.shape[0], 1), None)
    shared = SimpleNamespace(expert_gate=gate)
    runner = SimpleNamespace(
        quant_method=SimpleNamespace(moe_kernel=SimpleNamespace(impl=impl)),
        _shared_experts=SimpleNamespace(_output=[shared_values], _output_idx=0),
        _encode_layer_name=lambda: "test",
    )

    def forward(*args):
        assert shared.expert_gate is None
        experts.moe_sum(expert_values, workspace)
        return shared_values, impl._finalize(None, workspace)

    def sum_add(values, gates, shared_output, destination):
        if not fuse:
            return False
        destination.copy_(values.sum(dim=1) + gates * shared_output)
        return True

    runner._forward_entry = forward
    mlp = SimpleNamespace(
        experts=SimpleNamespace(runner=runner, is_internal_router=True),
        shared_expert=shared,
    )
    monkeypatch.setattr(fused_moe, "_eligible", lambda *_: True)
    monkeypatch.setattr(fused_moe, "sum_gate_add", sum_add)
    actual = try_prefill_moe(mlp, inputs, out=out)
    assert torch.equal(actual, expert_values.sum(dim=1) + 0.5 * shared_values)
    if owned:
        assert actual is out
        # The fused kernel writes the destination directly, with no workspace copy.
        if fuse:
            assert torch.equal(workspace, torch.full_like(workspace, -9))
        workspace.fill_(-99)
        assert torch.equal(actual, expert_values.sum(dim=1) + 0.5 * shared_values)
    assert shared.expert_gate is gate
    assert "moe_sum" not in experts.__dict__ and "_finalize" not in impl.__dict__


@pytest.mark.parametrize(
    "out", [torch.empty(2, 4), torch.empty(2, 3).double(), torch.empty(2, 6)[:, ::2]]
)
def test_moe_rejects_invalid_destination_before_execution(monkeypatch, out):
    from vllm_fl.cp import fused_moe

    monkeypatch.setattr(fused_moe, "_eligible", lambda *_: True)
    with pytest.raises(ValueError, match="output must match"):
        try_prefill_moe(None, torch.ones(2, 3), out=out)
