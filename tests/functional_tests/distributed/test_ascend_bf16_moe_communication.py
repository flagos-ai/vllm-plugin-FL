# SPDX-License-Identifier: Apache-2.0
"""NPU-only component gate, launched with torchrun (no model required).

Example: torchrun --nproc-per-node=8 <this-file> --mode alltoall --hidden 6144
Or: torchrun --nproc-per-node=8 -m pytest <this-file> -q
Cross-host launch uses normal torchrun rendezvous arguments. Plain pytest
collects the cases and explicitly skips them without a multi-rank launcher.
"""

import argparse
import os
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist


@contextmanager
def _communication_group():
    import torch_npu

    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    group = None
    try:
        group = dist.new_group(backend="hccl")
        yield group
    finally:
        if group is not None:
            dist.destroy_process_group(group)
        dist.destroy_process_group()


def _run_case(args, group):
    import torch_npu

    from vllm_fl.dispatch import CachedOp
    from vllm_fl.distributed.device_communicators.npu_moe import (
        NpuAllToAllDispatcher,
        NpuMC2Dispatcher,
    )

    rank, world = dist.get_rank(), dist.get_world_size()
    assert args.experts % world == 0 and world > 1
    device = torch.device("npu", torch_npu.npu.current_device())
    dtype = torch.bfloat16
    # Rank zero exercises idle participation; remaining ranks differ in activity.
    active = 0 if rank == 0 else max(1, args.rows - rank % args.rows)
    rows = args.rows if args.mode == "mc2" else active
    generator = torch.Generator(device="cpu").manual_seed(73 + rank)
    x = (torch.randn(rows, args.hidden, generator=generator) * 0.05).to(device, dtype)
    ids = (
        (torch.arange(rows * args.topk).reshape(rows, args.topk) + rank * args.topk)
        % args.experts
    ).to(device, torch.int32)
    probs = torch.full((rows, args.topk), 1 / args.topk, device=device)
    mask = torch.arange(rows, device=device) < active
    probs = probs * mask[:, None]
    generator.manual_seed(97)
    w1 = (
        torch.randn(
            args.experts, 2 * args.intermediate, args.hidden, generator=generator
        )
        * 0.02
    ).to(device, dtype)
    w2 = (
        torch.randn(args.experts, args.hidden, args.intermediate, generator=generator)
        * 0.02
    ).to(device, dtype)
    local = args.experts // world
    if args.mode == "mc2":
        mc2_dispatcher = NpuMC2Dispatcher(group, args.experts)
        routed = mc2_dispatcher.dispatch(x, ids, probs, mask)
        dispatcher = mc2_dispatcher
    else:
        alltoall_dispatcher = NpuAllToAllDispatcher(group, args.experts)
        routed = alltoall_dispatcher.dispatch(x, ids, probs)
        dispatcher = alltoall_dispatcher
    assert routed.hidden_states.dtype == dtype and routed.pertoken_scale is None
    local_out = CachedOp("bf16_grouped_experts")(
        routed.hidden_states,
        w1[rank * local : (rank + 1) * local],
        w2[rank * local : (rank + 1) * local],
        group_list=routed.group_list,
        group_list_type=routed.group_list_type,
        activation="swigluoai_uninterleave",
        alpha=1.702,
        beta=1.0,
        limit=7.0,
    )
    actual = dispatcher.combine(local_out, routed.handle)
    # Independent explicit reference: packed halves, upper-only gate clamp,
    # symmetric up clamp, alpha sigmoid, beta, routing exactly once.
    expected = torch.zeros_like(x)
    for row in range(active):
        for slot in range(args.topk):
            expert = int(ids[row, slot].item())
            projected = x[row] @ w1[expert].T
            gate, up = projected.chunk(2)
            gate = gate.clamp(max=7)
            up = up.clamp(-7, 7)
            activated = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
            expected[row] += (activated @ w2[expert].T) * probs[row, slot]
    torch.testing.assert_close(actual[:active], expected[:active], atol=0.02, rtol=0.05)
    assert torch.isfinite(actual[:active]).all()
    print(
        f"rank={rank} mode={args.mode} active={active} hidden={args.hidden} PASS",
        flush=True,
    )
    dist.barrier()


def main(argv=None, *, group=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mc2", "alltoall"), required=True)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--intermediate", type=int, default=16)
    args = parser.parse_args(argv)
    if group is None:
        with _communication_group() as group:
            _run_case(args, group)
    else:
        _run_case(args, group)


@pytest.fixture(scope="module")
def communication_group():
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("requires torchrun with at least two NPU ranks")
    from vllm.platforms import current_platform

    if (
        getattr(current_platform, "vendor_name", None) != "ascend"
        or current_platform.device_type != "npu"
    ):
        pytest.skip("requires the Ascend FL platform")
    pytest.importorskip("torch_npu")
    with _communication_group() as group:
        yield group


@pytest.mark.gpu
@pytest.mark.multi_gpu
@pytest.mark.functional
@pytest.mark.parametrize("mode", ("mc2", "alltoall"))
def test_ascend_bf16_moe_communication(mode, communication_group):
    main(["--mode", mode], group=communication_group)


if __name__ == "__main__":
    main()


@pytest.mark.parametrize(
    "activation,alpha,beta,limit,error",
    [
        ("swigluoai_uninterleave", None, 1.0, 7.0, ValueError),
        ("swigluoai_uninterleave", 1.702, None, 7.0, ValueError),
        ("swigluoai_uninterleave", 1.702, 1.0, None, ValueError),
        ("silu", 2.0, None, None, NotImplementedError),
        ("silu", None, 1.0, None, NotImplementedError),
        ("unknown", None, None, None, NotImplementedError),
    ],
)
@pytest.mark.parametrize("entry", ["grouped", "routed"])
def test_activation_rejected_before_device_work(
    activation, alpha, beta, limit, error, entry, monkeypatch
):
    from vllm_fl.dispatch.backends.vendor.ascend.impl import fused_moe

    def unexpected(*args, **kwargs):
        pytest.fail("device work started before activation validation")

    monkeypatch.setattr(fused_moe.torch_npu, "npu_grouped_matmul", unexpected)
    monkeypatch.setattr(fused_moe.torch_npu, "npu_moe_init_routing_v2", unexpected)
    kwargs = dict(activation=activation, alpha=alpha, beta=beta, limit=limit)
    with pytest.raises(error):
        if entry == "grouped":
            fused_moe.bf16_grouped_experts(
                None, None, None, group_list=None, group_list_type=1, **kwargs
            )
        else:
            fused_moe._torch_fused_experts_impl(
                None, None, None, None, None, expert_map=object(), **kwargs
            )
