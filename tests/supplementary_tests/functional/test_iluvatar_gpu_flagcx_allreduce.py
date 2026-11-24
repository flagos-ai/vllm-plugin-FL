# Copyright 2025 BAAI. All rights reserved.
"""Functional 2-process FlagCX all-reduce on the Iluvatar BI-V150.

The single-rank supplementary test
(``tests/supplementary_tests/unit/dispatch/test_iluvatar_gpu_communicator_allreduce.py``)
pins ``IluvatarCommunicatorFL.__init__`` on one card, but the collective
itself needs a second rank: production TP>1 execution goes
``GroupCoordinator.all_reduce`` → ``CommunicatorFL.all_reduce`` →
``PyFlagcxCommunicator.all_reduce`` (the FlagCX C library), and the vision
fp32 upcast rides on top of that path.

This test spawns two real processes, each bound to its own BI-V150 with a
gloo rendezvous group, then:

1. constructs ``IluvatarCommunicatorFL`` with ``world_size == 2`` so
   ``CommunicatorFL.__init__`` builds a real ``PyFlagcxCommunicator``
   (unique-id exchange through the gloo group, ``flagcxCommInitRank`` on the
   device);
2. runs fp32/bf16 all-reduce through the FL wrapper on device tensors and
   checks the numerical result against an independent per-rank oracle
   (each rank knows both inputs, so the expected sum is computed in torch);
3. runs the vision-context bf16 round trip on device tensors: the collective
   must observe fp32 (upcast by the wrapper), and the result must come back
   bf16 matching an fp32-accumulated oracle.

Tolerances follow ``tests/platforms/iluvatar.yaml``: default rtol/atol
1e-3, bfloat16 1e-2. The subprocess result (including any assertion failure
with its message) is shipped back through a multiprocessing Queue so the
parent pytest process reports a clean pass/fail.
"""

from __future__ import annotations

import multiprocessing as mp
import uuid

import pytest
import torch

pytestmark = [
    pytest.mark.supplementary,
    pytest.mark.platforms("iluvatar"),
    pytest.mark.gpu,
    pytest.mark.serial,
]

# tests/platforms/iluvatar.yaml tolerance.inference
RTOL_DEFAULT = 1e-3
ATOL_DEFAULT = 1e-3
RTOL_BF16 = 1e-2
ATOL_BF16 = 1e-2

WORLD_SIZE = 2
SECRET = uuid.uuid4().hex


def _rank_worker(rank: int, queue, init_file: str):
    """One FlagCX rank. Runs in a spawned subprocess with its own device."""
    import torch.distributed as dist

    try:
        # Bind this rank to its own device before any CUDA context exists.
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=WORLD_SIZE,
        )
        from vllm_fl.dispatch.backends.vendor.iluvatar.communicator import (
            IluvatarCommunicatorFL,
        )
        from vllm_fl.dispatch.backends.vendor.iluvatar.vision_precision import (
            _vision_forward,
        )

        device = torch.device("cuda", rank % torch.cuda.device_count())
        comm = IluvatarCommunicatorFL(
            cpu_group=dist.group.WORLD,
            device=device,
            device_group=dist.group.WORLD,
            unique_name="tp:0",
        )
        results = {}

        # --- construction contract -----------------------------------------
        if comm.pyflagcx_comm is None or comm.pyflagcx_comm.disabled:
            raise AssertionError(
                f"rank {rank}: FlagCX communicator did not initialize: "
                f"{getattr(comm.pyflagcx_comm, 'init_error', None) if comm.pyflagcx_comm else 'None'}"
            )
        results["flagcx_active"] = True
        results["world_size"] = comm.world_size

        # --- fp32 all-reduce vs oracle ------------------------------------
        generator = torch.Generator().manual_seed(4104 + rank)
        local = torch.randn(64, 32, generator=generator).to(device)
        local_snapshot = local.clone()
        # exchange the peer's input through gloo (independent of FlagCX)
        gathered = [torch.empty_like(local.cpu()) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered, local.cpu())
        expected_fp32 = (gathered[0] + gathered[1]).to(device)

        out = comm.all_reduce(local)
        results["fp32_dtype"] = str(out.dtype)
        results["fp32_ok"] = bool(
            torch.allclose(
                out.cpu(), expected_fp32.cpu(), rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT
            )
        )

        # --- bf16 all-reduce vs oracle -------------------------------------
        local_bf16 = torch.randn(32, 16, generator=generator).bfloat16().to(device)
        gathered_bf = [torch.empty_like(local_bf16.cpu()) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered_bf, local_bf16.cpu())
        expected_bf16 = (gathered_bf[0].float() + gathered_bf[1].float()).to(device)

        out_bf16 = comm.all_reduce(local_bf16)
        results["bf16_dtype"] = str(out_bf16.dtype)
        results["bf16_ok"] = bool(
            torch.allclose(
                out_bf16.float().cpu(),
                expected_bf16.float().cpu(),
                rtol=RTOL_BF16,
                atol=ATOL_BF16,
            )
        )

        # --- vision fp32 upcast round trip ---------------------------------
        vision_local = torch.randn(32, 16, generator=generator).bfloat16().to(device)
        gathered_v = [torch.empty_like(vision_local.cpu()) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered_v, vision_local.cpu())
        expected_vision = (gathered_v[0].float() + gathered_v[1].float()).to(device)

        token = _vision_forward.set(True)
        try:
            out_vision = comm.all_reduce(vision_local)
        finally:
            _vision_forward.reset(token)
        results["vision_dtype"] = str(out_vision.dtype)
        results["vision_ok"] = bool(
            torch.allclose(
                out_vision.float().cpu(),
                expected_vision.float().cpu(),
                rtol=RTOL_BF16,
                atol=ATOL_BF16,
            )
        )

        # --- input immutability ---------------------------------------------
        results["input_untouched"] = bool(torch.equal(local.cpu(), local_snapshot))
        queue.put({"rank": rank, "ok": True, "results": results})
    except BaseException as exc:  # surface any failure to the parent
        import traceback

        queue.put(
            {"rank": rank, "ok": False, "error": f"{exc}", "traceback": traceback.format_exc()}
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture(scope="module")
def flagcx_results(tmp_path_factory):
    """Spawn the two ranks and collect their results."""
    import torch.multiprocessing as tmp_mp

    if not torch.cuda.is_available():
        pytest.skip("Iluvatar BI-V150 accelerator required")
    if torch.cuda.device_count() < 1:
        pytest.skip("no visible CUDA device")

    # FlagCX needs to be importable: the container ships the corex wheel.
    from vllm_fl.distributed.device_communicators import flagcx as fc

    if fc.FLAGCXLibrary is None and not __import__("os").environ.get("FLAGCX_PATH"):
        # the default import arm needs plugin.interservice; the container
        # exports FLAGCX_PATH in production. Try the wheel layout explicitly.
        import os

        spec_root = None
        try:
            import flagcx as _flagcx_pkg

            spec_root = os.path.dirname(os.path.dirname(_flagcx_pkg.__file__))
            if not (
                os.path.isfile(os.path.join(spec_root, "flagcx", "api.py"))
                and os.path.isfile(
                    os.path.join(spec_root, "flagcx", "lib", "libflagcx.so")
                )
            ):
                spec_root = None
        except ImportError:
            spec_root = None
        if spec_root is None:
            pytest.skip(
                "FlagCX python bindings unavailable (FLAGCX_PATH unset, "
                "no wheel layout found)"
            )
        os.environ["FLAGCX_PATH"] = os.path.join(spec_root, "flagcx")

    init_file = str(
        tmp_path_factory.mktemp("iluvatar_flagcx") / f"init-{SECRET}"
    )
    ctx = tmp_mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    for rank in range(WORLD_SIZE):
        p = ctx.Process(target=_rank_worker, args=(rank, queue, init_file))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=300)

    outcomes = [queue.get(timeout=10) for _ in range(WORLD_SIZE)]
    for p in procs:
        if p.is_alive():
            p.terminate()
    return outcomes


def test_two_rank_flagcx_allreduce(flagcx_results):
    """fp32/bf16 device all-reduce matches the independent gloo oracle."""
    for outcome in flagcx_results:
        assert outcome["ok"], (
            f"rank {outcome['rank']} failed: {outcome.get('error')}\n"
            f"{outcome.get('traceback', '')}"
        )
        results = outcome["results"]
        assert results["flagcx_active"], "FlagCX communicator must initialize"
        assert results["world_size"] == 2
        assert results["fp32_dtype"] == "torch.float32"
        assert results["fp32_ok"], "fp32 all-reduce mismatch vs oracle"
        assert results["bf16_dtype"] == "torch.bfloat16"
        assert results["bf16_ok"], "bf16 all-reduce mismatch vs oracle"
        assert results["input_untouched"], "wrapper must not mutate its input"


def test_two_rank_vision_upcast_round_trip(flagcx_results):
    """Inside the vision context bf16 is reduced in fp32 and cast back."""
    for outcome in flagcx_results:
        assert outcome["ok"]
        results = outcome["results"]
        assert results["vision_dtype"] == "torch.bfloat16"
        assert results["vision_ok"], "vision bf16 upcast round trip mismatch"
