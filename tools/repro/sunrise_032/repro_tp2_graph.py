#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""Reproduce the Sunrise/FlagCX TP2 asynchronous shared-pool graph failure.

This is a portable refactor of the separately archived executed probe, and has
NOT been hardware-retested as part of this PR. It loads no model and applies
neither the stream-fence nor synchronous-scheduling experiment. It does not
instantiate a model runner: the V2 environment flag only matches the target
installation and is not evidence of Runner V2 execution.

Only an explicit main() invocation imports accelerator packages or uses devices.
--help works on a CPU-only Python installation. Choose an idle, healthy, directly
connected physical pair yourself. This program does not retry, reset devices, or
manage containers/processes outside its two spawned workers. An external host
timeout does not guarantee native device resources are cleaned up; inspect the
failure before attempting another run.
"""

import argparse
import hashlib
import json
import os
import platform
import tempfile
import uuid
from datetime import timedelta
from importlib import metadata
from pathlib import Path

WORLD_SIZE = 2
WIDTH = 2560
LAYERS = 32
CAPTURE_BATCHES = (2, 1)
BATCH_SEQUENCE = (2, 2, 1, 2, 1, 1, 2, 1)


def visible_pair(value: str) -> str:
    """Validate exactly two explicit physical ordinals without choosing cards."""
    parts = value.split(",")
    if len(parts) != WORLD_SIZE or any(not p.strip().isdigit() for p in parts):
        raise argparse.ArgumentTypeError("provide two physical IDs, e.g. 4,5")
    ids = [int(p.strip()) for p in parts]
    if len(set(ids)) != WORLD_SIZE:
        raise argparse.ArgumentTypeError("the two physical IDs must be different")
    return ",".join(str(i) for i in ids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible-devices", required=True, type=visible_pair)
    parser.add_argument(
        "--ta-close-graph-memory",
        choices=("1", "unset"),
        default="1",
        help="record the gate setting (default: 1) or the historical unset setting",
    )
    parser.add_argument("--init-timeout-seconds", type=int, default=30)
    parser.add_argument("--rendezvous-dir", type=Path)
    return parser


def environment_updates(args, environ) -> dict[str, str]:
    """Validate first; never silently overwrite the caller's device selection."""
    updates = {}
    current_mask = environ.get("TANG_VISIBLE_DEVICES")
    if current_mask is None:
        updates["TANG_VISIBLE_DEVICES"] = args.visible_devices
    elif visible_pair(current_mask) != args.visible_devices:
        raise ValueError("--visible-devices conflicts with TANG_VISIBLE_DEVICES")

    if args.ta_close_graph_memory == "unset":
        if "TA_CLOSE_GRAPH_MEMORY" in environ:
            raise ValueError("unset mode requires TA_CLOSE_GRAPH_MEMORY to be absent")
    elif environ.get("TA_CLOSE_GRAPH_MEMORY", "1") != "1":
        raise ValueError("this variant requires TA_CLOSE_GRAPH_MEMORY=1")
    else:
        updates.setdefault("TA_CLOSE_GRAPH_MEMORY", "1")

    for name, expected in (("PCCL_PROTO", "Simple"), ("GEMS_VENDOR", "sunrise")):
        if environ.get(name, expected) != expected:
            raise ValueError(f"this variant requires {name}={expected}")
        if name not in environ:
            updates[name] = expected
    plugins = environ.get("VLLM_PLUGINS")
    if plugins is not None and "fl" not in {p.strip() for p in plugins.split(",")}:
        raise ValueError("VLLM_PLUGINS must include fl")
    if plugins is None:
        updates["VLLM_PLUGINS"] = "fl"
    if environ.get("VLLM_USE_V2_MODEL_RUNNER", "1") != "1":
        raise ValueError("this target environment requires VLLM_USE_V2_MODEL_RUNNER=1")
    if "VLLM_USE_V2_MODEL_RUNNER" not in environ:
        updates["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    return updates


def new_rendezvous(directory: Path) -> Path:
    """Use one parent-generated UUID for both ranks; never remove old stores."""
    if not directory.is_dir():
        raise ValueError(f"rendezvous directory does not exist: {directory}")
    path = directory.resolve() / f"sunrise_tp2_graph_{os.getpid()}_{uuid.uuid4().hex}"
    if path.exists():
        raise FileExistsError(f"refusing an existing rendezvous store: {path}")
    return path


def emit(stage: str, **details) -> None:
    print(json.dumps({"stage": stage, **details}, sort_keys=True), flush=True)


def versions() -> dict[str, str | None]:
    result = {"python": platform.python_version()}
    for name in (
        "torch",
        "torch-ptpu",
        "vllm",
        "vllm-plugin-fl",
        "flagtree",
        "triton",
        "flag_gems",
        "flaggems-vllm",
        "flagcx",
    ):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def worker(rank: int, rendezvous: str, visible_devices: str, init_timeout: int) -> None:
    # Imports deliberately stay inside the user-invoked hardware path.
    import torch
    import torch.distributed as dist

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state as ps
    from vllm.platforms import current_platform

    from vllm_fl.dispatch.backends.vendor.sunrise.patch import apply_sunrise_patches

    def mark(stage, **details):
        emit(stage, rank=rank, **details)

    assert visible_pair(os.environ["TANG_VISIBLE_DEVICES"]) == visible_devices
    assert current_platform.device_type == "ptpu"
    assert torch.ptpu.device_count() == WORLD_SIZE
    device = torch.device(f"ptpu:{rank}")
    current_platform.set_device(device)
    apply_sunrise_patches()
    mark("init_start", current_device=torch.ptpu.current_device())
    ps.init_distributed_environment(
        WORLD_SIZE,
        rank,
        Path(rendezvous).as_uri(),
        rank,
        backend="pccl",
        timeout=timedelta(seconds=init_timeout),
    )
    config = VllmConfig()
    config.parallel_config.disable_custom_all_reduce = True
    with set_current_vllm_config(config):
        ps.initialize_model_parallel(
            tensor_model_parallel_size=WORLD_SIZE, backend="pccl"
        )
    tp = ps.get_tp_group()
    comm = tp.device_communicator.pyflagcx_comm
    assert comm is not None and not comm.disabled
    mark("init_done", communicator=type(comm).__name__)
    cpu_group = tp.cpu_group
    static_inputs = {
        n: torch.full((n, WIDTH), rank + 1.0, dtype=torch.bfloat16, device=device)
        for n in CAPTURE_BATCHES
    }

    def forward(x):
        value = x
        for _ in range(LAYERS):
            value = comm.all_reduce(value)
            value.mul_(0.5)
        return value

    for n in CAPTURE_BATCHES:
        warmup_output = forward(static_inputs[n])
        torch.ptpu.synchronize()
        torch.testing.assert_close(
            warmup_output.cpu(), torch.full((n, WIDTH), 1.5, dtype=torch.bfloat16)
        )
    dist.barrier(group=cpu_group)
    pool = torch.ptpu.graph_pool_handle()
    graphs, outputs = {}, {}
    # Both shapes share ONE capture stream and ONE pool, as in native Runner V2.
    with ps.graph_capture(device) as context:
        assert int(comm._current_stream().ptpu_stream) == int(
            context.stream.ptpu_stream
        )
        for n in CAPTURE_BATCHES:
            graph = torch.ptpu.PTPUGraph()
            mark(
                "capture_begin",
                batch=n,
                pool=str(pool),
                layers=LAYERS,
                stream=int(context.stream.ptpu_stream),
            )
            with torch.ptpu.graph(graph, pool=pool, stream=context.stream):
                outputs[n] = forward(static_inputs[n])
            graphs[n] = graph
            mark("capture_end", batch=n, native_type=type(graph).__name__)
    torch.ptpu.synchronize()
    dist.barrier(group=cpu_group)
    mark(
        "async_sequence_begin",
        batches=BATCH_SEQUENCE,
        stream=int(torch.ptpu.current_stream().ptpu_stream),
    )
    receipts = []
    for iteration, batch in enumerate(BATCH_SEQUENCE):
        # Preserve the failure: no per-step barrier, wait_stream or synchronize.
        static_inputs[batch].fill_(rank + 1.0 + 2 * iteration)
        gathered = torch.empty(
            (WORLD_SIZE * batch, WIDTH), dtype=torch.bfloat16, device=device
        )
        mark("async_replay_begin", iteration=iteration, batch=batch)
        graphs[batch].replay()
        mark("async_gather_begin", iteration=iteration, batch=batch)
        comm.all_gather(gathered, outputs[batch])
        receipts.append((iteration, batch, outputs[batch].clone(), gathered))
        mark("async_enqueued", iteration=iteration, batch=batch)
    mark("final_synchronize_begin")
    torch.ptpu.synchronize()
    mark("final_synchronize_end")
    for iteration, batch, actual, gathered in receipts:
        expected = 1.5 + 2 * iteration
        torch.testing.assert_close(
            actual.cpu(), torch.full((batch, WIDTH), expected, dtype=torch.bfloat16)
        )
        torch.testing.assert_close(
            gathered.cpu(),
            torch.full((WORLD_SIZE * batch, WIDTH), expected, dtype=torch.bfloat16),
        )
        mark("numeric_pass", iteration=iteration, batch=batch, expected=expected)
    dist.barrier(group=cpu_group)
    mark("all_pass")
    ps.destroy_model_parallel()
    dist.destroy_process_group()


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.init_timeout_seconds <= 0:
        parser.error("--init-timeout-seconds must be positive")
    try:
        updates = environment_updates(args, os.environ)
        rendezvous = new_rendezvous(args.rendezvous_dir or Path(tempfile.gettempdir()))
    except (ValueError, argparse.ArgumentTypeError, FileExistsError) as exc:
        return parser.error(str(exc))
    # Process-local only, after explicit device selection and full validation.
    os.environ.update(updates)
    emit(
        "provenance",
        variant="async_shared_pool_no_fence",
        portable_refactor=True,
        model_loaded=False,
        model_runner_instantiated=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions=versions(),
        visible_devices=os.environ["TANG_VISIBLE_DEVICES"],
        TA_CLOSE_GRAPH_MEMORY=os.environ.get("TA_CLOSE_GRAPH_MEMORY"),
        PCCL_PROTO=os.environ["PCCL_PROTO"],
        GEMS_VENDOR=os.environ["GEMS_VENDOR"],
        parent_pid=os.getpid(),
        rendezvous=str(rendezvous),
        dtype="bfloat16",
        width=WIDTH,
        layers=LAYERS,
        capture_batches=CAPTURE_BATCHES,
        batches=BATCH_SEQUENCE,
    )
    try:
        import torch.multiprocessing as mp

        mp.spawn(
            worker,
            args=(str(rendezvous), args.visible_devices, args.init_timeout_seconds),
            nprocs=WORLD_SIZE,
            join=True,
        )
    except Exception as exc:
        emit(
            "failed",
            error=str(exc),
            automatic_retry=False,
            native_device_cleanup_guaranteed=False,
        )
        raise
    emit("all_workers_pass", workers=WORLD_SIZE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
