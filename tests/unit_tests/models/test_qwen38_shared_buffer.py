# SPDX-License-Identifier: Apache-2.0
"""Real CPU multiprocess publication, lifetime and producer-failure checks."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def load_buffer():
    path = (
        Path(__file__).resolve().parents[3]
        / "vllm_fl/models/qwen3_8_flash_next/host_lookup.py"
    )
    spec = importlib.util.spec_from_file_location("qwen_shared_buffer_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SharedNgramBuffer


def publish_worker(rank, rendezvous, result_dir):
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=3
    )
    group = SimpleNamespace(
        rank_in_group=rank,
        world_size=3,
        ranks=list(range(3)),
        cpu_group=dist.group.WORLD,
    )
    Buffer = load_buffer()
    buffer = Buffer(group, 17, 32, torch.bfloat16)
    destination = torch.empty(17, 32, dtype=torch.bfloat16)
    lookups = 0
    retained = []
    for epoch, actual in enumerate([0, 1, 17, 6, 3, 16, 2, 17]):
        expected = (
            torch.arange(actual * 32, dtype=torch.float32)
            .reshape(actual, 32)
            .add(epoch * 1000)
            .to(torch.bfloat16)
        )

        def produce(rows, actual=actual, expected=expected):
            nonlocal lookups
            assert rank == 0
            lookups += 1
            rows[:actual].copy_(expected)
            return actual

        destination.fill_(-1)
        got = buffer.exchange(produce, destination, 17)
        assert got == actual
        torch.testing.assert_close(destination[:actual], expected, rtol=0, atol=0)
        assert torch.all(destination[actual:] == -1)
        retained.append((destination.clone(), actual, expected.clone()))
    # Saved consumer staging does not alias a reused shared publication.
    for saved, actual, expected in retained:
        torch.testing.assert_close(saved[:actual], expected, rtol=0, atol=0)

    def fail(_):
        raise ValueError("intentional producer fixture failure")

    try:
        buffer.exchange(fail, destination, 17)
    except RuntimeError as exc:
        assert "producer failed" in str(exc)
    else:
        raise AssertionError("producer error was not visible to every rank")
    assert lookups == (8 if rank == 0 else 0)
    assert buffer.productions == lookups and buffer.consumptions == 8
    buffer.close()
    buffer.close()
    Path(result_dir, f"rank{rank}.txt").write_text(f"{lookups}:8")
    dist.destroy_process_group()


def test_shared_result_has_one_producer_and_safe_multi_rank_reuse(tmp_path):
    mp.spawn(
        publish_worker,
        args=(str(tmp_path / "gloo"), str(tmp_path)),
        nprocs=3,
        join=True,
    )
    assert [(tmp_path / f"rank{i}.txt").read_text() for i in range(3)] == [
        "8:8",
        "0:8",
        "0:8",
    ]
