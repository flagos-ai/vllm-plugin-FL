# Copyright (c) 2026 BAAI. All rights reserved.

"""Validate GCU MoE alignment using independent CPU route/block invariants."""

import hashlib
import importlib.metadata
import inspect
from collections import Counter
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.flaggems]

_NUM_EXPERTS = 256
_TOPK = 8


def _routes(num_tokens, dtype, *, skewed=False, rotation=0):
    # Distinct experts within each token, with both balanced and sparse routing.
    routes = torch.arange(num_tokens * _TOPK, device="cpu", dtype=torch.int64)
    active_experts = 32 if skewed else _NUM_EXPERTS
    return (
        ((routes * 17) % active_experts + rotation)
        .remainder(_NUM_EXPERTS)
        .to(dtype)
        .reshape(num_tokens, _TOPK)
    )


def _provider_identity(export, provider):
    # Observe only the provider actually called; this is a single-file byte
    # observation, not verification of an entire installed package.
    source = inspect.getsourcefile(inspect.unwrap(provider))
    assert source is not None, "The selected MoE provider must have a source file"
    path = Path(source)
    before = path.stat()
    with path.open("rb") as stream:
        raw = stream.read()
        eof = stream.read(1) == b""
    after = path.stat()
    stable = (
        before.st_dev == after.st_dev
        and before.st_ino == after.st_ino
        and before.st_size == after.st_size == len(raw)
        and before.st_mtime_ns == after.st_mtime_ns
    )
    return {
        "export": export,
        "module": provider.__module__,
        "qualname": provider.__qualname__,
        "flag_gems_version": importlib.metadata.version("flag-gems"),
        "source_file": str(path),
        "source_bytes": len(raw),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_read_eof": eof,
        "source_stat_stable": stable,
        "source_mtime_ns_before": before.st_mtime_ns,
        "source_mtime_ns_after": after.st_mtime_ns,
    }


def _assert_alignment(outputs, routes_cpu, block_size, device, expert_map, pad_ids):
    assert isinstance(outputs, tuple) and len(outputs) == 3
    sorted_ids, expert_ids, num_post_pad = outputs
    for output in outputs:
        assert output.device == device
        assert output.dtype == torch.int32
        assert output.ndim == 1
    assert num_post_pad.numel() == 1
    if pad_ids:
        assert sorted_ids.numel() % block_size == 0

    flat_routes = routes_cpu.reshape(-1).to(torch.int64)
    num_routes = flat_routes.numel()
    counts = Counter(flat_routes.tolist())
    expected_blocks = {
        expert: (count + block_size - 1) // block_size
        for expert, count in counts.items()
    }
    expected_post_pad = sum(expected_blocks.values()) * block_size
    actual_post_pad = int(num_post_pad.cpu().item())
    assert actual_post_pad == expected_post_pad
    assert 0 < actual_post_pad <= sorted_ids.numel()
    num_blocks = actual_post_pad // block_size
    assert num_blocks <= expert_ids.numel()

    # Only the post-pad region is a result. Unused expert_ids allocation may
    # contain provider-specific initialization and is deliberately not compared.
    active_ids = sorted_ids[:actual_post_pad].cpu().to(torch.int64)
    active_experts = expert_ids[:num_blocks].cpu().to(torch.int64)
    assert bool(((active_ids >= 0) & (active_ids <= num_routes)).all())
    valid = active_ids < num_routes
    torch.testing.assert_close(
        torch.sort(active_ids[valid]).values,
        torch.arange(num_routes, device="cpu", dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    assert int((active_ids == num_routes).sum()) == actual_post_pad - num_routes

    mapping = list(range(_NUM_EXPERTS)) if expert_map is None else expert_map.tolist()
    inverse_mapping = {mapped: original for original, mapped in enumerate(mapping)}
    assert len(inverse_mapping) == _NUM_EXPERTS
    mapped_expected_blocks = {
        mapping[expert]: blocks for expert, blocks in expected_blocks.items()
    }
    assert Counter(active_experts.tolist()) == mapped_expected_blocks
    for block_index, mapped_expert in enumerate(active_experts.tolist()):
        block = active_ids[block_index * block_size : (block_index + 1) * block_size]
        block_valid = block[block < num_routes]
        assert block_valid.numel() > 0, "An active expert block cannot be all padding"
        original_expert = inverse_mapping[mapped_expert]
        assert bool((flat_routes[block_valid] == original_expert).all())


def _check_moe_align_gcu(
    backend,
    device,
    monkeypatch,
    record_property,
    num_tokens,
    block_size,
    dtype,
    skewed,
    pad_ids,
    mapped,
    *,
    use_graph=False,
):
    if backend != "enflame":
        pytest.skip("Requires the Enflame GCU backend")
    assert device.type == "gcu", "Enflame tests must use an actual GCU device"
    assert torch.gcu.is_available(), "The selected GCU runtime is unavailable"

    # Missing providers on GCU fail. Observe the public export actually selected
    # while preserving its real native implementation and call arguments.
    import flag_gems
    import flaggems_vllm

    from vllm_fl.dispatch.backends.flaggems.impl.fused_moe import (
        moe_align_block_size_flaggems,
    )

    calls = []

    def observe(export, provider):
        def call(*args, **kwargs):
            if not calls:
                identity = _provider_identity(export, provider)
                record_property("gcu_moe_align_provider", identity)
                assert identity["source_read_eof"]
                assert identity["source_stat_stable"]
            calls.append((export, provider.__module__))
            return provider(*args, **kwargs)

        return call

    for package in (flag_gems, flaggems_vllm):
        name = "moe_align_block_size_triton"
        provider = getattr(package, name)
        monkeypatch.setattr(
            package, name, observe(f"{package.__name__}.{name}", provider)
        )

    record_property(
        "gcu_moe_align_case",
        {
            "adapter": "moe_align_block_size_flaggems",
            "device": str(device),
            "num_tokens": num_tokens,
            "topk": _TOPK,
            "num_experts": _NUM_EXPERTS,
            "block_size": block_size,
            "dtype": str(dtype),
            "skewed": skewed,
            "pad_sorted_ids": pad_ids,
            "expert_map": mapped,
            "graph": use_graph,
        },
    )
    initial_routes = _routes(num_tokens, dtype, skewed=skewed)
    topk_ids = initial_routes.to(device)
    map_cpu = (
        torch.arange(_NUM_EXPERTS - 1, -1, -1, dtype=torch.int32, device="cpu")
        if mapped
        else None
    )
    map_device = None if map_cpu is None else map_cpu.to(device)

    def apply():
        return moe_align_block_size_flaggems(
            topk_ids,
            block_size,
            _NUM_EXPERTS,
            expert_map=map_device,
            pad_sorted_ids=pad_ids,
        )

    if use_graph:
        apply()  # Compile before capture, using the same shapes and options.
        torch.gcu.synchronize()
        graph = torch.gcu.GCUGraph()
        with torch.gcu.graph(graph):
            outputs = apply()
        for rotation in (13, 97):
            replay_routes = _routes(num_tokens, dtype, skewed=skewed, rotation=rotation)
            topk_ids.copy_(replay_routes)
            # Reset captured outputs so a no-op or partial replay cannot pass
            # from values left by capture or an earlier replay.
            for output in outputs:
                output.fill_(-1)
            graph.replay()
            torch.gcu.synchronize()
            _assert_alignment(
                outputs, replay_routes, block_size, device, map_cpu, pad_ids
            )
            torch.testing.assert_close(topk_ids.cpu(), replay_routes, rtol=0, atol=0)
        record_property("gcu_moe_align_graph_replays", 2)
    else:
        outputs = apply()
        torch.gcu.synchronize()
        _assert_alignment(outputs, initial_routes, block_size, device, map_cpu, pad_ids)
        torch.testing.assert_close(topk_ids.cpu(), initial_routes, rtol=0, atol=0)

    assert calls, "The production adapter did not invoke an observed provider"
    assert {export for export, _ in calls} == {"flag_gems.moe_align_block_size_triton"}
    # SpecOpRegistrar loads vendor modules under the short _enflame alias.
    # Accept that official namespace as well as the fully qualified import.
    assert all(
        module.startswith(("_enflame.", "flag_gems.runtime.backend._enflame."))
        for _, module in calls
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=["i32", "i64"])
@pytest.mark.parametrize(
    "num_tokens,block_size,skewed,pad_ids,mapped",
    [
        pytest.param(1, 32, False, False, False, id="decode-one"),
        pytest.param(17, 64, False, True, False, id="decode-padded"),
        pytest.param(513, 32, True, False, True, id="prefill-sparse-mapped"),
        pytest.param(2049, 64, False, True, False, id="prefill-large-padded"),
    ],
)
def test_moe_align_gcu_routes(
    backend,
    device,
    monkeypatch,
    record_property,
    num_tokens,
    block_size,
    dtype,
    skewed,
    pad_ids,
    mapped,
):
    _check_moe_align_gcu(
        backend,
        device,
        monkeypatch,
        record_property,
        num_tokens,
        block_size,
        dtype,
        skewed,
        pad_ids,
        mapped,
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=["i32", "i64"])
def test_moe_align_gcu_graph(backend, device, monkeypatch, record_property, dtype):
    _check_moe_align_gcu(
        backend,
        device,
        monkeypatch,
        record_property,
        513,
        64,
        dtype,
        True,
        True,
        True,
        use_graph=True,
    )
