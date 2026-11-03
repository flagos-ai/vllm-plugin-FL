# Copyright (c) 2026 BAAI. All rights reserved.

"""Enflame writes to vLLM 0.32's concatenated K/V cache, including replay."""

import math

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [7, 65537])
@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "graph"])
def test_concatenated_kv_cache_preserves_values_and_padding(
    device, dtype, tokens, use_graph, record_property
):
    if device.type != "gcu":
        pytest.skip("GCU cache kernel requires Enflame")
    from vllm_fl.dispatch.backends.vendor.gcu.impl.reshape_and_cache import (
        reshape_and_cache_flash,
    )

    blocks, block_size, heads, head_size = math.ceil(tokens / 16), 16, 2, 16
    cpu_key = (
        torch.arange(tokens * heads * head_size).reshape(tokens, heads, head_size) % 31
    ).to(dtype)
    cpu_value = -cpu_key
    slots_cpu = torch.arange(tokens, dtype=torch.int64)
    slots_cpu[-1] = -(2**32)
    key, value, slots = cpu_key.to(device), cpu_value.to(device), slots_cpu.to(device)
    backing = torch.full(
        (blocks, block_size, heads, 2 * head_size), -91, dtype=dtype, device=device
    )
    key_cache, value_cache = backing.split(head_size, dim=-1)
    scale = torch.tensor(1.0, device=device)

    def apply():
        reshape_and_cache_flash(
            key, value, key_cache, value_cache, slots, "auto", scale, scale
        )

    def check(revision):
        expected = torch.full(backing.shape, -91, dtype=dtype)
        flat = expected.view(-1, heads, 2 * head_size)
        flat[: tokens - 1, :, :head_size] = cpu_key[: tokens - 1] + revision
        flat[: tokens - 1, :, head_size:] = cpu_value[: tokens - 1] - revision
        torch.testing.assert_close(backing.cpu(), expected, rtol=0, atol=0)

    apply()
    torch.gcu.synchronize()
    check(0)
    if use_graph:
        graph = torch.gcu.GCUGraph()
        with torch.gcu.graph(graph):
            apply()
        for revision in (1, 2):
            key.copy_(cpu_key + revision)
            value.copy_(cpu_value - revision)
            backing.fill_(-91)
            graph.replay()
            torch.gcu.synchronize()
            check(revision)
        record_property("gcu_kv_cache_graph_replays", 2)
