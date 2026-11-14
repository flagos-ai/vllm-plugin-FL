# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""Real PTPU verification of native 0.32 combined KV-cache writes."""

import pytest
import torch


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_sunrise_combined_cache_writer_preserves_kv_and_padding(layout):
    pytest.importorskip("torch_ptpu")
    if not torch.ptpu.is_available():
        pytest.skip("Sunrise PTPU is unavailable")
    from flaggems_vllm import reshape_and_cache_flash_strided

    from vllm_fl.dispatch.backends.vendor.sunrise.impl.kv_cache import split_kv_cache

    torch.ptpu.set_device(0)
    device = torch.device("ptpu:0")
    if layout == "NHD":
        logical = torch.zeros(
            2, 16, 2, 16, dtype=torch.bfloat16, device=device
        ).transpose(1, 2)
    else:
        logical = torch.zeros(2, 2, 16, 16, dtype=torch.bfloat16, device=device)
    key_cache, value_cache = split_kv_cache(logical, 8)
    key_cpu = torch.arange(48, dtype=torch.float32).reshape(3, 2, 8).bfloat16()
    value_cpu = key_cpu + 100
    key, value = key_cpu.to(device), value_cpu.to(device)
    slots = torch.tensor([3, -1, 17], dtype=torch.int64, device=device)
    reshape_and_cache_flash_strided(key, value, key_cache, value_cache, slots)
    torch.ptpu.synchronize()
    expected = torch.zeros(2, 2, 16, 16, dtype=torch.bfloat16)
    expected[0, :, 3, :8] = key_cpu[0]
    expected[0, :, 3, 8:] = value_cpu[0]
    expected[1, :, 1, :8] = key_cpu[2]
    expected[1, :, 1, 8:] = value_cpu[2]
    torch.testing.assert_close(logical.cpu(), expected, rtol=0, atol=0)
