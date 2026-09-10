# Copyright (c) 2026 BAAI. All rights reserved.

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm_fl.dispatch.backends.vendor.ascend.patches.accelerator_compat import (
    patch_accelerator_memory,
)


@pytest.mark.gpu
def test_vllm_memory_snapshot_on_ascend():
    from vllm.utils.mem_utils import MemorySnapshot

    patch_accelerator_memory()
    device = torch.device("npu", torch.npu.current_device())
    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats(device)
    before = MemorySnapshot(device=device)
    allocation = torch.ones(1024 * 1024, device=device, dtype=torch.float32)
    after = MemorySnapshot(device=device)

    assert after.torch_allocated >= before.torch_allocated + allocation.nbytes
    assert after.torch_peak >= after.torch_allocated
    assert after.torch_memory >= after.torch_allocated
    assert 0 < after.free_memory <= after.total_memory
    del allocation
    torch.accelerator.empty_cache()
