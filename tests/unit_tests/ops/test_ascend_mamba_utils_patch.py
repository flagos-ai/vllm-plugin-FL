from types import SimpleNamespace

import torch


def test_ascend_mamba_copy_launches_local_kernel(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.ascend.patches import (
        patch_mamba_utils,
    )

    launch = {}

    class FakeKernel:
        def __getitem__(self, grid):
            launch["grid"] = grid

            def run(*args, **kwargs):
                launch["args"] = args
                launch["kwargs"] = kwargs

            return run

    monkeypatch.setattr(
        patch_mamba_utils,
        "batch_memcpy_kernel",
        FakeKernel(),
    )
    ptrs = SimpleNamespace(shape=(3,))
    sizes = SimpleNamespace(shape=(3,))

    patch_mamba_utils._batch_memcpy_ascend(ptrs, ptrs, sizes)

    assert launch["grid"] == (3,)
    assert launch["args"] == (ptrs, ptrs, sizes)
    assert launch["kwargs"] == {"BLOCK_SIZE": 8192}


def test_tensor_view_copy_handles_overlapping_ranges():
    from vllm_fl.dispatch.backends.vendor.ascend.patches.patch_mamba_utils import (
        _do_mamba_copy_block_torch,
    )

    state = torch.arange(8)
    copy_bufs = SimpleNamespace(
        offset=1,
        _tensor_copy_pairs=[(state[:6], state[2:])],
    )

    _do_mamba_copy_block_torch(copy_bufs)

    assert state.tolist() == [0, 1, 0, 1, 2, 3, 4, 5]
    assert copy_bufs._tensor_copy_pairs == []
