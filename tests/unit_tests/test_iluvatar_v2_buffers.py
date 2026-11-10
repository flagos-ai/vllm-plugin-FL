from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.iluvatar.worker import (
    _is_uva_available,
    patch_v2_uva,
)


@pytest.mark.parametrize("vendor", ["iluvatar", "nvidia", "metax"])
def test_patch_is_vendor_scoped(monkeypatch, vendor):
    import vllm.platforms
    from vllm.v1.worker.gpu import buffer_utils

    original = lambda: True
    monkeypatch.setattr(buffer_utils, "is_uva_available", original)
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name=vendor)
    )
    patch_v2_uva()
    patched = buffer_utils.is_uva_available
    patch_v2_uva()
    assert buffer_utils.is_uva_available is patched
    assert patched() is (vendor != "iluvatar")
    assert patched is (_is_uva_available if vendor == "iluvatar" else original)


@pytest.fixture
def explicit_copy_buffers(monkeypatch):
    import vllm.platforms
    from vllm.v1.worker.gpu import buffer_utils

    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(vendor_name="iluvatar", device_type="cpu"),
    )
    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: True)
    patch_v2_uva()
    return buffer_utils


def test_pool_rotation_growth_and_partial_copy(explicit_copy_buffers):
    buffers = explicit_copy_buffers
    pool = buffers.UvaBufferPool(2, torch.int32, max_concurrency=2)
    assert pool._buffer_cls is buffers.NonUvaBuffer
    first = pool.copy_to_uva([3, 7]).clone()
    grown = pool.copy_to_uva([9, 8, 7, 6, 5])
    assert first.tolist() == [3, 7]
    assert grown.tolist() == [9, 8, 7, 6, 5]
    assert pool.copy_to_uva([11]).tolist() == [11]
    destination = torch.empty(3, dtype=torch.int32)
    assert pool.copy_to_gpu(torch.tensor([2, 4, 6]), destination) is destination
    assert destination.tolist() == [2, 4, 6]


def test_backed_tensor_synchronizes_updates(explicit_copy_buffers):
    backed = explicit_copy_buffers.UvaBackedTensor(4, torch.int64)
    backed.np[:] = [5, 4, 3, 2]
    assert backed.copy_to_uva().tolist() == [5, 4, 3, 2]
    backed.np[:2] = [17, 19]
    assert backed.copy_to_uva(2).tolist() == [17, 19]


def test_large_cold_tensor_does_not_require_uva(explicit_copy_buffers):
    staged = explicit_copy_buffers.StagedWriteTensor(
        (2, 4), torch.int32, torch.device("cpu"), uva_instead_of_gpu=True
    )
    assert staged.write_contents is None
    assert staged.gpu.shape == (2, 4)
    assert staged.write_indices._buffer_cls is explicit_copy_buffers.NonUvaBuffer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")
def test_device_copies_and_staged_writes(monkeypatch):
    import vllm.platforms
    from vllm.v1.worker.gpu import buffer_utils

    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(vendor_name="iluvatar", device_type="cuda"),
    )
    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: True)
    patch_v2_uva()
    pool = buffer_utils.UvaBufferPool(2, torch.int32, max_concurrency=3)
    for size in [2, 9, 1, 17, 0, 8]:
        expected = list(range(size))
        assert pool.copy_to_uva(expected).cpu().tolist() == expected
    staged = buffer_utils.StagedWriteTensor(
        (3, 8), torch.int32, torch.device("cuda"), uva_instead_of_gpu=True
    )
    staged.stage_write(1, 2, [5, 9, 13])
    staged.apply_write()
    assert staged.gpu[1].cpu().tolist() == [0, 0, 5, 9, 13, 0, 0, 0]
