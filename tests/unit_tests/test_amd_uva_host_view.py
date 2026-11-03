"""Contracts for the AMD ROCm zero-copy host view.

``_C::get_cuda_view_from_cpu_tensor`` has no kernel in an empty-mode install,
so AMD supplies it from a HIP mapping. The implementation is loaded straight
from its source file: it only needs ctypes/torch, and loading it this way keeps
these contracts runnable without importing the whole plugin.

The two offset/lifetime cases below are regressions. The view is built from a
raw pointer, which gives neither bounds nor ownership checking, so getting
either wrong corrupts data silently rather than raising.
"""

import gc
import importlib.util
from pathlib import Path

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "vllm_fl/dispatch/backends/vendor/amd/impl/uva.py"
)


def _load_uva_module():
    spec = importlib.util.spec_from_file_location("_amd_uva_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def uva():
    assert SOURCE.is_file(), SOURCE
    return _load_uva_module()


def _pinned(*args, **kwargs):
    try:
        return torch.empty(*args, **kwargs).pin_memory()
    except RuntimeError as exc:  # pragma: no cover - no pinned-memory support
        pytest.skip(f"pinned host memory unavailable: {exc}")


@pytest.mark.gpu
@pytest.mark.parametrize("lo,hi", [(0, 10), (2, 5), (7, 10), (1, 2)])
def test_storage_offset_is_applied_once(uva, lo, hi):
    """A sliced tensor's view must start where the slice starts.

    ``tensor.data_ptr()`` already includes the storage offset. Mapping that
    address and then also passing ``storage_offset`` to ``set_`` shifted the
    view twice, so a pinned ``arange(10)[2:5]`` read ``[4, 5, 6]``; a slice
    near the end of the allocation read out of bounds.
    """
    base = _pinned(10, dtype=torch.float32)
    base.copy_(torch.arange(10, dtype=torch.float32))
    sliced = base[lo:hi]

    view = uva.get_cuda_view_from_cpu_tensor_amd(sliced)

    assert view.cpu().tolist() == sliced.tolist()


@pytest.mark.gpu
def test_source_allocation_outlives_the_view(uva):
    """The view must keep the pinned allocation it points into alive.

    ``_construct_storage_from_data_pointer`` wraps a raw pointer with no
    deleter. vLLM's UVA offloader assigns the view to ``p.data`` and drops its
    own handle on the pinned tensor, so without an explicit retain the
    allocation is recycled while the parameter still points at it.
    """

    def build_view_and_drop_source():
        source = _pinned(8, dtype=torch.float32)
        source.fill_(7.0)
        return uva.get_cuda_view_from_cpu_tensor_amd(source)

    view = build_view_and_drop_source()
    gc.collect()

    # Churn pinned allocations so a freed region would be handed out again.
    for _ in range(64):
        reused = _pinned(8, dtype=torch.float32)
        reused.fill_(-1.0)

    assert view.cpu().tolist() == [7.0] * 8


@pytest.mark.gpu
def test_view_is_zero_copy_not_a_snapshot(uva):
    """Device writes must be visible on the host: this is a view, not a copy."""
    host = _pinned(4, dtype=torch.float32)
    host.fill_(0.0)

    view = uva.get_cuda_view_from_cpu_tensor_amd(host)
    view.add_(5.0)
    torch.cuda.synchronize()

    assert host.tolist() == [5.0] * 4


@pytest.mark.gpu
def test_multidimensional_view_preserves_shape_and_rows(uva):
    matrix = _pinned((3, 4), dtype=torch.float32)
    matrix.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    rows = matrix[1:3]

    view = uva.get_cuda_view_from_cpu_tensor_amd(rows)

    assert view.shape == rows.shape
    assert view.cpu().tolist() == rows.tolist()


def test_rejects_unpinned_and_non_cpu_tensors(uva):
    """The contract requires page-locked host memory; say so rather than crash."""
    with pytest.raises(ValueError, match="pinned"):
        uva.get_cuda_view_from_cpu_tensor_amd(torch.zeros(4))

    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="CPU tensor"):
            uva.get_cuda_view_from_cpu_tensor_amd(torch.zeros(4, device="cuda"))
