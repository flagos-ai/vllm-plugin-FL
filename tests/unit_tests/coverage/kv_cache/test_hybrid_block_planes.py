# Copyright (c) 2026 BAAI. All rights reserved.

"""Block-plane operations for the Ascend dense hybrid cache.

``copy_block_planes`` and ``zero_block_planes`` are pure-torch row
operations over the byte planes returned by ``allocate_ascend_kv_cache``;
they never touch NPU kernels (the module docstring only explains why they
avoid ``index_copy_`` on uint8 — on CPU every path is plain torch). They
are therefore exercised here with real byte planes and real snapshots:

* copy snapshots source rows before writing (offset views never corrupt
  untouched rows);
* aligned (int32-viewable) planes use ``index_copy_``/``index_fill_``;
* unaligned planes fall back to per-row copies;
* duplicate planes (same storage, K and V halves of one manager tensor)
  are deduplicated so a block is copied/zeroed exactly once;
* empty inputs are no-ops.

The module is loaded through the same stub graph + ``spec_from_file_location``
idiom as ``test_ascend_hybrid_gaps.py``; only the external vllm surface is
faked, the functions under test run unmodified.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch


class _Spec(types.ModuleType):
    """Attributes the hybrid module binds from the vllm stub graph."""

    def __init__(self, name, **attrs):
        super().__init__(name)
        for key, value in attrs.items():
            setattr(self, key, value)


def _install_stubs():
    """Minimal vllm graph for hybrid.py's module-level imports."""
    ifaces = SimpleNamespace(
        AttentionSpec=type("AttentionSpec", (), {"__init__": None}),
        MambaSpec=type("MambaSpec", (), {"__init__": None}),
        UniformTypeKVCacheSpecs=type("UniformTypeKVCacheSpecs", (), {"__init__": None}),
    )
    models = _Spec("vllm.model_executor.models", ModelRegistry=SimpleNamespace())
    math_utils = _Spec("vllm.utils.math_utils", cdiv=lambda a, b: -(-a // b))
    torch_utils = _Spec(
        "vllm.utils.torch_utils",
        STR_DTYPE_TO_TORCH_DTYPE={"int8": torch.int8, "float16": torch.float16},
        get_dtype_size=lambda d: torch.empty((), dtype=d).element_size(),
    )
    utils = _Spec("vllm.utils", math_utils=math_utils, torch_utils=torch_utils)
    iface = _Spec(
        "vllm.v1.kv_cache_interface",
        AttentionSpec=ifaces.AttentionSpec,
        MambaSpec=ifaces.MambaSpec,
        UniformTypeKVCacheSpecs=ifaces.UniformTypeKVCacheSpecs,
    )
    stubs = {
        "vllm": _Spec("vllm"),
        "vllm.model_executor": _Spec("vllm.model_executor", models=models),
        "vllm.model_executor.models": models,
        "vllm.utils": utils,
        "vllm.utils.math_utils": math_utils,
        "vllm.utils.torch_utils": torch_utils,
        "vllm.v1": _Spec("vllm.v1"),
        "vllm.v1.kv_cache_interface": iface,
        "vllm.v1.worker": _Spec("vllm.v1.worker"),
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    return saved


_SAVED = _install_stubs()

import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "vllm_fl"
    / "kv_cache"
    / "ascend"
    / "hybrid.py"
)
_spec = importlib.util.spec_from_file_location("supp_hybrid_plane_ops", _SOURCE)
hybrid = importlib.util.module_from_spec(_spec)
sys.modules["supp_hybrid_plane_ops"] = hybrid
try:
    _spec.loader.exec_module(hybrid)
finally:
    for name, module in _SAVED.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _aligned_plane(rows: int, cols: int) -> torch.Tensor:
    """A uint8 plane whose rows are int32-viewable (cols multiple of 4)."""
    return torch.arange(rows * cols, dtype=torch.uint8).reshape(rows, cols).contiguous()


def _unaligned_plane(rows: int, cols: int) -> torch.Tensor:
    """A plane whose storage offset breaks the int32 view (fallback path).

    One leading pad byte shifts ``storage_offset`` to 1 (not %4), so both
    copy and zero take the per-row fallback; data values are non-zero so
    the assertions can distinguish zeroed rows from untouched ones.
    """
    pad = torch.zeros(rows * cols + 1, dtype=torch.uint8)
    pad[1:] = torch.arange(1, rows * cols + 1, dtype=torch.uint8)
    return pad[1:].reshape(rows, cols)


class TestCopyBlockPlanes:
    """copy_block_planes snapshot semantics over real byte planes."""

    def test_aligned_plane_copies_rows(self):
        """Aligned planes route through index_copy_ on an int32 view."""
        plane = _aligned_plane(6, 8)
        before = plane.clone()
        # Copy block 3 into block 0 and block 0 into block 5 (a swap shape
        # that would corrupt data if the source rows were not snapshotted).
        copies = [[3, 0], [0, 5]]
        hybrid.copy_block_planes([plane], copies)
        assert torch.equal(plane[0], before[3])
        assert torch.equal(plane[5], before[0])
        # Untouched rows are unchanged.
        for row in (1, 2, 3, 4):
            assert torch.equal(plane[row], before[row])

    def test_unaligned_plane_falls_back_to_row_copies(self):
        """Storage-offset planes copy row by row, still snapshot-first."""
        plane = _unaligned_plane(4, 6)
        before = plane.clone()
        copies = [[1, 3], [3, 1]]
        hybrid.copy_block_planes([plane], copies)
        assert torch.equal(plane[3], before[1])
        assert torch.equal(plane[1], before[3])
        assert torch.equal(plane[0], before[0])
        assert torch.equal(plane[2], before[2])

    def test_swap_within_one_plane_preserves_snapshot(self):
        """A↔B swap must not chain (B receiving the new A)."""
        plane = _aligned_plane(8, 4)
        before = plane.clone()
        copies = [[0, 1], [1, 0]]
        hybrid.copy_block_planes([plane], copies)
        assert torch.equal(plane[0], before[1])
        assert torch.equal(plane[1], before[0])

    def test_duplicate_planes_copied_once(self):
        """K/V halves of one manager tensor deduplicate to a single write."""
        manager = torch.arange(4 * 2 * 6, dtype=torch.uint8).reshape(4, 2, 6)
        # Simulate the alias the allocator returns: both planes share
        # storage with the manager so zero/copy sees one deduplicated entry.
        k_view = manager[:, 0]
        v_view = manager[:, 1]
        before = manager.clone()
        hybrid.copy_block_planes([k_view, v_view], [[2, 0]])
        assert torch.equal(manager[0, 0], before[2, 0])
        assert torch.equal(manager[0, 1], before[2, 1])

    def test_empty_inputs_are_noops(self):
        """No planes or no copies returns without touching anything."""
        plane = _aligned_plane(4, 8)
        before = plane.clone()
        hybrid.copy_block_planes([], [[1, 0]])
        hybrid.copy_block_planes([plane], [])
        hybrid.copy_block_planes([plane], None)
        assert torch.equal(plane, before)

    def test_multiple_planes_copied_independently(self):
        """Two independent planes each receive their block copies."""
        a = _aligned_plane(4, 8)
        b = _aligned_plane(4, 8)
        a_before, b_before = a.clone(), b.clone()
        hybrid.copy_block_planes([a, b], [[1, 0]])
        assert torch.equal(a[0], a_before[1])
        assert torch.equal(b[0], b_before[1])
        assert torch.equal(a[1], a_before[1])
        assert torch.equal(b[1], b_before[1])


class TestZeroBlockPlanes:
    """zero_block_planes in-place row zeroing."""

    def test_aligned_plane_zeroed(self):
        """Aligned planes zero rows through index_fill_ on an int32 view."""
        plane = _aligned_plane(6, 8)
        hybrid.zero_block_planes([plane], [1, 4])
        assert torch.equal(plane[1], torch.zeros(8, dtype=torch.uint8))
        assert torch.equal(plane[4], torch.zeros(8, dtype=torch.uint8))
        assert plane[0].max() > 0
        assert plane[2].max() > 0
        assert plane[3].max() > 0
        assert plane[5].max() > 0

    def test_unaligned_plane_zeroed_row_by_row(self):
        """Fallback per-row zeroing for storage-offset planes."""
        plane = _unaligned_plane(4, 6)
        hybrid.zero_block_planes([plane], [2])
        assert torch.equal(plane[2], torch.zeros(6, dtype=torch.uint8))
        assert plane[0].max() > 0
        assert plane[1].max() > 0
        assert plane[3].max() > 0

    def test_empty_inputs_are_noops(self):
        plane = _aligned_plane(4, 8)
        before = plane.clone()
        hybrid.zero_block_planes([], [0])
        hybrid.zero_block_planes([plane], [])
        hybrid.zero_block_planes([plane], None)
        assert torch.equal(plane, before)

    def test_repeat_zero_is_idempotent(self):
        """Zeroing an already-zero row is stable."""
        plane = _aligned_plane(3, 8)
        hybrid.zero_block_planes([plane], [1])
        snapshot = plane.clone()
        hybrid.zero_block_planes([plane], [1])
        assert torch.equal(plane, snapshot)


class TestDeduplicateBlockPlanes:
    """Storage-alias deduplication keying (device, data_ptr, shape, stride)."""

    def test_views_sharing_storage_deduplicate(self):
        """Two views of one tensor collapse to one plane."""
        manager = torch.zeros(4, 2, 8, dtype=torch.uint8)
        planes = [manager[:, 0], manager[:, 1]]
        dedup = hybrid._deduplicate_block_planes(planes)
        # K and V planes have distinct data pointers → both survive.
        assert len(dedup) == 2

    def test_identical_view_deduplicates(self):
        """The very same view object listed twice collapses to one."""
        plane = _aligned_plane(4, 8)
        dedup = hybrid._deduplicate_block_planes([plane, plane])
        assert len(dedup) == 1
        # Byte-plane form: rows at dim 0, flattened columns.
        assert dedup[0].shape == (4, 8)

    def test_planes_with_different_shapes_stay_separate(self):
        """Different shapes never collide even on shared storage."""
        flat = torch.arange(64, dtype=torch.uint8)
        a = flat.reshape(8, 8)
        b = flat.reshape(4, 16)
        dedup = hybrid._deduplicate_block_planes([a, b])
        assert len(dedup) == 2

    def test_plane_view_is_flattened_bytes(self):
        """The returned plane is a uint8 view with rows at dim 0."""
        plane = torch.arange(2 * 3 * 4, dtype=torch.uint8).reshape(2, 3, 4)
        dedup = hybrid._deduplicate_block_planes([plane])
        assert dedup[0].dtype == torch.uint8
        assert dedup[0].shape == (2, 12)
        # Row 0 is the first 12 values of the original.
        assert torch.equal(dedup[0][0], plane.reshape(2, 12)[0])


@pytest.mark.parametrize("cols", [4, 8, 12])
def test_aligned_contract_boundary(cols):
    """The alignment predicate (offset, row stride, width all %4==0).

    A contiguous plane with cols%4==0 satisfies it on every torch build
    this suite runs on; cols is parameterized so the arithmetic stays
    honest rather than being pinned to a single lucky width.
    """
    plane = _aligned_plane(4, cols)
    before = plane.clone()
    hybrid.copy_block_planes([plane], [[1, 0]])
    assert torch.equal(plane[0], before[1])
