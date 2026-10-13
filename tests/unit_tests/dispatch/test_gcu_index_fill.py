# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU-only launch geometry and compatibility checks for the GCU adapter."""

import hashlib
import importlib.util
from itertools import product
from math import prod
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = (
    Path(__file__).parents[3]
    / "vllm_fl/dispatch/backends/vendor/gcu/impl/index_fill.py"
)
_SPEC = importlib.util.spec_from_file_location("gcu_index_fill_under_test", _PATH)
index_fill = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(index_fill)


class Tensor:
    def __init__(self, shape, device="gcu", strides=None, offset=0, writes=None):
        self.shape = tuple(shape)
        self.device = SimpleNamespace(type=device)
        self.strides = tuple(
            strides or [prod(shape[i + 1 :]) for i in range(len(shape))]
        )
        self.offset = offset
        self.writes = {} if writes is None else writes

    def numel(self):
        return prod(self.shape)

    def size(self, dim):
        return self.shape[dim]

    def stride(self):
        return self.strides

    def view(self, *shape):
        assert prod(shape) == self.numel()
        return Tensor(shape, self.device.type, offset=self.offset, writes=self.writes)

    def narrow(self, dim, start, length):
        shape = list(self.shape)
        assert 0 <= start < shape[dim] and 0 < length <= shape[dim] - start
        shape[dim] = length
        return Tensor(
            shape,
            self.device.type,
            self.strides,
            self.offset + start * self.strides[dim],
            self.writes,
        )


class Indices:
    def __init__(self, length, values=None, offset=0):
        self.length, self.values, self.offset = length, values, offset

    def numel(self):
        return self.length

    def narrow(self, dim, start, length):
        assert (
            dim == 0 and 0 <= start < self.length and 0 < length <= self.length - start
        )
        return Indices(length, self.values, self.offset + start)


class Kernel:
    def __init__(self, simulate=False):
        self.calls = []
        self.simulate = simulate

    def __getitem__(self, grid):
        def call(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            if not self.simulate:
                return
            out, indices, value, count, index_len, dim_size = args[:6]
            shapes, strides = args[6:14], args[14:22]
            for linear in range(count):
                offset = out.offset
                for dim in range(kwargs["RANK"]):
                    size = index_len if dim == kwargs["DIM"] else shapes[dim]
                    coord, linear = linear % size, linear // size
                    if dim == kwargs["DIM"]:
                        coord = indices.values[indices.offset + coord]
                        if coord < 0:
                            coord += dim_size
                    offset += coord * strides[dim]
                out.writes[offset] = value

        return call


def provider(simulate=False):
    delegated = []
    sentinel = object()

    def original(out, dim, index, value, value_is_tensor, fill_numel):
        delegated.append((out, dim, index, value, value_is_tensor, fill_numel))
        return sentinel

    module = SimpleNamespace(
        _index_fill_contiguous_launch=original,
        index_fill_kernel=Kernel(simulate),
        _BLOCK_SIZE=512,
        _MAX_RANK=8,
    )
    return module, delegated, sentinel


@pytest.mark.parametrize("inner,adapted", [(16320, False), (16321, True)])
def test_grid_y_boundary_preserves_safe_official_launcher(inner, adapted):
    module, delegated, sentinel = provider()
    launch = index_fill._make_contiguous_launch(
        module, module._index_fill_contiguous_launch
    )
    out, indices = Tensor((4, inner)), Indices(2)
    result = launch(out, 0, indices, 0, False, 2 * inner)
    assert bool(module.index_fill_kernel.calls) is adapted
    assert bool(delegated) is (not adapted)
    if not adapted:
        assert result is sentinel
        assert delegated == [(out, 0, indices, 0, False, 2 * inner)]


def test_other_devices_preserve_original_even_for_unsafe_gcu_grid():
    module, delegated, sentinel = provider()
    out, indices = Tensor((4, 151936), "cuda"), Indices(2)
    launch = index_fill._make_contiguous_launch(
        module, module._index_fill_contiguous_launch
    )
    assert launch(out, 0, indices, 0, False, 303872) is sentinel
    assert delegated and not module.index_fill_kernel.calls


def test_large_index_count_tiles_without_increasing_official_block():
    module, _, _ = provider()
    launch = index_fill._make_contiguous_launch(
        module, module._index_fill_contiguous_launch
    )
    launch(Tensor((256, 151936)), 0, Indices(256), 0, False, 256 * 151936)
    calls = module.index_fill_kernel.calls
    assert len(calls) == 2
    assert sum(args[3] for _, args, _ in calls) == 256 * 151936
    assert all(len(grid) == 1 and grid[0] <= 65535 for grid, _, _ in calls)
    assert all(kwargs["BLOCK_SIZE"] == 512 for _, _, kwargs in calls)
    assert [args[1].offset for _, args, _ in calls] == [0, 220]


def test_original_grid_x_overflow_is_also_tiled():
    module, delegated, _ = provider()
    launch = index_fill._make_contiguous_launch(
        module, module._index_fill_contiguous_launch
    )
    count = 65535 * 512 + 1
    launch(Tensor((4,)), 0, Indices(count), 0, False, count)
    assert not delegated
    assert [grid for grid, _, _ in module.index_fill_kernel.calls] == [(65535,), (1,)]


def test_high_rank_contiguous_fill_uses_three_metadata_dimensions():
    module, _, _ = provider()
    out = Tensor((1, 1, 1, 1, 1, 1, 4, 1, 151936))
    launch = index_fill._make_contiguous_launch(
        module, module._index_fill_contiguous_launch
    )
    value = object()
    launch(out, 6, Indices(2), value, True, 2 * 151936)
    _, args, kwargs = module.index_fill_kernel.calls[0]
    assert args[0].writes is out.writes
    assert args[2] is value
    assert args[6:14] == (1, 4, 151936, 1, 1, 1, 1, 1)
    assert args[14:22] == (4 * 151936, 151936, 1, 0, 0, 0, 0, 0)
    assert kwargs == {"DIM": 1, "RANK": 3, "VALUE_IS_TENSOR": True, "BLOCK_SIZE": 512}


@pytest.mark.parametrize(
    "shape,capacity",
    [
        ((3, 7, 5), 12),
        ((2, 31, 4), 8),
        ((5, 2, 9), 11),
        ((1, 151936, 256), 65535 * 512),
    ],
)
def test_tiles_cover_every_fill_coordinate_once(shape, capacity):
    outer, inner, index_len = shape
    tiles = list(index_fill._tiles(outer, inner, index_len, capacity))
    assert sum(o * i * n for _, o, _, i, _, n in tiles) == outer * inner * index_len
    assert all(0 < o * i * n <= capacity for _, o, _, i, _, n in tiles)
    if outer * inner * index_len < 1000:
        coords = []
        for os, oc, ins, inc, ns, nc in tiles:
            coords.extend(
                product(range(os, os + oc), range(ins, ins + inc), range(ns, ns + nc))
            )
        assert len(coords) == len(set(coords))
        assert set(coords) == set(product(range(outer), range(inner), range(index_len)))


def test_strided_tiles_preserve_values_with_duplicate_indices_across_tiles(monkeypatch):
    module, _, _ = provider(simulate=True)
    module._BLOCK_SIZE = 4
    monkeypatch.setattr(index_fill, "_MAX_GRID_X", 2)
    monkeypatch.setattr(index_fill, "_MAX_GRID_Y", 1)
    launch = index_fill._make_contiguous_launch(
        module, module._index_fill_contiguous_launch
    )
    out = Tensor((2, 3, 9))
    launch(out, 1, Indices(3, [0, -1, 0]), 37, False, 2 * 9 * 3)
    expected = {
        outer * 27 + dim * 9 + inner: 37
        for outer, dim, inner in product(range(2), (0, 2), range(9))
    }
    assert out.writes == expected
    assert all(grid[0] <= 2 for grid, _, _ in module.index_fill_kernel.calls)


def source_provider(tmp_path, monkeypatch):
    module, _, _ = provider()
    source = tmp_path / "official_index_fill.py"
    raw = b"test-provider-raw-bytes\n"
    source.write_bytes(raw)
    module.__file__ = str(source)
    monkeypatch.setattr(index_fill, "_SOURCE_SHA256", hashlib.sha256(raw).hexdigest())
    return module


def test_patch_reapplication_is_idempotent_and_rejects_method_replacement(
    tmp_path, monkeypatch
):
    module = source_provider(tmp_path, monkeypatch)
    original = module._index_fill_contiguous_launch
    assert index_fill._install_index_fill_gcu_patch(module) is True
    wrapper = module._index_fill_contiguous_launch
    assert wrapper.__wrapped__ is original
    assert index_fill._install_index_fill_gcu_patch(module) is False
    assert module._index_fill_contiguous_launch is wrapper
    module._index_fill_contiguous_launch = original
    with pytest.raises(RuntimeError, match="changed after patching"):
        index_fill._install_index_fill_gcu_patch(module)


def test_patch_rejects_generic_kernel_replacement(tmp_path, monkeypatch):
    module = source_provider(tmp_path, monkeypatch)
    index_fill._install_index_fill_gcu_patch(module)
    module.index_fill_kernel = Kernel()
    with pytest.raises(RuntimeError, match="changed after patching"):
        index_fill._install_index_fill_gcu_patch(module)


@pytest.mark.parametrize("change", ["source", "signature", "rank", "block"])
def test_unsupported_provider_fails_before_mutation(tmp_path, monkeypatch, change):
    module = source_provider(tmp_path, monkeypatch)
    if change == "source":
        Path(module.__file__).write_bytes(b"unexpected provider")
    elif change == "signature":
        module._index_fill_contiguous_launch = lambda unexpected: None
    elif change == "rank":
        module._MAX_RANK = 4
    else:
        module._BLOCK_SIZE = 1024
    original = module._index_fill_contiguous_launch
    with pytest.raises(RuntimeError, match="Unsupported FlagGems index_fill"):
        index_fill._install_index_fill_gcu_patch(module)
    assert module._index_fill_contiguous_launch is original
    assert not hasattr(module, index_fill._PATCH_STATE)
