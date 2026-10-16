# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU-only launch geometry and compatibility checks for the GCU adapter."""

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
    def __init__(
        self, shape, device="gcu", strides=None, offset=0, writes=None, dtype="int32"
    ):
        self.shape = tuple(shape)
        self.device = SimpleNamespace(type=device)
        self.dtype = dtype
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
        return Tensor(
            shape,
            self.device.type,
            offset=self.offset,
            writes=self.writes,
            dtype=self.dtype,
        )

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
            self.dtype,
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
        self.arg_names = index_fill._KERNEL_PARAMETERS
        self.jit_function = SimpleNamespace(
            arg_names=self.arg_names, constexprs=index_fill._KERNEL_CONSTEXPRS
        )

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
        torch=SimpleNamespace(int64="int64"),
    )
    return module, delegated, sentinel


@pytest.mark.parametrize("inner,adapted", [(16320, False), (16321, True)])
@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_grid_y_boundary_preserves_safe_official_launcher(inner, adapted, dtype):
    module, delegated, sentinel = provider()
    launch = index_fill._make_contiguous_launch(
        module, module._index_fill_contiguous_launch
    )
    out, indices = Tensor((4, inner), dtype=dtype), Indices(2)
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


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_strided_tiles_preserve_values_with_duplicate_indices_across_tiles(
    monkeypatch, dtype
):
    module, _, _ = provider(simulate=True)
    module._BLOCK_SIZE = 4
    monkeypatch.setattr(index_fill, "_MAX_GRID_X", 2)
    monkeypatch.setattr(index_fill, "_MAX_GRID_Y", 1)
    launch = index_fill._make_contiguous_launch(
        module, module._index_fill_contiguous_launch
    )
    out = Tensor((2, 3, 9), dtype=dtype)
    launch(out, 1, Indices(3, [0, -1, 0]), 37, False, 2 * 9 * 3)
    expected = {
        outer * 27 + dim * 9 + inner: 37
        for outer, dim, inner in product(range(2), (0, 2), range(9))
    }
    assert out.writes == expected
    assert all(grid[0] <= 2 for grid, _, _ in module.index_fill_kernel.calls)


@pytest.mark.parametrize("inner,block,grid", [(151936, 256, 1187), (151808, 512, 593)])
@pytest.mark.parametrize("tensor_value", [False, True])
def test_int64_tail_uses_full_blocks_without_converting_value(
    inner, block, grid, tensor_value
):
    module, delegated, _ = provider()
    launch = index_fill._make_contiguous_launch(
        module, module._index_fill_contiguous_launch
    )
    value = object() if tensor_value else -(2**40) - 3
    launch(
        Tensor((4, inner), dtype="int64"), 0, Indices(2), value, tensor_value, 2 * inner
    )
    assert not delegated
    assert len(module.index_fill_kernel.calls) == 1
    actual_grid, args, kwargs = module.index_fill_kernel.calls[0]
    assert actual_grid == (grid,)
    assert kwargs["BLOCK_SIZE"] == block
    assert grid * block == args[3] == 2 * inner
    assert args[2] is value
    assert kwargs["VALUE_IS_TENSOR"] is tensor_value
    assert args[0].dtype == "int64"


def _assert_full_block_partition(shape, max_grid, max_block_size=512):
    launches = list(index_fill._full_block_tiles(*shape, max_block_size, max_grid))
    boxes = []
    for block, (os, oc, ins, inc, ns, nc) in launches:
        assert 0 < block <= max_block_size and block & (block - 1) == 0
        count = oc * inc * nc
        assert count % block == 0 and 0 < count // block <= max_grid
        box = ((os, oc), (ins, inc), (ns, nc))
        assert all(
            0 <= start < end and 0 < size <= end - start
            for (start, size), end in zip(box, shape)
        )
        for previous in boxes:
            assert any(
                start + size <= other_start or other_start + other_size <= start
                for (start, size), (other_start, other_size) in zip(box, previous)
            )
        boxes.append(box)
    # Disjoint boxes inside the original domain with equal volume cover it exactly.
    assert sum(prod(size for _, size in box) for box in boxes) == prod(shape)
    return launches


@pytest.mark.parametrize(
    "shape,max_grid",
    [
        ((1, 151935, 2), 65535),
        ((1, 151937, 2), 65535),
        ((3, 131071, 5), 65535),
        ((65537, 1, 1), 65535),
        ((1, 1, 65537), 65535),
        ((2, 4, 32), 1),
        ((8, 9, 8), 2),
        ((32, 16, 32), 7),
        ((1, 151936, 256), 65535),
    ],
)
def test_int64_full_block_partition_covers_large_odd_and_mixed_axes(shape, max_grid):
    _assert_full_block_partition(shape, max_grid)


def test_int64_full_block_partition_exhaustive_small_geometry():
    for shape in product(range(1, 10), repeat=3):
        for max_grid in (3, 7):
            launches = _assert_full_block_partition(shape, max_grid, max_block_size=8)
            coordinates = set()
            for _, (os, oc, ins, inc, ns, nc) in launches:
                coordinates.update(
                    product(
                        range(os, os + oc), range(ins, ins + inc), range(ns, ns + nc)
                    )
                )
            assert coordinates == set(product(*(range(size) for size in shape)))


def source_provider(tmp_path, monkeypatch):
    module, _, _ = provider()
    source = tmp_path / "official_index_fill.py"
    raw = b"# Test provider; source formatting is not an API contract.\n"
    source.write_bytes(raw)
    module.__file__ = str(source)
    monkeypatch.setattr(index_fill, "version", lambda name: "5.4.0")
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


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "signature",
        "keyword_only",
        "missing_launcher",
        "rank",
        "block",
        "kernel_arguments",
        "jit_arguments",
        "constexprs",
        "missing_kernel",
        "nonlaunchable_kernel",
    ],
)
def test_unsupported_provider_fails_before_mutation(tmp_path, monkeypatch, change):
    module = source_provider(tmp_path, monkeypatch)
    if change == "version":
        monkeypatch.setattr(index_fill, "version", lambda name: "5.5.0")
    elif change == "signature":
        module._index_fill_contiguous_launch = lambda unexpected: None
    elif change == "keyword_only":

        def incompatible(*, out, dim, index, value, value_is_tensor, fill_numel):
            pass

        module._index_fill_contiguous_launch = incompatible
    elif change == "missing_launcher":
        del module._index_fill_contiguous_launch
    elif change == "rank":
        module._MAX_RANK = 4
    elif change == "block":
        module._BLOCK_SIZE = 1024
    elif change == "kernel_arguments":
        module.index_fill_kernel.arg_names = ("unexpected",)
    elif change == "jit_arguments":
        module.index_fill_kernel.jit_function.arg_names = ("unexpected",)
    elif change == "constexprs":
        module.index_fill_kernel.jit_function.constexprs = (23, 24, 25)
    elif change == "missing_kernel":
        del module.index_fill_kernel
    else:
        module.index_fill_kernel = SimpleNamespace(
            arg_names=index_fill._KERNEL_PARAMETERS,
            jit_function=module.index_fill_kernel.jit_function,
        )
    original = getattr(module, "_index_fill_contiguous_launch", None)
    with pytest.raises(RuntimeError, match="Unsupported FlagGems"):
        index_fill._install_index_fill_gcu_patch(module)
    assert getattr(module, "_index_fill_contiguous_launch", None) is original
    assert not hasattr(module, index_fill._PATCH_STATE)


@pytest.mark.parametrize("source_change", ["comment", "formatting", "missing_source"])
def test_source_layout_does_not_change_supported_runtime_api(
    tmp_path, monkeypatch, source_change
):
    module = source_provider(tmp_path, monkeypatch)
    source = Path(module.__file__)
    if source_change == "comment":
        source.write_bytes(source.read_bytes() + b"# Downstream comment.\n")
    elif source_change == "formatting":
        source.write_bytes(source.read_bytes().replace(b"\n", b"\r\n\r\n"))
    else:
        source.unlink()
    assert index_fill._install_index_fill_gcu_patch(module) is True


def test_missing_distribution_version_fails_before_mutation(tmp_path, monkeypatch):
    module = source_provider(tmp_path, monkeypatch)

    def missing(name):
        raise index_fill.PackageNotFoundError(name)

    monkeypatch.setattr(index_fill, "version", missing)
    original = module._index_fill_contiguous_launch
    with pytest.raises(RuntimeError, match="version is unavailable"):
        index_fill._install_index_fill_gcu_patch(module)
    assert module._index_fill_contiguous_launch is original
    assert not hasattr(module, index_fill._PATCH_STATE)


def release_verifier(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "enflame_index_fill_verifier_under_test",
        Path(__file__).parents[3] / "docker/enflame/verify_index_fill.py",
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    package = tmp_path / "flag_gems/__init__.py"
    package.parent.mkdir()
    package.write_text("# Must not import the provider in a setup check.\n")
    source = tmp_path / verifier.SOURCE_PATH
    source.parent.mkdir()
    raw = b"# A test release fixture.\n"
    source.write_bytes(raw)
    import hashlib

    monkeypatch.setattr(verifier, "EXPECTED_BYTES", len(raw))
    monkeypatch.setattr(verifier, "EXPECTED_SHA256", hashlib.sha256(raw).hexdigest())
    distribution = SimpleNamespace(
        version="5.4.0", locate_file=lambda path: tmp_path / path
    )
    monkeypatch.setattr(
        verifier.importlib.metadata, "distribution", lambda name: distribution
    )
    monkeypatch.setattr(
        verifier.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(package)),
    )
    return verifier, distribution, source


def test_release_byte_guard_succeeds_for_exact_setup_provider(tmp_path, monkeypatch):
    verifier, _, source = release_verifier(tmp_path, monkeypatch)
    result = verifier.verify_index_fill()
    assert result["passed"] is True
    assert result["bytes"] == source.stat().st_size
    assert result["sha256"] == verifier.EXPECTED_SHA256
    assert result["stable_during_read"] is True


@pytest.mark.parametrize(
    "change", ["comment", "crlf", "same_size", "version", "origin"]
)
def test_release_byte_guard_remains_strict_in_setup(tmp_path, monkeypatch, change):
    verifier, distribution, source = release_verifier(tmp_path, monkeypatch)
    if change == "source":
        source.write_bytes(
            source.read_bytes() + b"# Nonsemantic, but not the release.\n"
        )
    elif change == "version":
        distribution.version = "5.5.0"
    else:
        shadow = tmp_path / "shadow.py"
        shadow.write_text("# Shadowed provider.\n")
        monkeypatch.setattr(
            verifier.importlib.util,
            "find_spec",
            lambda name: SimpleNamespace(origin=str(shadow)),
        )
    with pytest.raises(RuntimeError):
        verifier.verify_index_fill()
