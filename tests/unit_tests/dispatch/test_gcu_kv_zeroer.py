# Copyright (c) 2026 BAAI. All rights reserved.

import ctypes
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_test_platform = os.environ.get("FL_TEST_PLATFORM")
if _test_platform and _test_platform != "enflame":
    pytest.skip("GCU-specific unit tests only run on Enflame", allow_module_level=True)

from vllm_fl.dispatch.backends.vendor.gcu.impl import kv_zeroer


class _CaptureKernel:
    def __init__(self):
        self.grid = None
        self.arguments = None
        self.keywords = None

    def __getitem__(self, grid):
        self.grid = grid

        def launch(*arguments, **keywords):
            self.arguments = arguments
            self.keywords = keywords

        return launch


@pytest.fixture
def outside_capture(monkeypatch):
    gcu = getattr(torch, "gcu", SimpleNamespace())
    monkeypatch.setattr(torch, "gcu", gcu, raising=False)
    monkeypatch.setattr(
        gcu, "is_current_stream_capturing", lambda: False, raising=False
    )


@pytest.mark.parametrize("program_limit", [65535, 4])
def test_launcher_bounds_grid_and_preserves_segment_metadata(
    monkeypatch, program_limit
):
    kernel = _CaptureKernel()
    monkeypatch.setattr(kv_zeroer, "_zero_kv_blocks_kernel_gcu", kernel)
    monkeypatch.setattr(kv_zeroer, "GCU_MAX_GRID_X", program_limit)
    copied = []
    index_tensor = object()

    def copy_ids(ids, **kwargs):
        copied.append((ids, kwargs))
        return index_tensor

    monkeypatch.setattr(kv_zeroer, "async_tensor_h2d", copy_ids)
    segments = (object(), object(), object())
    zeroer = SimpleNamespace(
        device="gcu", _meta=(*segments, 3, 4, 784), _gcu_device_addresses_ready=True
    )
    ids = list(range(100))
    kv_zeroer.zero_block_ids_gcu(zeroer, ids)

    assert copied == [(ids, {"device": "gcu", "dtype": torch.int64})]
    assert kernel.grid == (program_limit,)
    assert kernel.arguments == (*segments, index_tensor, 100 * 784 * 3)
    assert kernel.keywords == {
        "BLOCK_SIZE": 4,
        "N_SEGS": 784,
        "MAX_CHUNKS": 3,
    }


def test_empty_blocks_and_missing_metadata_do_not_copy_or_launch(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Empty zeroing work must not copy or launch")

    monkeypatch.setattr(kv_zeroer, "async_tensor_h2d", unexpected)
    monkeypatch.setattr(kv_zeroer, "_zero_kv_blocks_kernel_gcu", unexpected)
    kv_zeroer.zero_block_ids_gcu(SimpleNamespace(_meta=None), [0])
    kv_zeroer.zero_block_ids_gcu(SimpleNamespace(_meta=object()), [])


def test_raw_address_table_cannot_launch(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Raw torch addresses must not reach a kernel")

    monkeypatch.setattr(kv_zeroer, "async_tensor_h2d", unexpected)
    monkeypatch.setattr(kv_zeroer, "_zero_kv_blocks_kernel_gcu", unexpected)
    with pytest.raises(RuntimeError, match="no initialized device-address table"):
        kv_zeroer.zero_block_ids_gcu(SimpleNamespace(_meta=(object(),)), [0])


def test_pointer_query_uses_only_the_active_sdk(monkeypatch, tmp_path):
    library_path = tmp_path / "libtopsrt.so.1.9"
    library_path.touch()
    original_read_text = Path.read_text
    maps = f"0-1 r-xp 0 00:00 0 {library_path}\n"
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            maps
            if str(path) == "/proc/self/maps"
            else original_read_text(path, *args, **kwargs)
        ),
    )
    calls = []
    query = SimpleNamespace()
    library = SimpleNamespace(topsPointerGetAttribute=query)
    monkeypatch.setattr(
        ctypes, "CDLL", lambda path, **kwargs: calls.append((path, kwargs)) or library
    )
    kv_zeroer._get_tops_pointer_query.cache_clear()
    try:
        assert kv_zeroer._get_tops_pointer_query() is query
        assert kv_zeroer._get_tops_pointer_query() is query
        assert calls == [
            (str(library_path.resolve()), {"mode": os.RTLD_NOW | os.RTLD_NOLOAD})
        ]
        assert query.argtypes == [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        assert query.restype is ctypes.c_int
    finally:
        kv_zeroer._get_tops_pointer_query.cache_clear()


@pytest.mark.parametrize("count", [0, 2])
def test_pointer_query_rejects_missing_or_ambiguous_sdk(monkeypatch, count):
    maps = "\n".join(
        f"0-1 r-xp 0 00:00 0 /opt/tops{index}/libtopsrt.so.1.9"
        for index in range(count)
    )
    monkeypatch.setattr(Path, "read_text", lambda path: maps)

    def unexpected(*args, **kwargs):
        pytest.fail("SDK discovery must not load another runtime")

    monkeypatch.setattr(ctypes, "CDLL", unexpected)
    kv_zeroer._get_tops_pointer_query.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="active Enflame SDK runtime"):
            kv_zeroer._get_tops_pointer_query()
    finally:
        kv_zeroer._get_tops_pointer_query.cache_clear()


def test_device_address_translation_reads_metadata_once(monkeypatch):
    calls = []
    raw = torch.tensor([4096, 4112, 8192], dtype=torch.uint64)

    class AddressTable:
        device = raw.device
        reads = 0

        def cpu(self):
            self.reads += 1
            return raw

    addresses = AddressTable()

    def query(output, attribute, pointer):
        calls.append((attribute, pointer.value))
        output._obj.value = pointer.value + 65536
        return 0

    monkeypatch.setattr(kv_zeroer, "_get_tops_pointer_query", lambda: query)
    converted = kv_zeroer._device_segment_addresses(addresses)
    assert addresses.reads == 1
    assert calls == [(3, 4096), (3, 4112), (3, 8192)]
    assert converted.dtype == torch.uint64
    assert converted.device == raw.device
    assert converted.tolist() == [69632, 69648, 73728]
    assert raw.tolist() == [4096, 4112, 8192]


@pytest.mark.parametrize("return_code,device_address", [(7, 69632), (0, None), (0, 1)])
def test_constructor_does_not_publish_partial_address_table(
    monkeypatch, outside_capture, return_code, device_address
):
    raw = torch.tensor([4096, 4112], dtype=torch.uint64)
    meta = (raw, object(), object(), 1, 1024, 2)
    calls = []

    def query(output, attribute, pointer):
        calls.append(pointer.value)
        output._obj.value = 69632 if len(calls) == 1 else device_address
        return 0 if len(calls) == 1 else return_code

    monkeypatch.setattr(kv_zeroer, "_get_tops_pointer_query", lambda: query)

    class Zeroer:
        def _fl_gcu_original_init(self):
            self._meta = meta

    zeroer = Zeroer()
    with pytest.raises(RuntimeError, match="Enflame KV"):
        kv_zeroer.init_kv_zeroer_gcu(zeroer)
    assert zeroer._meta is meta
    assert calls == [4096, 4112]
    assert not zeroer._gcu_device_addresses_ready
    with pytest.raises(RuntimeError, match="no initialized device-address table"):
        kv_zeroer.zero_block_ids_gcu(zeroer, [0])


def test_constructor_preserves_arguments_and_nonaddress_metadata(
    monkeypatch, outside_capture
):
    raw = torch.tensor([4096], dtype=torch.uint64)
    meta = (raw, object(), object(), 1, 1024, 1)
    converted = object()
    calls = []
    groups = iter([object()])

    class Zeroer:
        def _fl_gcu_original_init(self, *args, **kwargs):
            calls.append((args, kwargs))
            self._meta = meta

    monkeypatch.setattr(
        kv_zeroer, "_device_segment_addresses", lambda addresses: converted
    )
    zeroer = Zeroer()
    kv_zeroer.init_kv_zeroer_gcu(zeroer, "gcu", groups, cache_dtype="auto")
    assert calls == [(("gcu", groups), {"cache_dtype": "auto"})]
    assert zeroer._meta[0] is converted
    assert all(
        actual is expected for actual, expected in zip(zeroer._meta[1:], meta[1:])
    )
    assert zeroer._gcu_device_addresses_ready

    def unexpected(*args, **kwargs):
        pytest.fail("Requests must not retranslate or read CPU metadata")

    monkeypatch.setattr(kv_zeroer, "_device_segment_addresses", unexpected)
    monkeypatch.setattr(kv_zeroer, "_get_tops_pointer_query", unexpected)
    monkeypatch.setattr(kv_zeroer, "_zero_kv_blocks_kernel_gcu", _CaptureKernel())
    monkeypatch.setattr(kv_zeroer, "async_tensor_h2d", lambda *args, **kwargs: object())
    zeroer.device = "gcu"
    kv_zeroer.zero_block_ids_gcu(zeroer, [0])
    kv_zeroer.zero_block_ids_gcu(zeroer, [1])


def test_constructor_without_attention_metadata_does_not_query(
    monkeypatch, outside_capture
):
    class Zeroer:
        def _fl_gcu_original_init(self):
            self._meta = None

    def unexpected(*args, **kwargs):
        pytest.fail("No attention metadata needs no SDK address query")

    monkeypatch.setattr(kv_zeroer, "_device_segment_addresses", unexpected)
    zeroer = Zeroer()
    kv_zeroer.init_kv_zeroer_gcu(zeroer)
    assert zeroer._meta is None
    assert zeroer._gcu_device_addresses_ready


def test_original_builder_preserves_virtual_segments_and_aliases(
    monkeypatch, outside_capture
):
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker import utils as worker_utils

    original_init = getattr(
        worker_utils.KVBlockZeroer,
        "_fl_gcu_original_init",
        worker_utils.KVBlockZeroer.__init__,
    )
    kv = torch.empty((2, 8, 16, 2, 16), dtype=torch.int32)
    backend_calls = []

    def block_dim(*args, **kwargs):
        backend_calls.append((args, kwargs))
        return 1

    group = SimpleNamespace(
        kv_cache_spec=FullAttentionSpec(
            block_size=32, num_kv_heads=2, head_size=16, dtype=torch.int32
        ),
        kv_cache_group_id=0,
        backend=SimpleNamespace(get_kv_cache_block_dim=block_dim),
        layer_names=["real", "alias", "runner", "non_tensor"],
    )
    context = {
        "real": SimpleNamespace(kv_cache=kv),
        "alias": SimpleNamespace(kv_cache=kv),
        "runner": SimpleNamespace(kv_cache=torch.empty(1)),
        "non_tensor": SimpleNamespace(kv_cache=[]),
    }
    snapshots = []

    class Zeroer(worker_utils.KVBlockZeroer):
        def _fl_gcu_original_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            snapshots.append(self._meta)

    queried = []

    def query(output, attribute, pointer):
        queried.append(pointer.value)
        output._obj.value = pointer.value + 256
        return 0

    monkeypatch.setattr(kv_zeroer, "_get_tops_pointer_query", lambda: query)
    zeroer = object.__new__(Zeroer)
    kv_zeroer.init_kv_zeroer_gcu(
        zeroer,
        torch.device("cpu"),
        (g for g in [group]),
        [16],
        "auto",
        context,
        {"runner"},
    )
    addresses = [kv.data_ptr() + offset for offset in (0, 2048, 16384, 18432)]
    assert backend_calls == [((16, 2, 16), {"cache_dtype_str": "auto"})]
    assert len(snapshots) == 1
    assert snapshots[0][0].tolist() == addresses
    assert queried == addresses
    assert zeroer._meta[0].tolist() == [address + 256 for address in addresses]
    assert zeroer._meta[0].dtype == torch.uint64
    assert zeroer._meta[0].device == kv.device
    assert zeroer._meta[1].tolist() == [1024] * 4
    assert zeroer._meta[2].tolist() == [512] * 4
    assert zeroer._meta[3:] == (1, 512, 4)
    assert all(
        actual is expected
        for actual, expected in zip(zeroer._meta[1:], snapshots[0][1:])
    )


def test_constructor_rejects_capture_before_building_or_reading_metadata(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Capture must be rejected before constructing/reading metadata")

    gcu = SimpleNamespace(is_current_stream_capturing=lambda: True)
    monkeypatch.setattr(torch, "gcu", gcu, raising=False)
    monkeypatch.setattr(kv_zeroer, "_device_segment_addresses", unexpected)

    class Zeroer:
        _fl_gcu_original_init = unexpected

    zeroer = Zeroer()
    with pytest.raises(RuntimeError, match="during capture"):
        kv_zeroer.init_kv_zeroer_gcu(zeroer)
    assert not zeroer._gcu_device_addresses_ready


def test_patch_is_gcu_only_and_warmup_uses_new_launcher(monkeypatch):
    import vllm.platforms
    from vllm.v1.worker import utils as worker_utils

    official_kernel = worker_utils._zero_kv_blocks_kernel
    original_method = worker_utils.KVBlockZeroer.zero_block_ids
    original_init = getattr(
        worker_utils.KVBlockZeroer,
        "_fl_gcu_original_init",
        worker_utils.KVBlockZeroer.__init__,
    )
    monkeypatch.setattr(worker_utils.KVBlockZeroer, "__init__", original_init)
    monkeypatch.delattr(
        worker_utils.KVBlockZeroer, "_fl_gcu_original_init", raising=False
    )
    monkeypatch.setattr(worker_utils.KVBlockZeroer, "zero_block_ids", original_method)
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="nvidia")
    )
    kv_zeroer.apply_kv_zeroer_gcu_patch()
    assert worker_utils.KVBlockZeroer.zero_block_ids is original_method
    assert worker_utils.KVBlockZeroer.__init__ is original_init
    assert not hasattr(worker_utils.KVBlockZeroer, "_fl_gcu_original_init")

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="enflame")
    )
    kv_zeroer.apply_kv_zeroer_gcu_patch()
    kv_zeroer.apply_kv_zeroer_gcu_patch()
    assert worker_utils.KVBlockZeroer.zero_block_ids is kv_zeroer.zero_block_ids_gcu
    assert worker_utils.KVBlockZeroer.__init__ is kv_zeroer.init_kv_zeroer_gcu
    assert worker_utils.KVBlockZeroer._fl_gcu_original_init is original_init
    assert worker_utils._zero_kv_blocks_kernel is official_kernel

    kernel = _CaptureKernel()
    monkeypatch.setattr(kv_zeroer, "_zero_kv_blocks_kernel_gcu", kernel)
    copied = []
    monkeypatch.setattr(
        kv_zeroer,
        "async_tensor_h2d",
        lambda ids, **kwargs: copied.append(ids) or object(),
    )
    zeroer = object.__new__(worker_utils.KVBlockZeroer)
    zeroer.device = "gcu"
    zeroer._meta = (object(), object(), object(), 3, 4, 784)
    zeroer._gcu_device_addresses_ready = True
    zeroer.warmup(4)
    assert copied == [[0]]
    assert kernel.grid == (784 * 3,)
    zeroer.warmup(0)
    assert copied == [[0]]


@pytest.mark.skipif(_test_platform != "enflame", reason="Requires an actual GCU")
def test_native_zeroing_784_segments_preserves_unselected_blocks_and_padding(
    monkeypatch,
):
    # Force multiple iterations per program, including skipped chunks of short
    # pages. This covers both the grid.y=784 failure and bounded-grid striding.
    monkeypatch.setattr(kv_zeroer, "GCU_MAX_GRID_X", 64)
    n_segs, n_blocks = 784, 4
    row_size = n_blocks * 4096
    original = torch.arange(1, n_segs * row_size + 1, dtype=torch.int32).reshape(
        n_segs, row_size
    )
    storage = original.to("gcu")
    strides = [4096 if segment % 2 == 0 else 3584 for segment in range(n_segs)]
    pages = [(1025, 2055, 3075)[segment % 3] for segment in range(n_segs)]
    addresses = [
        storage.data_ptr() + segment * row_size * 4 for segment in range(n_segs)
    ]
    device = storage.device
    meta = (
        torch.tensor(addresses, dtype=torch.uint64, device=device),
        torch.tensor(strides, dtype=torch.int64, device=device),
        torch.tensor(pages, dtype=torch.int64, device=device),
        4,
        1024,
        n_segs,
    )

    class Zeroer:
        def _fl_gcu_original_init(self):
            self.device = device
            self._meta = meta

    zeroer = Zeroer()
    kv_zeroer.init_kv_zeroer_gcu(zeroer)
    block_ids = [2, 0]
    kv_zeroer.zero_block_ids_gcu(zeroer, block_ids)

    def check(ids):
        expected = original.clone()
        for segment in range(n_segs):
            for block_id in ids:
                start = block_id * strides[segment]
                expected[segment, start : start + pages[segment]] = 0
        torch.testing.assert_close(storage.cpu(), expected, rtol=0, atol=0)

    check(block_ids)

    # Capture the already initialized metadata/kernel; read changed block IDs
    # on replay. Address queries and CPU metadata reads have finished above.
    def unexpected(*args, **kwargs):
        pytest.fail("Graph execution must not query or rebuild address metadata")

    monkeypatch.setattr(kv_zeroer, "_get_tops_pointer_query", unexpected)
    monkeypatch.setattr(kv_zeroer, "_device_segment_addresses", unexpected)
    indices = torch.tensor(block_ids, dtype=torch.int64, device=device)
    storage.copy_(original)
    torch.gcu.synchronize()
    graph = torch.gcu.GCUGraph()
    jobs = len(block_ids) * n_segs * 4
    with torch.gcu.graph(graph):
        kv_zeroer._zero_kv_blocks_kernel_gcu[(64,)](
            *zeroer._meta[:3],
            indices,
            jobs,
            BLOCK_SIZE=1024,
            N_SEGS=n_segs,
            MAX_CHUNKS=4,
        )
    storage.copy_(original)
    changed_ids = [1, 3]
    indices.copy_(torch.tensor(changed_ids, dtype=torch.int64, device=device))
    torch.gcu.synchronize()
    graph.replay()
    torch.gcu.synchronize()
    check(changed_ids)
