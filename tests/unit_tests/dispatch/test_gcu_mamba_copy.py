# Copyright (c) 2026 BAAI. All rights reserved.

import ctypes
import gc
import os
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_test_platform = os.environ.get("FL_TEST_PLATFORM")
if _test_platform and _test_platform != "enflame":
    pytest.skip("GCU-specific unit tests only run on Enflame", allow_module_level=True)

from vllm.v1.worker import mamba_utils

from vllm_fl.dispatch.backends.vendor.gcu.impl import mamba_copy


class _Storage:
    def __init__(self, base, size=4096):
        self.base, self.size = base, size

    def data_ptr(self):
        return self.base

    def nbytes(self):
        return self.size


class _Tensor:
    def __init__(self, storage, device=0):
        self.storage = storage
        self.device = SimpleNamespace(type="gcu", index=device)

    def untyped_storage(self):
        return self.storage

    def cpu(self):
        pytest.fail("Address initialization must not read GPU tensor data")


class _Buffer:
    def __init__(self, values, dtype, uploads):
        self.np = np.array(values, dtype=dtype)
        self.uploads = uploads

    def copy_to_gpu(self, count):
        result = self.np[:count].copy()
        self.uploads.append(result)
        return result


@pytest.fixture
def sdk(monkeypatch):
    gcu = getattr(torch, "gcu", SimpleNamespace())
    monkeypatch.setattr(torch, "gcu", gcu, raising=False)
    monkeypatch.setattr(
        gcu, "is_current_stream_capturing", lambda: False, raising=False
    )
    calls = []

    def query(output, attribute, pointer):
        assert attribute == 3
        calls.append(pointer.value)
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = (
            pointer.value + 0x100000
        )
        return 0

    monkeypatch.setattr(mamba_copy.kv_zeroer, "_get_tops_pointer_query", lambda: query)
    return calls


def _copy_buffers(registry, sources, destinations, sizes, count=None):
    uploads = []
    buffers = SimpleNamespace(
        src_ptrs=_Buffer(sources, np.uint64, uploads),
        dst_ptrs=_Buffer(destinations, np.uint64, uploads),
        sizes=_Buffer(sizes, np.int32, uploads),
        offset=len(sizes) if count is None else count,
    )
    setattr(buffers, mamba_copy._REGISTRY_ATTRIBUTE, registry)
    return buffers, uploads


def test_startup_retains_alias_storage_and_queries_each_allocation_once(sdk):
    storage = _Storage(0x1000)
    reference = weakref.ref(storage)
    tensors = [_Tensor(storage), _Tensor(storage)]
    registry = mamba_copy.MambaAddressRegistry.from_tensors(tensors)
    assert sdk == [0x1000]
    assert len(registry.entries) == 1
    del tensors, storage
    gc.collect()
    assert reference() is registry.entries[0].storage
    del registry
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize(
    "offset,size", [(0, 4096), (2, 1023), (1023, 1024), (3071, 1025), (4095, 1)]
)
def test_complete_storage_and_byte_tail_offsets_use_cached_device_base(
    sdk, monkeypatch, offset, size
):
    registry = mamba_copy.MambaAddressRegistry.from_tensors([_Tensor(_Storage(0x1000))])
    monkeypatch.setattr(
        mamba_copy.kv_zeroer,
        "_get_tops_pointer_query",
        lambda: pytest.fail("SDK query in hot path"),
    )
    assert registry.translate(0x1000 + offset, size) == 0x101000 + offset
    assert registry.translate(0x101000 + offset, size) == 0x101000 + offset
    assert sdk == [0x1000]


@pytest.mark.parametrize(
    "address,size", [(0, 1), (0xFFF, 1), (0x2000, 1), (0x1FFF, 2), (0x1000, -1)]
)
def test_invalid_or_out_of_bounds_addresses_fail_before_upload(sdk, address, size):
    registry = mamba_copy.MambaAddressRegistry.from_tensors([_Tensor(_Storage(0x1000))])
    with pytest.raises(RuntimeError):
        registry.translate(address, size)


def test_storage_resize_or_pointer_change_invalidates_cached_binding(sdk):
    storage = _Storage(0x1000)
    registry = mamba_copy.MambaAddressRegistry.from_tensors([_Tensor(storage)])
    storage.size -= 1
    with pytest.raises(RuntimeError, match="storage changed"):
        registry.translate(0x1000, 2)


def test_capture_and_mixed_devices_are_rejected_before_query(sdk, monkeypatch):
    monkeypatch.setattr(torch.gcu, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="during capture"):
        mamba_copy.MambaAddressRegistry.from_tensors([_Tensor(_Storage(0x1000))])
    monkeypatch.setattr(torch.gcu, "is_current_stream_capturing", lambda: False)
    with pytest.raises(RuntimeError, match="mix GCU devices"):
        mamba_copy.MambaAddressRegistry.from_tensors(
            [_Tensor(_Storage(0x1000), 0), _Tensor(_Storage(0x4000), 1)]
        )
    assert sdk == []


@pytest.mark.parametrize(
    "return_code,device_base", [(1, 0x8000), (0, 0), (0, mamba_copy._MAX_POINTER)]
)
def test_sdk_failure_null_and_device_span_overflow_do_not_publish_registry(
    sdk, monkeypatch, return_code, device_base
):
    def query(output, attribute, pointer):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = device_base
        return return_code

    monkeypatch.setattr(mamba_copy.kv_zeroer, "_get_tops_pointer_query", lambda: query)
    with pytest.raises(RuntimeError):
        mamba_copy.MambaAddressRegistry.from_tensors([_Tensor(_Storage(0x1000))])


def test_all_active_rows_are_converted_before_real_original_uploads(sdk, monkeypatch):
    registry = mamba_copy.MambaAddressRegistry.from_tensors(
        [_Tensor(_Storage(0x1000)), _Tensor(_Storage(0x4000))]
    )
    buffers, uploads = _copy_buffers(
        registry,
        [0x1002, 0x13FF, 0x1BFF, 0],
        [0x4002, 0x43FF, 0x4BFF, 0],
        [1023, 1024, 1025, -1],
        count=3,
    )
    original = mamba_utils.do_mamba_copy_block
    original = getattr(original, "__wrapped__", original)
    module = SimpleNamespace(do_mamba_copy_block=original)
    runner_type = _runner_type({})
    mamba_copy._install_mamba_hooks(runner_type, module)
    original_kernel = mamba_utils.batch_memcpy_kernel
    copied = []
    monkeypatch.setattr(mamba_utils, "batch_memcpy", lambda *args: copied.append(args))
    monkeypatch.setattr(
        mamba_copy.kv_zeroer,
        "_get_tops_pointer_query",
        lambda: pytest.fail("SDK query in copy/upload path"),
    )
    module.do_mamba_copy_block(buffers)
    assert len(uploads) == 3 and len(copied) == 1
    np.testing.assert_array_equal(uploads[0], [0x101002, 0x1013FF, 0x101BFF])
    np.testing.assert_array_equal(uploads[1], [0x104002, 0x1043FF, 0x104BFF])
    np.testing.assert_array_equal(uploads[2], [1023, 1024, 1025])
    assert buffers.src_ptrs.np[3] == 0  # Unused capacity is untouched.
    assert mamba_utils.batch_memcpy_kernel is original_kernel
    module.do_mamba_copy_block(buffers)  # Already translated metadata remains valid.
    assert len(uploads) == 6


def test_failure_on_last_row_leaves_all_pinned_tables_unchanged(sdk):
    registry = mamba_copy.MambaAddressRegistry.from_tensors(
        [_Tensor(_Storage(0x1000)), _Tensor(_Storage(0x4000))]
    )
    buffers, uploads = _copy_buffers(
        registry, [0x1002, 0x9000], [0x4002, 0x4004], [2, 2]
    )
    before = [
        array.np.copy() for array in (buffers.src_ptrs, buffers.dst_ptrs, buffers.sizes)
    ]
    with pytest.raises(RuntimeError, match="Unknown"):
        mamba_copy.translate_mamba_copy_buffers(buffers)
    for expected, array in zip(
        before, (buffers.src_ptrs, buffers.dst_ptrs, buffers.sizes)
    ):
        np.testing.assert_array_equal(expected, array.np)
    assert uploads == []


def test_empty_work_and_missing_registry_do_not_upload():
    mamba_copy.translate_mamba_copy_buffers(SimpleNamespace(offset=0))
    with pytest.raises(RuntimeError, match="no initialized"):
        mamba_copy.translate_mamba_copy_buffers(SimpleNamespace(offset=1))


def _runner_type(caches):
    class Runner:
        def initialize_kv_cache_tensors(self, config, blocks):
            self.calls.append(("bind", config, blocks))
            return caches

        def _get_mamba_bufs(self):
            return self._mamba_bufs

        def _cleanup_profiling_kv_cache(self):
            assert getattr(self, mamba_copy._REGISTRY_ATTRIBUTE) is None
            assert (
                getattr(self._mamba_bufs.preprocess, mamba_copy._REGISTRY_ATTRIBUTE)
                is None
            )
            self.calls.append(("cleanup",))

    return Runner


def test_binding_precedes_sdk_and_cleanup_releases_storage_registry(sdk, monkeypatch):
    caches = {"mamba": _Tensor(_Storage(0x1000))}
    runner_type = _runner_type(caches)
    module = SimpleNamespace(do_mamba_copy_block=lambda buffers: None)
    mamba_copy._install_mamba_hooks(runner_type, module)
    initialized = runner_type.initialize_kv_cache_tensors
    mamba_copy._install_mamba_hooks(runner_type, module)
    assert runner_type.initialize_kv_cache_tensors is initialized
    monkeypatch.setattr(
        mamba_copy, "_mamba_tensors", lambda config, values: [values["mamba"]]
    )
    runner = runner_type()
    runner.calls = []
    runner._mamba_bufs = SimpleNamespace(preprocess=SimpleNamespace())
    assert runner.initialize_kv_cache_tensors("config", [16]) is caches
    assert runner.calls == [("bind", "config", [16])]
    assert sdk == [0x1000]
    monkeypatch.setattr(
        mamba_copy.kv_zeroer,
        "_get_tops_pointer_query",
        lambda: pytest.fail("SDK query in lazy buffer path"),
    )
    buffers = runner._get_mamba_bufs()
    assert getattr(buffers.preprocess, mamba_copy._REGISTRY_ATTRIBUTE) is getattr(
        runner, mamba_copy._REGISTRY_ATTRIBUTE
    )
    runner._cleanup_profiling_kv_cache()
    assert runner.calls[-1] == ("cleanup",)


def test_non_gcu_platform_does_not_install_hooks(monkeypatch):
    import vllm.platforms

    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="nvidia")
    )
    monkeypatch.setattr(
        mamba_copy, "_install_mamba_hooks", lambda *args: pytest.fail("Non-GCU patch")
    )
    mamba_copy.apply_mamba_copy_gcu_patch()


def test_real_spec_types_select_only_mamba_backing_pages():
    # Only the type and per-layer map are consumed by this selector. Keep real
    # upstream types without inventing a complete model/cache configuration.
    spec = object.__new__(mamba_copy.MambaSpec)
    uniform = object.__new__(mamba_copy.UniformTypeKVCacheSpecs)
    object.__setattr__(
        uniform, "kv_cache_specs", {"mixed_mamba": spec, "attention": object()}
    )
    first, second, unused = object(), object(), object()
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=spec, layer_names=["mamba"]),
            SimpleNamespace(
                kv_cache_spec=uniform, layer_names=["mixed_mamba", "attention"]
            ),
        ]
    )
    assert list(
        mamba_copy._mamba_tensors(
            config, {"mamba": first, "mixed_mamba": [second], "attention": unused}
        )
    ) == [first, second]


def test_capture_hook_rejects_before_original_cache_binding(sdk, monkeypatch):
    runner_type = _runner_type({})
    mamba_copy._install_mamba_hooks(
        runner_type, SimpleNamespace(do_mamba_copy_block=lambda buffers: None)
    )
    runner = runner_type()
    runner.calls = []
    runner._mamba_bufs = None
    monkeypatch.setattr(torch.gcu, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="during capture"):
        runner.initialize_kv_cache_tensors("config", [16])
    assert runner.calls == [] and sdk == []
