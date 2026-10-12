# Copyright (c) 2026 BAAI. All rights reserved.

"""Translate Mamba indirect copy addresses before their CPU tables upload."""

from __future__ import annotations

import ctypes
import operator
from dataclasses import dataclass
from functools import wraps

import numpy as np
import torch

from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

from . import kv_zeroer

_MAX_POINTER = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
_REGISTRY_ATTRIBUTE = "_fl_gcu_mamba_addresses"


@dataclass(frozen=True)
class _StorageAddress:
    storage: object
    host_base: int
    device_base: int
    nbytes: int

    def check_live(self) -> None:
        if (
            self.storage.data_ptr() != self.host_base
            or self.storage.nbytes() != self.nbytes
        ):
            raise RuntimeError("Enflame Mamba storage changed after initialization")


class MambaAddressRegistry:
    """Own the storage lifetimes and startup SDK address observations."""

    def __init__(self, entries: tuple[_StorageAddress, ...]):
        self.entries = entries

    @classmethod
    def from_tensors(cls, tensors) -> MambaAddressRegistry:
        if torch.gcu.is_current_stream_capturing():
            raise RuntimeError(
                "Cannot initialize Enflame Mamba addresses during capture"
            )
        storages = {}
        device_indices = set()
        for tensor in tensors:
            if tensor.device.type != "gcu":
                raise RuntimeError("Enflame Mamba storage must be on GCU")
            storage = tensor.untyped_storage()
            device_indices.add(tensor.device.index)
            base, nbytes = int(storage.data_ptr()), int(storage.nbytes())
            if base <= 0 or nbytes <= 0 or base + nbytes - 1 > _MAX_POINTER:
                raise RuntimeError("Invalid Enflame Mamba storage address range")
            key = (tensor.device.index, base, nbytes)
            storages.setdefault(key, storage)
        if len(device_indices) > 1:
            raise RuntimeError("Enflame Mamba address registry cannot mix GCU devices")
        # Query only here. The hot path retains no SDK function or GPU tensor
        # address table to query/read back.
        query = kv_zeroer._get_tops_pointer_query() if storages else None
        entries = []
        for (_, base, nbytes), storage in storages.items():
            address = ctypes.c_void_p()
            rc = query(ctypes.byref(address), 3, ctypes.c_void_p(base))
            if rc != 0 or not address.value:
                raise RuntimeError(
                    f"Enflame Mamba device-address query failed: SDK return code {rc}"
                )
            if address.value + nbytes - 1 > _MAX_POINTER:
                raise RuntimeError("Enflame Mamba device address range overflows")
            entries.append(_StorageAddress(storage, base, address.value, nbytes))
        return cls(tuple(entries))

    def translate(self, address: int, size: int) -> int:
        if address <= 0 or address > _MAX_POINTER or size < 0:
            raise RuntimeError("Invalid Enflame Mamba copy address or size")
        matches = set()
        for entry in self.entries:
            # Recognize our already translated table too: callers can submit
            # the same copy twice without rebuilding its pinned CPU metadata.
            for base in (entry.host_base, entry.device_base):
                offset = address - base
                if 0 <= offset < entry.nbytes and size <= entry.nbytes - offset:
                    entry.check_live()
                    matches.add(entry.device_base + offset)
        if len(matches) != 1:
            raise RuntimeError(
                "Unknown, ambiguous or out-of-range Enflame Mamba copy storage"
            )
        return matches.pop()


def translate_mamba_copy_buffers(copy_bufs) -> None:
    """Validate every active row, then change CPU addresses before any upload."""
    if isinstance(copy_bufs.offset, bool):
        raise RuntimeError("Invalid Enflame Mamba copy count")
    n = operator.index(copy_bufs.offset)
    if n == 0:
        return
    if n < 0:
        raise RuntimeError("Invalid Enflame Mamba copy count")
    registry = getattr(copy_bufs, _REGISTRY_ATTRIBUTE, None)
    if not isinstance(registry, MambaAddressRegistry):
        raise RuntimeError("Enflame Mamba copies have no initialized address registry")
    src, dst, sizes = copy_bufs.src_ptrs.np, copy_bufs.dst_ptrs.np, copy_bufs.sizes.np
    if (
        any(array.ndim != 1 or len(array) < n for array in (src, dst, sizes))
        or src.dtype != np.dtype("uint64")
        or dst.dtype != np.dtype("uint64")
        or sizes.dtype != np.dtype("int32")
        or not src.flags.writeable
        or not dst.flags.writeable
    ):
        raise RuntimeError("Invalid Enflame Mamba CPU copy metadata buffers")
    translated_src, translated_dst = [], []
    for i in range(n):
        size = int(sizes[i])
        translated_src.append(registry.translate(int(src[i]), size))
        translated_dst.append(registry.translate(int(dst[i]), size))
    # Do not restore raw values after the original do_mamba_copy_block: its
    # nonblocking H2D transfers may still be reading these pinned buffers.
    src[:n] = translated_src
    dst[:n] = translated_dst


def _mamba_tensors(kv_cache_config, kv_caches):
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        for name in group.layer_names:
            layer_spec = (
                spec.kv_cache_specs[name]
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else spec
            )
            if isinstance(layer_spec, MambaSpec):
                # The real 0.28 runner returns an int8 page tensor here; the
                # bound layer's conv/SSM views share this backing storage.
                value = kv_caches[name]
                yield from value if isinstance(value, (list, tuple)) else (value,)


def _drop_registry(runner) -> None:
    setattr(runner, _REGISTRY_ATTRIBUTE, None)
    buffers = getattr(runner, "_mamba_bufs", None)
    if buffers is not None:
        setattr(buffers.preprocess, _REGISTRY_ATTRIBUTE, None)


def _install_mamba_hooks(runner_type, mamba_utils) -> None:
    if not getattr(runner_type, "_fl_gcu_mamba_patched", False):
        initialize = runner_type.initialize_kv_cache_tensors
        get_buffers = runner_type._get_mamba_bufs
        cleanup = runner_type._cleanup_profiling_kv_cache

        @wraps(initialize)
        def initialize_gcu(self, kv_cache_config, *args, **kwargs):
            if torch.gcu.is_current_stream_capturing():
                raise RuntimeError(
                    "Cannot initialize Enflame Mamba addresses during capture"
                )
            _drop_registry(self)
            caches = initialize(self, kv_cache_config, *args, **kwargs)
            registry = MambaAddressRegistry.from_tensors(
                _mamba_tensors(kv_cache_config, caches)
            )
            setattr(self, _REGISTRY_ATTRIBUTE, registry)
            return caches

        @wraps(get_buffers)
        def get_buffers_gcu(self, *args, **kwargs):
            registry = getattr(self, _REGISTRY_ATTRIBUTE, None)
            if not isinstance(registry, MambaAddressRegistry):
                raise RuntimeError("Enflame Mamba KV storage was not initialized")
            buffers = get_buffers(self, *args, **kwargs)
            setattr(buffers.preprocess, _REGISTRY_ATTRIBUTE, registry)
            return buffers

        @wraps(cleanup)
        def cleanup_gcu(self, *args, **kwargs):
            # Release the extra storage owners before the original cleanup
            # synchronizes/frees profiling KV caches and measures free memory.
            _drop_registry(self)
            return cleanup(self, *args, **kwargs)

        runner_type.initialize_kv_cache_tensors = initialize_gcu
        runner_type._get_mamba_bufs = get_buffers_gcu
        runner_type._cleanup_profiling_kv_cache = cleanup_gcu
        runner_type._fl_gcu_mamba_patched = True
    if not getattr(
        mamba_utils.do_mamba_copy_block, "_fl_gcu_translates_addresses", False
    ):
        do_copy = mamba_utils.do_mamba_copy_block

        @wraps(do_copy)
        def do_copy_gcu(copy_bufs):
            translate_mamba_copy_buffers(copy_bufs)
            return do_copy(copy_bufs)

        do_copy_gcu._fl_gcu_translates_addresses = True
        mamba_utils.do_mamba_copy_block = do_copy_gcu


def apply_mamba_copy_gcu_patch() -> None:
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "enflame":
        return
    from vllm.v1.worker import mamba_utils

    from vllm_fl.worker.model_runner import ModelRunnerFL

    _install_mamba_hooks(ModelRunnerFL, mamba_utils)
