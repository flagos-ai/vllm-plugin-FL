# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from vLLM v0.28.0, vllm/v1/worker/utils.py (2cf0a6915ce5).

"""Zero KV segments using device addresses and bounded Enflame grids."""

import ctypes
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d

logger = logging.getLogger(__name__)

GCU_MAX_GRID_X = 65535


@lru_cache(maxsize=1)
def _get_tops_pointer_query():
    # Use the SDK already initialized by torch.gcu. Loading another runtime can
    # query a different context, so fail if the active SDK is absent/ambiguous.
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = fields[5]
        if path.startswith("/") and re.fullmatch(
            r"libtopsrt\.so(?:\.[0-9]+)*", Path(path).name
        ):
            paths.add(str(Path(path).resolve()))
    if len(paths) != 1 or not hasattr(os, "RTLD_NOLOAD"):
        raise RuntimeError("Cannot identify the active Enflame SDK runtime")
    library = ctypes.CDLL(paths.pop(), mode=os.RTLD_NOW | os.RTLD_NOLOAD)
    query = library.topsPointerGetAttribute
    # SDK signature: topsError_t (void*, topsPointer_attribute, topsDeviceptr_t).
    query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    query.restype = ctypes.c_int
    return query


def _device_segment_addresses(raw_addresses: torch.Tensor) -> torch.Tensor:
    query = _get_tops_pointer_query()
    translated = []
    # The official constructor stores torch handles, which differ from the
    # device VAs used by indirect Triton loads on GCU. Synchronize only here,
    # before worker warmup/capture; requests retain the original GPU metadata.
    for segment, address in enumerate(raw_addresses.cpu().tolist()):
        if not 0 < address < 1 << (ctypes.sizeof(ctypes.c_void_p) * 8):
            raise RuntimeError(f"Invalid Enflame KV address for segment {segment}")
        device_address = ctypes.c_void_p()
        # TOPS_POINTER_ATTRIBUTE_DEVICE_POINTER = 3; topsSuccess = 0.
        rc = query(ctypes.byref(device_address), 3, ctypes.c_void_p(address))
        if rc != 0 or not device_address.value:
            raise RuntimeError(
                f"Enflame KV device-address query failed for segment {segment}: "
                f"SDK return code {rc}"
            )
        if device_address.value % 4:
            raise RuntimeError(
                f"Unaligned Enflame KV device address: segment {segment}"
            )
        translated.append(device_address.value)
    return torch.tensor(translated, dtype=torch.uint64, device=raw_addresses.device)


def init_kv_zeroer_gcu(self, *args, **kwargs) -> None:
    self._gcu_device_addresses_ready = False
    if torch.gcu.is_current_stream_capturing():
        raise RuntimeError(
            "Cannot initialize Enflame KV device addresses during capture"
        )
    self._fl_gcu_original_init(*args, **kwargs)
    if self._meta is not None:
        device_addresses = _device_segment_addresses(self._meta[0])
        # Publish the translated table only after every SDK query succeeds.
        self._meta = (device_addresses, *self._meta[1:])
    self._gcu_device_addresses_ready = True


@triton.jit(do_not_specialize=["n_jobs"])
def _zero_kv_blocks_kernel_gcu(
    seg_addrs_ptr,
    seg_block_strides_ptr,
    seg_page_sizes_ptr,
    block_ids_ptr,
    n_jobs,
    BLOCK_SIZE: tl.constexpr,
    N_SEGS: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,
):
    # Flatten the upstream (block, segment, chunk) grid. A bounded number of
    # programs strides through all jobs, keeping grid.y and grid.z equal to 1.
    job_index = tl.program_id(0).to(tl.int64)
    while job_index < n_jobs:
        block_index = job_index // (N_SEGS * MAX_CHUNKS)
        seg_index = (job_index // MAX_CHUNKS) % N_SEGS
        chunk_index = job_index % MAX_CHUNKS
        block_stride_el = tl.load(seg_block_strides_ptr + seg_index)
        page_size_el = tl.load(seg_page_sizes_ptr + seg_index)
        chunk_offset = chunk_index * BLOCK_SIZE
        if chunk_offset < page_size_el:
            block_id = tl.load(block_ids_ptr + block_index)
            seg_addr = tl.load(seg_addrs_ptr + seg_index)
            ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
            block_offset = block_id.to(tl.int64) * block_stride_el.to(tl.int64)
            cols = chunk_offset + tl.arange(0, BLOCK_SIZE).to(tl.int64)
            tl.store(
                ptr + block_offset + cols,
                tl.zeros([BLOCK_SIZE], dtype=tl.int32),
                mask=cols < page_size_el,
            )
        job_index += tl.num_programs(0).to(tl.int64)


def zero_block_ids_gcu(self, block_ids: list[int]) -> None:
    """Use vLLM's original segment metadata and asynchronous block-ID copy."""
    if not block_ids or self._meta is None:
        return
    if not getattr(self, "_gcu_device_addresses_ready", False):
        raise RuntimeError("Enflame KV zeroer has no initialized device-address table")
    seg_addrs, seg_block_strides, seg_page_sizes, max_chunks, blk_size, n_segs = (
        self._meta
    )
    idx = async_tensor_h2d(block_ids, device=self.device, dtype=torch.int64)
    n_jobs = len(block_ids) * n_segs * max_chunks
    num_programs = min(n_jobs, GCU_MAX_GRID_X)
    _zero_kv_blocks_kernel_gcu[(num_programs,)](
        seg_addrs,
        seg_block_strides,
        seg_page_sizes,
        idx,
        n_jobs,
        BLOCK_SIZE=blk_size,
        N_SEGS=n_segs,
        MAX_CHUNKS=max_chunks,
    )


def apply_kv_zeroer_gcu_patch() -> None:
    """Install in each Enflame worker; leave other platforms unchanged."""
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) not in ("gcu", "enflame"):
        return
    from vllm.v1.worker import utils as worker_utils

    # Keep the official kernel and builder; translate its address table once.
    zeroer_class = worker_utils.KVBlockZeroer
    if (
        zeroer_class.__init__ is init_kv_zeroer_gcu
        and zeroer_class.zero_block_ids is zero_block_ids_gcu
    ):
        return
    if not hasattr(zeroer_class, "_fl_gcu_original_init"):
        zeroer_class._fl_gcu_original_init = zeroer_class.__init__
    zeroer_class.__init__ = init_kv_zeroer_gcu
    zeroer_class.zero_block_ids = zero_block_ids_gcu
    logger.info("Installed Enflame KV zeroer with device addresses and a bounded grid")
