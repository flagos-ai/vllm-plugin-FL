# Copyright (c) 2026 BAAI. All rights reserved.
"""Lazy FlagCX comm init and capture-stream bind on PTPU."""

from __future__ import annotations

import ctypes
import logging
import os
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.distributed.utils import StatelessProcessGroup

try:
    import torch.ptpu as torch_ptpu
except ImportError:
    torch_ptpu = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


def patch_flagcx_comm_lifecycle() -> None:
    """Defer ``flagcxCommInitRank`` to first collective; bind comm on capture stream."""
    try:
        from vllm.platforms import current_platform

        if current_platform.device_type != "ptpu":
            return
    except Exception:
        return

    # Import flagcx first so FLAGCX_PATH is on sys.path for plugin imports.
    from vllm_fl.distributed.device_communicators.flagcx import (
        FLAGCXLibrary,
        PyFlagcxCommunicator,
        flagcxUniqueId,
    )

    if flagcxUniqueId is None or FLAGCXLibrary is None:
        logger.warning("FlagCX not available; skipping comm lifecycle patch")
        return

    if getattr(PyFlagcxCommunicator, "_sunrise_comm_lifecycle_patched", False):
        return

    assert torch_ptpu is not None, "torch.ptpu required for PTPU comm lifecycle patch"

    def _ptpu_init(
        self,
        group: Union[ProcessGroup, StatelessProcessGroup],
        device: Union[str, torch.device],
        library_path: Optional[str] = None,
    ) -> None:
        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            assert dist.get_backend(group) != dist.Backend.NCCL, (
                "PyNcclCommunicator should be attached to a non-NCCL group."
            )
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        else:
            self.rank = group.rank
            self.world_size = group.world_size

        self.group = group

        if self.world_size == 1:
            self.available = False
            self.disabled = True
            return

        try:
            if library_path is None:
                flagcx_path = os.getenv("FLAGCX_PATH")
                library_path = os.path.join(flagcx_path, "build/lib/libflagcx.so")
                self.flagcx = FLAGCXLibrary(library_path)
            else:
                self.flagcx = FLAGCXLibrary(library_path)
        except Exception:
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        if self.rank == 0:
            self.unique_id = self.flagcx.flagcxGetUniqueId()
        else:
            self.unique_id = flagcxUniqueId()

        if not isinstance(group, StatelessProcessGroup):
            tensor = torch.ByteTensor(list(self.unique_id.internal))
            ranks = dist.get_process_group_ranks(group)
            dist.broadcast(tensor, src=ranks[0], group=group)
            byte_list = tensor.tolist()
            for i, byte in enumerate(byte_list):
                self.unique_id.internal[i] = byte
        else:
            self.unique_id = group.broadcast_obj(self.unique_id, src=0)

        if isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device
        self._device_ctx = torch_ptpu.device(self.device)
        self.comm = None

    def _ensure_initialized(self) -> None:
        if self.comm is not None:
            return
        if not getattr(self, "available", False) or self.disabled:
            return

        with self._device_ctx:
            self.comm = self.flagcx.flagcxCommInitRank(
                self.world_size, self.unique_id, self.rank
            )

    def bind_comm_to_active_capture_stream(self) -> None:
        if self.disabled:
            return
        self._ensure_initialized()
        if self.comm is None:
            return
        from vllm.platforms import current_platform

        stream = torch_ptpu.current_stream(self.device)
        data = torch.zeros(1, device=self.device)
        self.all_reduce(data, stream=stream)
        current_platform.torch_device_fn.synchronize()

    def _wrap_lazy(method):
        def wrapper(self, *args, **kwargs):
            if not self.disabled:
                self._ensure_initialized()
            return method(self, *args, **kwargs)

        wrapper.__name__ = getattr(method, "__name__", "wrapper")
        wrapper.__qualname__ = getattr(method, "__qualname__", wrapper.__name__)
        return wrapper

    _lazy_methods = (
        "all_gatherv",
        "reduce_scatterv",
        "group_start",
        "group_end",
    )
    _saved = {name: getattr(PyFlagcxCommunicator, name) for name in _lazy_methods}

    PyFlagcxCommunicator.__init__ = _ptpu_init
    PyFlagcxCommunicator._ensure_initialized = _ensure_initialized
    PyFlagcxCommunicator.bind_comm_to_active_capture_stream = (
        bind_comm_to_active_capture_stream
    )
    for name in _lazy_methods:
        setattr(PyFlagcxCommunicator, name, _wrap_lazy(_saved[name]))

    PyFlagcxCommunicator._sunrise_comm_lifecycle_patched = True
    logger.info("Patched FlagCX comm lifecycle for Sunrise/PTPU")
