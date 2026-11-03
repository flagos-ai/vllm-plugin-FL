# SPDX-License-Identifier: Apache-2.0
"""One CPU N-gram table per TP host, with per-process device registrations."""

import mmap
import os
import tempfile

import torch
import torch.distributed as dist


class SharedHostTable:
    """Keep the shared allocation alive through CPU tensors and device views."""

    def __init__(self, group, rows, width, dtype):
        self.group = group
        self.rank = group.rank_in_group
        self.tensor = None
        self.view = None
        self._mapping = None
        nbytes = (
            rows * width * torch.empty((), dtype=dtype, device="cpu").element_size()
        )
        payload = [None]
        if self.rank == 0:
            try:
                space = os.statvfs("/dev/shm")
                if space.f_bavail * space.f_frsize < nbytes:
                    raise RuntimeError(
                        f"Shared PLE table needs {nbytes} free /dev/shm bytes"
                    )
                fd, path = tempfile.mkstemp(prefix="qwen38-uva-table-", dir="/dev/shm")
                try:
                    os.ftruncate(fd, nbytes)
                finally:
                    os.close(fd)
                payload[0] = {"path": path}
            except Exception as error:
                payload[0] = {"error": str(error)}
        if group.world_size > 1:
            dist.broadcast_object_list(
                payload, src=group.ranks[0], group=group.cpu_group
            )
        if "error" in payload[0]:
            raise RuntimeError("Shared PLE allocation failed: " + payload[0]["error"])
        path = payload[0]["path"]
        error = None
        try:
            fd = os.open(path, os.O_RDWR)
            try:
                self._mapping = mmap.mmap(fd, nbytes, access=mmap.ACCESS_WRITE)
            finally:
                os.close(fd)
            self.tensor = torch.frombuffer(self._mapping, dtype=dtype).reshape(
                rows, width
            )
        except Exception as failure:
            error = str(failure)
        errors = self._collect_errors(error)
        if self.rank == 0:
            os.unlink(path)
        if any(errors):
            self.close()
            raise RuntimeError(f"Shared PLE mapping failed: {errors}")

    def _collect_errors(self, error):
        errors = [None] * self.group.world_size
        if self.group.world_size > 1:
            dist.all_gather_object(errors, error, group=self.group.cpu_group)
        else:
            errors[0] = error
        return errors

    def register(self, *, on_all_ranks, row_start=0, row_end=None):
        """All ranks finish loading before any GPU can read the shared table."""
        if self.group.world_size > 1:
            dist.barrier(group=self.group.cpu_group)
        error = None
        if on_all_ranks or self.rank == 0:
            try:
                from vllm_fl.dispatch.backends.vendor.metax.host_memory import (
                    register_host_tensor,
                )

                self.view = register_host_tensor(self.tensor[row_start:row_end])
            except Exception as failure:
                error = str(failure)
        errors = self._collect_errors(error)
        if any(errors):
            self.close()
            raise RuntimeError(f"PLE UVA registration failed: {errors}")

    def close(self):
        # Native view destruction unregisters after device completion. Its
        # storage retains the CPU owner, which in turn retains the mmap object.
        # Do not explicitly mmap.close while any Tensor alias still exists.
        self.view = None
        self.tensor = None
        self._mapping = None
