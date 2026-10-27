# SPDX-License-Identifier: Apache-2.0
"""Single-node CPU publication buffer for Qwen N-gram lookup results.

The large table stays private to the producer. Only bounded gathered rows use
shared memory. Every consumer copies to its own pinned staging before release,
so no asynchronous H2D transfer can observe a later publication.
"""
from __future__ import annotations

import mmap
import os
import tempfile

import torch
import torch.distributed as dist


class SharedNgramBuffer:
    def __init__(self, group, max_tokens: int, embedding_dim: int, dtype):
        self.group = group
        self.rank = group.rank_in_group
        self.world_size = group.world_size
        self.is_producer = self.rank == 0
        self.max_tokens = max_tokens
        self.embedding_dim = embedding_dim
        self.productions = 0
        self.consumptions = 0
        self._epoch = 0
        self._closed = False
        size = 4096 + max_tokens * embedding_dim * torch.empty((), dtype=dtype, device="cpu").element_size()
        path = None
        if self.is_producer:
            fd, path = tempfile.mkstemp(prefix="qwen38-ngram-", dir="/dev/shm")
            os.ftruncate(fd, size)
            os.close(fd)
        payload = [path]
        if self.world_size > 1:
            dist.broadcast_object_list(payload, src=group.ranks[0], group=group.cpu_group)
        path = payload[0]
        fd = os.open(path, os.O_RDWR)
        try:
            self._mapping = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
        finally:
            os.close(fd)
        self.header = torch.frombuffer(self._mapping, dtype=torch.int64, count=3)
        self.rows = torch.frombuffer(
            self._mapping, dtype=dtype,
            count=max_tokens * embedding_dim, offset=4096,
        ).reshape(max_tokens, embedding_dim)
        self._barrier()
        if self.is_producer:
            # All processes now hold mappings; no path or resource-tracker entry
            # needs cleanup after workers exit, even after a failed engine run.
            os.unlink(path)

    def _barrier(self):
        if self.world_size > 1:
            dist.barrier(group=self.group.cpu_group)

    def exchange(self, producer, destination: torch.Tensor, capacity: int) -> int:
        """Publish once, then release only after all CPU consumers have copied.

        producer(rows) returns actual token count. Only rank zero calls it.
        Producer exceptions are published before raising, avoiding a stranded
        consumer at the ready barrier. No tensors are sent through Gloo.
        """
        if self._closed or not 0 <= capacity <= self.max_tokens:
            raise ValueError("invalid shared N-gram publication capacity/lifetime")
        error = None
        if self.is_producer:
            try:
                actual = int(producer(self.rows))
                if not 0 <= actual <= capacity:
                    raise ValueError("N-gram actual tokens exceed publication capacity")
                self.header.copy_(torch.tensor([self._epoch, capacity, actual], dtype=torch.int64, device="cpu"))
                self.productions += 1
            except Exception as exc:
                error = exc
                self.header.copy_(torch.tensor([self._epoch, capacity, -1], dtype=torch.int64, device="cpu"))
        self._barrier()
        epoch, published_capacity, actual = self.header.tolist()
        if actual < 0:
            raise RuntimeError("N-gram CPU producer failed before publication") from error
        if epoch != self._epoch or published_capacity != capacity:
            raise RuntimeError("N-gram TP ranks disagree on publication generation/shape")
        if actual:
            destination[:actual].copy_(self.rows[:actual])
        self.consumptions += 1
        self._barrier()
        self._epoch += 1
        return actual

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.rows = None
        self.header = None
        self._mapping.close()
