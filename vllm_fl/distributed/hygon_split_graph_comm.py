# Copyright (c) 2026 BAAI. All rights reserved.
"""Opt-in RCCL capture/eager isolation for the Hygon asynchronous TP path.

The current Hygon runtime can stall when outstanding captured and ordinary
collectives share a communicator. Keep asynchronous scheduling and stream order;
select a separate, eagerly initialized communicator during graph capture.
Replay then uses the captured communicator without Python-side synchronization.
NCCL symmetric-memory windows are deliberately outside this candidate's scope.
"""

import torch

from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_allocator import is_symmetric_memory_enabled
from vllm.logger import init_logger

logger = init_logger(__name__)


class CaptureSeparatedPyNccl:
    """Own both communicators; expose only the qualified collective interface."""

    _collectives = frozenset({
        'all_reduce', 'all_gather', 'all_gatherv', 'reduce_scatter',
        'reduce_scatterv', 'broadcast', 'send', 'recv', 'batch_isend_irecv',
        'group_start', 'group_end',
    })
    _metadata = frozenset({'rank', 'world_size', 'device', 'available'})

    def __init__(self, eager, captured):
        if eager is captured or eager.comm.value == captured.comm.value:
            raise ValueError('capture and eager communicators must be distinct')
        if any(getattr(eager, key) != getattr(captured, key) for key in self._metadata):
            raise ValueError('capture and eager communicator metadata must match')
        if not eager.available or eager.disabled or captured.disabled:
            raise RuntimeError('both RCCL communicators must be available')
        self.eager = eager
        self.captured = captured
        self._destroyed = False

    @property
    def disabled(self):
        return self.eager.disabled or self.captured.disabled

    @disabled.setter
    def disabled(self, value):
        self.eager.disabled = value
        self.captured.disabled = value

    def __getattr__(self, name):
        if name in self._metadata:
            return getattr(self.eager, name)
        if name in self._collectives:
            comm = self.captured if torch.cuda.is_current_stream_capturing() else self.eager
            return getattr(comm, name)
        raise AttributeError(f'{type(self).__name__} does not support {name}')

    def destroy(self):
        # CudaCommunicator.destroy owns this lifecycle, after graphs are cleared.
        if self._destroyed:
            return
        self._destroyed = True
        try:
            self.captured.destroy()
        finally:
            self.eager.destroy()


class HygonSplitGraphCommunicator(CudaCommunicator):
    """Preserve upstream dispatch; isolate only its TP PyNccl fallback."""

    def __init__(self, cpu_group, device=None, device_group=None, unique_name='',
                 global_ranks=None, global_world_size=None, tcp_store_group=None):
        if is_symmetric_memory_enabled():
            raise RuntimeError('Hygon split Graph communicator does not support NCCL symmetric memory')
        super().__init__(cpu_group, device, device_group, unique_name,
                         global_ranks, global_world_size, tcp_store_group)
        if unique_name.split(':')[0] != 'tp' or self.world_size <= 1:
            return
        eager = self.pynccl_comm
        if eager is None or not eager.available or eager.disabled:
            super().destroy()
            raise RuntimeError('Hygon split Graph communicator requires active PyNccl')
        captured = None
        try:
            captured = PyNcclCommunicator(
                group=cpu_group if tcp_store_group is None else tcp_store_group,
                device=self.device,
            )
            self.pynccl_comm = CaptureSeparatedPyNccl(eager, captured)
        except BaseException:
            if captured is not None:
                captured.destroy()
            super().destroy()
            raise
        logger.info('HYGON_SPLIT_GRAPH_COMM group=%s rank=%s eager=%s captured=%s',
                    unique_name, eager.rank, eager.comm.value, captured.comm.value)
