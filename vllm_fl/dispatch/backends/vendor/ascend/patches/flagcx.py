# Copyright (c) 2026 BAAI. All rights reserved.

from threading import Lock

import torch


class FlagCXAllReduceStream:
    """Join FlagCX's HCCL submission stream to the calling NPU stream."""

    def __init__(self, device):
        self.stream = torch.npu.Stream(device=device)
        self.input_ready = torch.npu.Event()
        self.output_ready = torch.npu.Event()
        self._submit_lock = Lock()

    def submit(self, all_reduce, in_tensor, out_tensor, op, caller_stream):
        # HCCL can add a stream dependency during capture. Submit on a
        # dedicated stream and join it back explicitly. Hold the lock across
        # enqueue so another caller cannot re-record an event before its wait.
        with self._submit_lock:
            self.input_ready.record(caller_stream)
            self.stream.wait_event(self.input_ready)
            with torch.npu.stream(self.stream):
                all_reduce(in_tensor, out_tensor, op, self.stream)
            self.output_ready.record(self.stream)
            caller_stream.wait_event(self.output_ready)
