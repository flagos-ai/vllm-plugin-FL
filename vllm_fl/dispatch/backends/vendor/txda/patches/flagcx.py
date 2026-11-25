# Copyright (c) 2026 BAAI. All rights reserved.

"""Keep FlagCX collectives off TXDA's legacy default stream.

FlagCX hands the caller's stream to the collectives library rendered for TXDA
(tccl, an NCCL fork).  When that stream is the legacy *default* stream — which
is what ``torch.txda.current_stream()`` returns for the main thread, with a
NULL native handle — NCCL's ``ncclLaunchPrepare`` has to honour the default
stream's implicit ordering and waits for it with ``txStreamSynchronize``
(``ncclStreamWaitStream(launchStream=0x0, ...)``, enqueue.cc:1518).  On TXDA
that synchronise never returns: ``txr::Event::awaitCompletion`` spins in
``sched_yield`` at 100% CPU, so the whole engine wedges during startup.

The failure only shows up once cudagraph is enabled, because the capture path
leaves work queued on the default stream.  Without capture the same collectives
run fine, which is why eager mode never hit it.

Submitting on a dedicated stream and joining it back with events keeps the
collective on a real (non-NULL) handle, so NCCL never takes the legacy-default
ordering path.  This mirrors ``FlagCXAllReduceStream`` for NPU, which exists
for the same reason (HCCL cannot add a stream dependency during capture).
"""

from threading import Lock

import torch


class TXDAFlagCXAllReduceStream:
    """Join FlagCX's submission stream to the calling TXDA stream."""

    def __init__(self, device):
        self.stream = torch.txda.Stream(device=device)
        self.input_ready = torch.txda.Event()
        self.output_ready = torch.txda.Event()
        self._submit_lock = Lock()

    def submit(self, all_reduce, in_tensor, out_tensor, op, caller_stream):
        # Record/wait on events rather than relying on the legacy default
        # stream's implicit ordering: that ordering is exactly what makes
        # NCCL synchronise the NULL handle.
        with self._submit_lock:
            self.input_ready.record(caller_stream)
            self.stream.wait_event(self.input_ready)
            with torch.txda.stream(self.stream):
                all_reduce(in_tensor, out_tensor, op, self.stream)
            self.output_ready.record(self.stream)
            caller_stream.wait_event(self.output_ready)
