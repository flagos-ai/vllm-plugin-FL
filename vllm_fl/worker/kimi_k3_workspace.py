from vllm_fl.ops.kimi_k3.workspace_contract import (
    MIN_PREFILL_TOKENS,
    MAX_PREFILL_TOKENS,
    TOP_K,
    ROUTE_COLUMNS,
    ACTIVATION_COLUMNS,
)

"""Explicit, stream-scoped temporary storage for ordinary TP8 prefill.

The model runner owns this object and reserves it inside memory profiling.
Operators borrow scratch storage; returned model outputs never alias it.
"""
import torch
import triton


class TP8PrefillWorkspace:
    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.index is None:
            raise ValueError("Workspace requires an explicit device index")
        self._buffers = {}

    def _stream_key(self):
        return int(triton.runtime.driver.active.get_current_stream(self.device.index))

    def reserve(self):
        """Reserve the current stream before KV sizing; never during capture."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Reserve the prefill workspace before Graph capture")
        stream = self._stream_key()
        if stream not in self._buffers:
            scratch = torch.empty(
                MAX_PREFILL_TOKENS * TOP_K * ROUTE_COLUMNS,
                dtype=torch.bfloat16,
                device=self.device,
            )
            act = torch.empty(
                (MAX_PREFILL_TOKENS * TOP_K, ACTIVATION_COLUMNS),
                dtype=torch.bfloat16,
                device=self.device,
            )
            self._buffers[stream] = (scratch, act)

    def get(self, device):
        """Borrow an already reserved stream's buffers; never allocate here."""
        if torch.device(device) != self.device:
            raise ValueError("Workspace belongs to another device")
        stream = self._stream_key()
        if stream not in self._buffers:
            raise RuntimeError(
                "Reserve this prefill stream before setting the KV budget"
            )
        return self._buffers[stream]

    def describe(self):
        """Return CPU metadata for optional startup diagnostics."""
        return [
            {
                "device": self.device.index,
                "stream": stream,
                "bytes": scratch.numel() * scratch.element_size()
                + act.numel() * act.element_size(),
                "arenas": len(self._buffers),
                "scratch_ptr": scratch.data_ptr(),
                "act_ptr": act.data_ptr(),
            }
            for stream, (scratch, act) in sorted(self._buffers.items())
        ]
