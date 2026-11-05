# SPDX-License-Identifier: Apache-2.0
"""Gate projection orchestration: original gate providers inside one FULL graph."""

from __future__ import annotations

import torch
from torch.nn.modules import module as nn_module

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available

from vllm_fl.utils import use_flaggems_op


class GateOverlap:
    def __init__(self):
        # No stream construction in model construction/meta initialization.
        self.stream = None
        self.device = None

    def begin(self, owner, hidden_states):
        gate = owner.linear_gate
        if (
            not use_flaggems_op("mul_scaled_int8_quant", default=False)
            or not owner.gated_mla
            or owner.use_hpc_gated_mla
            or gate is None
            or owner.config.gating_type != "elementwise"
            or not hidden_states.is_cuda
            or hidden_states.dtype != torch.bfloat16
            or not is_forward_context_available()
            or get_forward_context().cudagraph_runtime_mode
            not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL)
            or BreakableCUDAGraphCapture.current() is not None
        ):
            return None
        # Gate must remain a purely local projection. Hooks could observe the
        # changed order relative to attention, so they retain the original path.
        if (
            gate.gather_output
            or gate.bias is not None
            or gate.weight.device != hidden_states.device
            or gate.weight.dtype != hidden_states.dtype
            or gate._forward_hooks
            or gate._forward_pre_hooks
            or nn_module._global_forward_hooks
            or nn_module._global_forward_pre_hooks
        ):
            return None
        if self.stream is None:
            if torch.cuda.is_current_stream_capturing():
                return None  # Exact branch must have warmed before FULL capture.
            self.device = hidden_states.device
            self.stream = torch.cuda.Stream(device=self.device)
        if self.device != hidden_states.device:
            return None
        # Native _warmup_and_capture uses NONE. Prepare only the stream there;
        # keep its exact original serial math/provider warmup. FULL capture can
        # then use the already-created stream and already-compiled providers.
        if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.NONE:
            return None
        parent = torch.cuda.current_stream(self.device)
        if parent == self.stream:
            return None
        # Fork/join edges belong to the same capture. No host synchronization.
        self.stream.wait_stream(parent)
        hidden_states.record_stream(self.stream)
        gate.weight.record_stream(self.stream)
        try:
            with torch.cuda.stream(self.stream):
                score = gate(hidden_states)[0]
                factor = torch.sigmoid(score)
        except BaseException:
            parent.wait_stream(self.stream)
            raise
        # Strong references last until the parent joins. Capture pool owns the
        # recorded allocations thereafter, just as for ordinary graph outputs.
        return GateTicket(parent, self.stream, hidden_states, gate, score, factor)


class GateTicket:
    def __init__(self, parent, auxiliary, hidden, gate, score, factor):
        self.parent, self.auxiliary = parent, auxiliary
        self.hidden, self.gate = hidden, gate
        self.score, self.factor = score, factor
        self.joined = False

    def finish(self):
        if self.joined:
            raise RuntimeError("gate ticket already joined")
        # Also called in the original attention exception path. The join is
        # enqueued even if a caller unexpectedly changed its current stream.
        self.parent.wait_stream(self.auxiliary)
        self.factor.record_stream(self.parent)
        self.joined = True
        if torch.cuda.current_stream(self.hidden.device) != self.parent:
            raise RuntimeError("gate consumer stream changed")
        return self.factor
