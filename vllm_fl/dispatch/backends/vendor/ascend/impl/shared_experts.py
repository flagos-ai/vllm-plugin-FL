# Copyright (c) 2026 BAAI. All rights reserved.

"""Ascend stream adapter for vLLM's modular shared-expert runner."""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401  # Registers ``torch.npu``.

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)

logger = init_logger(__name__)
_SHARED_EXPERTS_STREAM: torch.npu.Stream | None = None


def shared_experts_calculation_stream() -> torch.npu.Stream:
    """Return the single process-wide stream used by every MoE layer."""

    global _SHARED_EXPERTS_STREAM
    if _SHARED_EXPERTS_STREAM is None:
        _SHARED_EXPERTS_STREAM = torch.npu.Stream()
    return _SHARED_EXPERTS_STREAM


class AscendSharedExperts(SharedExperts):
    """Run shared experts on an auxiliary NPU stream when explicitly enabled.

    The surrounding upstream ``MoERunner`` already owns when shared experts
    are started and when their output is combined with routed experts.  Only
    CUDA-specific stream primitives prevent that implementation from working
    on Ascend, so keep the runner contract and adapt those primitives here.
    """

    def __init__(self, *args, multistream_enabled: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._ascend_multistream_enabled = multistream_enabled
        self._stream = (
            shared_experts_calculation_stream() if multistream_enabled else None
        )
        if multistream_enabled:
            logger.info_once("Enabled separate NPU stream for MoE shared experts")

    def _determine_shared_experts_order(
        self,
        hidden_states: torch.Tensor,
    ) -> SharedExpertsOrder:
        del hidden_states

        if self._disable_shared_experts_overlap:
            return SharedExpertsOrder.NO_OVERLAP
        if self._quant_method.mk_owns_shared_expert:
            return SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
        if self._ascend_multistream_enabled and self._stream is not None:
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        return SharedExpertsOrder.NO_OVERLAP

    def maybe_sync_shared_experts_stream(
        self,
        shared_experts_input: torch.Tensor,
    ) -> None:
        order = self._determine_shared_experts_order(shared_experts_input)
        if order != SharedExpertsOrder.MULTI_STREAM_OVERLAPPED:
            return

        assert self._stream is not None
        assert self._moe_config.disable_inplace
        shared_experts_input.record_stream(self._stream)
        self._stream.wait_stream(torch.npu.current_stream())

    def _run_in_aux_stream(
        self,
        shared_experts_input: torch.Tensor,
    ) -> torch.Tensor:
        assert self._stream is not None
        with torch.npu.stream(self._stream):
            output = self._layer(shared_experts_input)
        torch.npu.current_stream().wait_stream(self._stream)
        return output


__all__ = ["AscendSharedExperts", "shared_experts_calculation_stream"]
