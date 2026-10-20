# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Scope MiniMax-M3 prompt reasoning-state detection to the current message."""

from collections.abc import Sequence

from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser


@ReasoningParserManager.register_module("minimax_m3_scoped")
class MiniMaxM3ScopedReasoningParser(MiniMaxM3ReasoningParser):
    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # The M3 system instructions and previous turns contain think markers.
        # Only the final message's suffix may determine the current state.
        message_start = self.vocab.get("]~b]")
        if message_start is None:
            raise ValueError("MiniMax-M3 message boundary token is missing")
        for index in range(len(input_ids) - 1, -1, -1):
            if input_ids[index] == message_start:
                return super().is_reasoning_end(input_ids[index + 1 :])
        # Generated-token sequences contain no message header.
        return super().is_reasoning_end(input_ids)
