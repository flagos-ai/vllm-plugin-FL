# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# CPU scan adapted from vLLM deepseek_ocr.py; copyright contributors to vLLM.
"""Opt-in batch masking for OCR's existing per-request NGram processor.

Only mask submission is batched. Request lifecycle and the CPU tuple scan retain
the core implementation's semantics (including its ngram_size=1 behavior).
No audit I/O is performed here: apply may run during compilation/capture.
"""

import os
from functools import partial

import torch

from vllm.model_executor.models.deepseek_ocr import (
    NGramPerReqLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
)


class NGramPerReqLogitsProcessorFL(NGramPerReqLogitsProcessor):
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if os.getenv("VLLM_FL_OCR_BATCH_NGRAM", "1") != "1":
            return super().apply(logits)

        # Validate the entire batch before making any mutation. Unknown state
        # shapes belong to core, including customized processor subclasses.
        states = []
        if logits.ndim != 2 or not isinstance(self.req_info, dict):
            return super().apply(logits)
        for req_idx, state in self.req_info.items():
            if (
                type(req_idx) is not int
                or not -logits.shape[0] <= req_idx < logits.shape[0]
                or not isinstance(state, partial)
                or type(state.func) is not NoRepeatNGramLogitsProcessor
                or len(state.args) != 1
                or state.keywords
                or not isinstance(state.args[0], list)
            ):
                return super().apply(logits)
            processor = state.func
            if (
                type(processor.ngram_size) is not int
                or processor.ngram_size <= 0
                or type(processor.window_size) is not int
                or processor.window_size <= 0
                or not isinstance(processor.whitelist_token_ids, set)
            ):
                return super().apply(logits)
            states.append((req_idx, processor, state.args[0]))

        rows, tokens = [], []
        for req_idx, processor, output_ids in states:
            ngram_size = processor.ngram_size
            window_size = processor.window_size
            if len(output_ids) < ngram_size:
                continue
            current_prefix = tuple(output_ids[-(ngram_size - 1) :])
            search_start = max(0, len(output_ids) - window_size)
            search_end = len(output_ids) - ngram_size + 1
            banned_tokens = set()
            for i in range(search_start, search_end):
                ngram = tuple(output_ids[i : i + ngram_size])
                if ngram[:-1] == current_prefix:
                    banned_tokens.add(ngram[-1])
            banned_tokens = banned_tokens - processor.whitelist_token_ids
            rows.extend([req_idx] * len(banned_tokens))
            tokens.extend(banned_tokens)

        if tokens:
            coordinates = torch.tensor(
                [rows, tokens], dtype=torch.int64, device=logits.device
            )
            logits[coordinates[0], coordinates[1]] = -float("inf")
        return logits
