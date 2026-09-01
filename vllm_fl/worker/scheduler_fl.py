# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project

from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler as _AsyncScheduler

logger = init_logger(__name__)


class _MTPStructuredOutputManager:
    """Batch xGrammar mask filling for MTP=1 without modifying vLLM."""

    def __init__(self, base_manager: Any):
        import xgrammar as xgr

        self._base = base_manager
        self._batch_matcher = xgr.BatchGrammarMatcher(max_threads=8)
        self._logged_active = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def grammar_bitmask(
        self,
        requests: dict[str, Any],
        structured_output_request_ids: list[str],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ):
        base = self._base
        if (
            base.vllm_config.num_speculative_tokens != 1
            or len(structured_output_request_ids) <= 1
            or any(
                base._get_reasoner(requests[req_id]) is not None
                for req_id in structured_output_request_ids
            )
        ):
            return base.grammar_bitmask(
                requests,
                structured_output_request_ids,
                scheduled_spec_decode_tokens,
            )

        if not self._logged_active:
            logger.warning(
                "Using plugin-native xGrammar batch mask filling for MTP=1."
            )
            self._logged_active = True

        if base._grammar_bitmask is None:
            if base.backend is None:
                return base.grammar_bitmask(
                    requests,
                    structured_output_request_ids,
                    scheduled_spec_decode_tokens,
                )
            max_batch_size = base.vllm_config.scheduler_config.max_num_seqs
            base._grammar_bitmask = base.backend.allocate_token_bitmask(
                max_batch_size * 2
            )

        cumulative_index = 0
        current_rows: list[tuple[Any, int, bool]] = []
        bonus_rows: list[tuple[Any, int, bool]] = []
        states: list[tuple[Any, str, list[int] | tuple[int, ...]]] = []

        for req_id in structured_output_request_ids:
            structured_request = requests[req_id].structured_output_request
            assert structured_request is not None
            grammar = structured_request.grammar
            assert grammar is not None
            req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
            assert len(req_tokens) <= 1, req_tokens

            if req_tokens:
                current_rows.append((grammar, cumulative_index, True))
                cumulative_index += 1
            bonus_rows.append((grammar, cumulative_index, True))
            cumulative_index += 1
            states.append((grammar, req_id, req_tokens))

        def batch_fill(rows: list[tuple[Any, int, bool]]) -> bool:
            matchers = []
            indices = []
            for grammar, index, apply_bitmask in rows:
                if apply_bitmask and not grammar.is_terminated():
                    matcher = getattr(grammar, "matcher", None)
                    if matcher is None:
                        return False
                    matchers.append(matcher)
                    indices.append(index)
                else:
                    base._grammar_bitmask[index].fill_(base._full_mask)
            if matchers:
                self._batch_matcher.batch_fill_next_token_bitmask(
                    matchers, base._grammar_bitmask, indices
                )
            return True

        if not batch_fill(current_rows):
            return base.grammar_bitmask(
                requests,
                structured_output_request_ids,
                scheduled_spec_decode_tokens,
            )

        advanced_grammars = []
        for grammar, req_id, req_tokens in states:
            if req_tokens and req_tokens[0] != -1 and not grammar.is_terminated():
                accepted = grammar.accept_tokens(req_id, [req_tokens[0]])
                assert accepted, (
                    req_tokens[0],
                    req_id,
                    scheduled_spec_decode_tokens,
                )
                advanced_grammars.append(grammar)

        if not batch_fill(bonus_rows):
            for grammar in advanced_grammars:
                grammar.rollback(1)
            return base.grammar_bitmask(
                requests,
                structured_output_request_ids,
                scheduled_spec_decode_tokens,
            )

        for grammar in advanced_grammars:
            grammar.rollback(1)

        bitmask_tensor = base._grammar_bitmask
        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]
        return bitmask_tensor.numpy()


class AsyncSchedulerFL(_AsyncScheduler):
    """vLLM async scheduler with plugin-owned MTP xGrammar batching."""

    def __init__(self, *args, **kwargs):
        vllm_config = kwargs.get("vllm_config")
        manager = kwargs.get("structured_output_manager")
        if (
            vllm_config is not None
            and vllm_config.num_speculative_tokens == 1
            and manager is not None
        ):
            kwargs["structured_output_manager"] = _MTPStructuredOutputManager(manager)
            logger.warning(
                "Enabled plugin-native xGrammar batch mask filling for MTP=1 "
                "(max_threads=8)."
            )
        super().__init__(*args, **kwargs)
