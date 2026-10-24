# Copyright (c) 2026 BAAI. All rights reserved.
"""Qwen PLE state adapter for the legacy model runner."""

from typing import Any

import numpy as np
import torch

from vllm.distributed import get_pp_group

from .ple_token_history import PLETokenHistory
from vllm_fl.worker.model_state import ModelRunnerState


class Qwen3_8FlashNextLegacyModelState(ModelRunnerState):
    def __init__(self, runner):
        super().__init__(runner)
        config = runner.model_config.hf_text_config
        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        if not self.uses_ngram_embedding:
            return
        self.ngram_context_len = int(config.ngram_size) - 1
        self.ngram_eos_token_id = int(config.eos_token_id)
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        if len(get_pp_group().ranks) > 1:
            raise RuntimeError(
                "N-gram PLE embedding requires pipeline_parallel_size=1."
            )
        self.ngram_context = runner._make_buffer(
            runner.max_num_reqs, self.ngram_context_len, dtype=torch.int32
        )
        self.ple_token_history = PLETokenHistory(
            runner.max_num_reqs,
            runner.max_model_len,
            runner.max_num_tokens,
            self.ngram_context_len,
            self.ngram_eos_token_id,
            runner.device,
            runner.pin_memory,
        )

    def remove_requests(self, req_ids):
        if self.uses_ngram_embedding:
            self.ple_token_history.forget(req_ids)

    def prepare_inputs(self, num_reqs, num_reqs_padded, input_ids):
        if (
            self.uses_ngram_embedding
            and get_pp_group().is_first_rank
            and not self.runner.model_config.is_encoder_decoder
            and input_ids is None
        ):
            raise RuntimeError(
                "N-gram PLE requires token ids on the first pipeline rank; "
                "inputs_embeds-only batches are not supported."
            )
        kwargs = {}
        self._maybe_add_ngram_kwargs(
            kwargs,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            is_first_rank=get_pp_group().is_first_rank,
            is_encoder_decoder=self.runner.model_config.is_encoder_decoder,
            use_dummy_context=False,
            query_start_loc=self.runner.query_start_loc.gpu[: num_reqs_padded + 1],
        )
        return kwargs

    def prepare_dummy_inputs(self, num_reqs, num_reqs_padded, num_scheduled_tokens):
        kwargs = {}
        self._maybe_add_ngram_kwargs(
            kwargs,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            is_first_rank=get_pp_group().is_first_rank,
            is_encoder_decoder=self.runner.model_config.is_encoder_decoder,
            use_dummy_context=True,
            num_scheduled_tokens=num_scheduled_tokens,
        )
        return kwargs

    def _prepare_ngram_context(
        self,
        num_reqs: int,
        num_reqs_padded: int,
    ) -> torch.Tensor:
        """Build the left context for every real or CUDA-graph padding row."""
        if not self.uses_ngram_embedding:
            raise RuntimeError("N-gram context requested for non-ngram model.")
        num_tokens = int(self.runner.query_start_loc.np[num_reqs])
        return self.ple_token_history.prepare(
            req_ids=self.runner.input_batch.req_ids[:num_reqs],
            num_computed_tokens=self.runner.input_batch.num_computed_tokens_cpu,
            num_scheduled_tokens=self.runner.num_scheduled_tokens.np[:num_reqs],
            token_ids_cpu=self.runner.input_batch.token_ids_cpu_tensor,
            is_token_ids=(
                self.runner.input_batch.is_token_ids
                if self.runner.enable_prompt_embeds
                else None
            ),
            req_indices=self.runner.req_indices.np[:num_tokens],
            query_positions=self.runner.query_pos.np[:num_tokens],
            input_ids=self.runner.input_ids.gpu[:num_tokens],
            context=self.ngram_context.gpu[:num_reqs_padded],
        )

    def _maybe_add_ngram_kwargs(
        self,
        model_kwargs: dict[str, Any],
        *,
        num_reqs: int,
        num_reqs_padded: int,
        is_first_rank: bool,
        is_encoder_decoder: bool,
        use_dummy_context: bool,
        query_start_loc: torch.Tensor | None = None,
        num_scheduled_tokens: np.ndarray | None = None,
    ) -> None:
        if not self.uses_ngram_embedding or not is_first_rank or is_encoder_decoder:
            return

        eos_token_id = int(self.ngram_eos_token_id)
        if query_start_loc is None:
            if num_scheduled_tokens is None:
                raise RuntimeError("query_start_loc is required for N-gram input.")
            cu_num_tokens = np.cumsum(num_scheduled_tokens, dtype=np.int32)
            last = int(cu_num_tokens[-1]) if num_reqs > 0 else 0
            self.runner.query_start_loc.np[0] = 0
            if num_reqs > 0:
                self.runner.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
            self.runner.query_start_loc.np[num_reqs + 1 :].fill(last)
            self.runner.query_start_loc.copy_to_gpu()
            query_start_loc = self.runner.query_start_loc.gpu[: num_reqs_padded + 1]
        model_kwargs["query_start_loc"] = query_start_loc

        if use_dummy_context:
            self.ngram_context.np[:num_reqs_padded].fill(eos_token_id)
            self.ngram_context.copy_to_gpu(num_reqs_padded)
            model_kwargs["ngram_context"] = self.ngram_context.gpu[:num_reqs_padded]
        else:
            model_kwargs["ngram_context"] = self._prepare_ngram_context(
                num_reqs, num_reqs_padded
            )
