# Copyright (c) 2026 BAAI. All rights reserved.

from __future__ import annotations

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    is_fp8,
)


def _get_masked_input_and_mask_eager(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PTPU fallback with no torch.compile decorator to keep eager mode."""
    org_vocab_mask = (input_ >= org_vocab_start_index) & (
        input_ < org_vocab_end_index
    )
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index
    )
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (
        added_offset * added_vocab_mask
    )
    vocab_mask = org_vocab_mask | added_vocab_mask
    masked_input = vocab_mask * (input_ - valid_offset)
    return masked_input, ~vocab_mask


class SunriseVocabParallelEmbedding(VocabParallelEmbedding):
    """Sunrise/PTPU version that use eager mode get_masked_input_and_mask."""

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return self.quant_method.embedding(self, input_.long())

        if self.tp_size > 1:
            masked_input, input_mask = _get_masked_input_and_mask_eager(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            masked_input, input_mask = input_, None

        output_parallel = self.quant_method.embedding(self, masked_input.long())
        output_dtype = output_parallel.dtype
        if is_fp8(output_dtype):
            # Each vocabulary token has one owner; reduce FP8 storage as bytes.
            output_parallel = output_parallel.view(torch.int8)
        output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        if self.reduce_results:
            output_parallel = (
                self.parallel_group.all_reduce(output_parallel)
                if self.parallel_group is not None
                else tensor_model_parallel_all_reduce(output_parallel)
            )
        return output_parallel.view(output_dtype)

    def forward_native(self, input_: torch.Tensor) -> torch.Tensor:
        return self.forward(input_)
