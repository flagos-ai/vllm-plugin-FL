# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM v0.32.0 for FL operator dispatch.
"""GPU-resident Qwen4Exp position-learning enhancement layers."""

import torch

from vllm.models.common.ops.sequence_parallel import sp_all_gather, sp_shard
from vllm.models.qwen4_exp.nvidia import ple_layer as upstream
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLEGroupedNorm

from .ops import ple_gate


class Qwen4ExpPLELayer(upstream.Qwen4ExpPLELayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if (
            not self.use_sequence_parallel
            and input_ids.shape[0] != hidden_states.shape[0]
        ):
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same "
                f"token length, got {input_ids.shape[0]} and "
                f"{hidden_states.shape[0]}"
            )
        embeddings = self.ple_embedding(
            hidden_states,
            input_ids,
            query_start_loc,
            ngram_context,
        )
        assert embeddings.shape[0] == hidden_states.shape[0]
        embeddings = self._dequantize_embeddings(embeddings, hidden_states.dtype)
        kv, _ = self.kv_proj(embeddings)
        key, value = kv.split(self.kv_proj.output_sizes, dim=-1)
        gated_output, conv_input = ple_gate(
            key,
            value,
            hidden_states,
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            self.norm_key.eps,
        )
        if self.use_sequence_parallel:
            # Only convolution needs full tokens. Add residuals on local shards.
            conv_input = sp_all_gather(conv_input)[: input_ids.shape[0]]
            # Metadata may omit padding tokens or all tokens during profiling.
            conv_output = torch.zeros_like(conv_input)
            self._short_conv(conv_input, conv_output, outer_residual=None)
            gated_output.add_(sp_shard(conv_output))
            gated_output.add_(hidden_states)
        else:
            self._short_conv(conv_input, gated_output, hidden_states)
        return gated_output


__all__ = [
    "Qwen4ExpPLEGroupedNorm",
    "Qwen4ExpPLELayer",
]
