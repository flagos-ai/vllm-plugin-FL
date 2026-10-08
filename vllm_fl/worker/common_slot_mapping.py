# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen metadata producer using the shared attention graph lifecycle."""

from collections.abc import Callable
from typing import Any

import torch

from vllm.platforms import current_platform

from vllm_fl.dispatch import resolve_op
from vllm_fl.worker.common_attention_metadata import CommonAttentionMetadataGraphRunner

compute_common_slot_mapping = resolve_op("compute_common_slot_mapping")


class CommonSlotMappingGraphRunner(CommonAttentionMetadataGraphRunner):
    """Retain dispatch ownership and T-Head ordering for PLE/QSA metadata."""

    def run(
        self,
        block_table: Any,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        *,
        use_graph: bool,
        capture: bool,
        compute: Callable[
            [Any, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            None,
        ] = compute_common_slot_mapping,
    ) -> bool:
        used_graph = super().run(
            block_table,
            num_reqs,
            query_start_loc,
            positions,
            seq_lens,
            num_computed_tokens,
            use_graph=use_graph,
            capture=capture,
            compute=compute,
        )
        # Complete the producer before its metadata is consumed by the model
        # graph. This also covers first capture and repeated capture replay.
        if used_graph and current_platform.device_name == "thead":
            current_platform.torch_device_fn.current_stream().synchronize()
        return used_graph
