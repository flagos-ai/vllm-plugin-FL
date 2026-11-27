# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 BAAI. All rights reserved.
"""V4.1 compressor projection and request-owned ring orchestration.

Adapted from vLLM e77daef89e18; all numerical pooling/norm lives in FGV.
"""

import torch
from torch import nn

from .projections import V41FP32MergedProjection
from vllm_fl.dispatch import resolve_op


class DeepseekV41Compressor(nn.Module):
    def __init__(self, vllm_config, ratio, *, prefix):
        super().__init__()
        if ratio not in (1, 2):
            raise ValueError("V4.1 compressor ratio must be 1 or 2")
        config = vllm_config.model_config.hf_config
        self.ratio = ratio
        self.eps = config.rms_norm_eps
        self.fused_wkv_wgate = V41FP32MergedProjection(
            config.hidden_size, [512] * ratio, prefix=prefix + ".fused_wkv_wgate"
        )
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(
            torch.empty(512, dtype=torch.bfloat16), requires_grad=False
        )
        self.compress = resolve_op("deepseek_v41_compress_norm")
        self.register_buffer(
            "state_cache",
            (
                torch.zeros(
                    (vllm_config.scheduler_config.max_num_seqs, 8, 1024),
                    dtype=torch.float32,
                )
                if ratio == 2
                else None
            ),
            persistent=False,
        )

    def reset_request(self, request_index):
        if self.state_cache is not None:
            self.state_cache[request_index].zero_()

    def forward(
        self,
        hidden,
        positions,
        cache_metadata,
        *,
        state_slots=None,
        token_to_req=None,
        query_start_loc=None,
    ):
        if self.ratio == 2 and any(
            t is None for t in (state_slots, token_to_req, query_start_loc)
        ):
            raise ValueError("ratio-2 compressor requires V2 request-ring metadata")
        scores = self.fused_wkv_wgate(hidden)
        latent = torch.empty(
            (hidden.shape[0], 512), device=hidden.device, dtype=torch.bfloat16
        )
        self.compress(
            scores,
            positions,
            self.state_cache,
            state_slots if self.ratio == 2 else cache_metadata.slot_mapping,
            query_start_loc if self.ratio == 2 else None,
            token_to_req if self.ratio == 2 else None,
            self.norm.weight,
            self.eps,
            self.ratio,
            latent,
        )
        return latent
