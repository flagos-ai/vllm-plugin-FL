# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from vLLM-Ascend v0.20.2rc1:
# vllm_ascend/ops/mm_encoder_attention.py and
# vllm_ascend/device/device_op.py::DeviceOperator.npu_flash_attention.
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

MIN_PAD_SIZE: int = 64
MAX_PAD_SIZE: int = 128


class AscendMMEncoderAttention(MMEncoderAttention):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
    ) -> None:
        """Ascend multimodal encoder attention backed by fused FA.

        The implementation follows vLLM-Ascend v0.20.2rc1. In particular,
        it must not materialize a ``[batch, heads, seq, seq]`` attention-score
        tensor: profile runs use the maximum multimodal token budget and such
        a tensor can be tens of GiB.
        """
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
        )

        self.enable_pad = MIN_PAD_SIZE < self.head_size < MAX_PAD_SIZE
        self.scale_value = self.head_size**-0.5

    @classmethod
    def maybe_compute_seq_lens(
        cls,
        attn_backend: AttentionBackendEnum,
        cu_seqlens: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor | None:
        if cu_seqlens is None:
            return None

        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        return torch.from_numpy(seq_lens).to("cpu", non_blocking=True)

    def _reshape_qkv_to_3d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = query.view(bsz * q_len, self.num_heads, self.head_size)
        key = key.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz * kv_len, self.num_kv_heads, self.head_size)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        if (num_repeat := self.num_queries_per_kv) > 1:
            key = torch.repeat_interleave(key, num_repeat, dim=1)
            value = torch.repeat_interleave(value, num_repeat, dim=1)

        return query, key, value

    @staticmethod
    def _maybe_compute_cu_seqlens(
        bsz: int,
        q_len: int,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cu_seqlens is not None:
            return cu_seqlens

        return torch.arange(
            0,
            (bsz + 1) * q_len,
            step=q_len,
            dtype=torch.int32,
            device="cpu",
        )

    @staticmethod
    def _npu_flash_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        head_num: int,
        scale_value: float,
        num_kv_heads: int,
    ) -> torch.Tensor:
        context_layer = torch.empty_like(query)
        torch_npu._npu_flash_attention_unpad(
            query=query,
            key=key,
            value=value,
            seq_len=seq_lens_cpu,
            scale_value=scale_value,
            num_heads=head_num,
            num_kv_heads=num_kv_heads,
            out=context_layer,
        )
        return context_layer

    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() == 4

        if sequence_lengths is not None:
            if sequence_lengths.device.type != "cpu":
                sequence_lengths = sequence_lengths.to("cpu")
            seq_lens_cpu = sequence_lengths
        else:
            cu_seqlens = self._maybe_compute_cu_seqlens(
                bsz, q_len, cu_seqlens
            )
            seq_lens_cpu = torch.diff(cu_seqlens).to("cpu")

        q, k, v = self._reshape_qkv_to_3d(
            query, key, value, bsz, q_len, kv_len
        )

        if self.enable_pad:
            origin_shape = q.shape[-1]
            pad_len = MAX_PAD_SIZE - origin_shape
            q = F.pad(q, (0, pad_len), mode="constant", value=0)
            k = F.pad(k, (0, pad_len), mode="constant", value=0)
            v = F.pad(v, (0, pad_len), mode="constant", value=0)

        context_layer = self._npu_flash_attention(
            query=q,
            key=k,
            value=v,
            seq_lens_cpu=seq_lens_cpu,
            head_num=self.num_heads,
            scale_value=self.scale_value,
            num_kv_heads=self.num_kv_heads,
        )

        if self.enable_pad:
            context_layer = context_layer[..., :origin_shape]

        if is_reshaped:
            context_layer = einops.rearrange(
                context_layer, "(b s) h d -> b s h d", b=bsz
            ).contiguous()
        else:
            context_layer = einops.rearrange(
                context_layer, "(b s) h d -> b s (h d)", b=bsz
            ).contiguous()
        return context_layer
