# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors

"""Vision deltas: patch projection and bounded frame batches."""

import torch

from vllm.models.minimax_m3.common import vision_tower as native


class MiniMaxVLPatchEmbed(native.MiniMaxVLPatchEmbed):
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.patch_embedding.weight.dtype != pixel_values.dtype:
            self.patch_embedding = self.patch_embedding.to(pixel_values.dtype)
        assert self.patch_embedding.bias is None
        assert self.patch_embedding.groups == 1
        assert self.patch_embedding.padding == (0, 0, 0)
        assert self.patch_embedding.dilation == (1, 1, 1)
        assert self.patch_embedding.kernel_size == self.patch_embedding.stride
        from vllm_fl.ops.minimax_m3.layers import patch_embed

        return patch_embed(pixel_values, self.patch_embedding.weight)


class MiniMaxVLVisionModel(native.MiniMaxVLVisionModel):
    _m3_forward_unbounded = native.MiniMaxVLVisionModel.forward

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: list[list[int]]
    ) -> torch.Tensor:
        from vllm_fl.ops.minimax_m3.vision_chunk import vision_forward

        return vision_forward(self, pixel_values, grid_thw)
