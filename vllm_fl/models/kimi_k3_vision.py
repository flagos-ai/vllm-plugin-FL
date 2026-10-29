# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded K3 vision execution through native component factories."""

import torch
from vllm_fl.dispatch import CachedOp

apply_rope = CachedOp("kimi_k3_vision_rope")

from vllm.model_executor.models.kimi_k25_vit import (
    MLP2,
    MoonViT3dEncoder,
    MoonViT3dPretrainedModel,
    MoonViTEncoderLayer,
    Rope2DPosEmbRepeated,
)






class K3VisionRope(Rope2DPosEmbRepeated):
    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid."""
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = (
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        # Move the real/imaginary pairs; the complex RoPE values are unchanged.
        pairs = torch.cat(
            [
                torch.view_as_real(x_cis).unsqueeze(-2),
                torch.view_as_real(y_cis).unsqueeze(-2),
            ],
            dim=-2,
        )
        freqs_cis = torch.view_as_complex(pairs)
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(
        self, grid_thws: torch.Tensor | list[list[int]], device: torch.device
    ) -> torch.Tensor:
        """
        Args:
            grid_thws (torch.Tensor): grid time, height and width

        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if not hasattr(self, "freqs_cis"):
            self.register_buffer(
                "freqs_cis", self._precompute_freqs_cis(device), persistent=False
            )

        shapes = grid_thws if isinstance(grid_thws, list) else grid_thws.tolist()
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )
        pairs = torch.view_as_real(self.freqs_cis)
        selected = torch.cat(
            [
                pairs[:h, :w].reshape(-1, self.dim // 2, 2).repeat(t, 1, 1)
                for t, h, w in shapes
            ],
            dim=0,
        )
        freqs_cis = torch.view_as_complex(selected)
        return freqs_cis


class K3VisionMLP(MLP2):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_data_parallel or x.ndim != 2 or x.shape[0] < 16384:
            return super().forward(x)
        output = torch.empty(
            (x.shape[0], self.fc1.output_size), dtype=x.dtype, device=x.device
        )
        for start in range(0, x.shape[0], 8192):
            stop = min(start + 8192, x.shape[0])
            block = super().forward(x[start:stop])
            output[start:stop].copy_(block)
            del block
        return output


class K3VisionEncoderLayer(MoonViTEncoderLayer):
    mlp_cls = K3VisionMLP
    rotary_embedding = staticmethod(apply_rope)


class K3VisionEncoder(MoonViT3dEncoder):
    rope_cls = K3VisionRope
    encoder_layer_cls = K3VisionEncoderLayer


class K3VisionModel(MoonViT3dPretrainedModel):
    encoder_cls = K3VisionEncoder
