# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hy4 processor parameters and device-normalization lifecycle.

The optional numerical implementation lives in FlagGems-vllm. Ordinary
processor-side normalization does not import that dependency.
"""

import math
from typing import NamedTuple

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.model_executor.custom_op import CustomOp
from vllm.transformers_utils.processor import get_processor_config


class NormParams(NamedTuple):
    """Processor flags resolved into per-channel affine parameters."""

    image_mean: list[float]
    image_std: list[float]
    rescale_factor: float

    @property
    def is_identity(self) -> bool:
        return all(
            math.isclose(self.rescale_factor / std, 1.0, rel_tol=1e-5, abs_tol=1e-8)
            and math.isclose(mean / std, 0.0, abs_tol=1e-8)
            for mean, std in zip(self.image_mean, self.image_std)
        )


def _load_norm_params(model_config: ModelConfig) -> NormParams:
    config = get_processor_config(model_config.model, revision=model_config.revision)
    config = config.get("image_processor", config)
    if not any(
        key in config
        for key in ("do_normalize", "do_rescale", "image_mean", "image_std")
    ):
        config = model_config.hf_image_processor_config
    config = config.get("media_proc_cfg", config)
    has_norm_params = "image_mean" in config and "image_std" in config
    do_normalize = bool(config.get("do_normalize", has_norm_params))
    do_rescale = bool(config.get("do_rescale", has_norm_params))
    rescale_factor = config.get("rescale_factor", 1 / 255) if do_rescale else 1.0
    image_mean = config.get("image_mean") if do_normalize else [0.0] * 3
    image_std = config.get("image_std") if do_normalize else [1.0] * 3
    if rescale_factor is None or image_mean is None or image_std is None:
        raise ValueError("Hy4 image normalization parameters must be specified")
    if not image_mean or len(image_mean) != len(image_std):
        raise ValueError("Hy4 image mean and standard deviation lengths must match")
    if any(float(std) == 0 for std in image_std):
        raise ValueError("Hy4 image standard deviations must be nonzero")
    return NormParams(
        [float(value) for value in image_mean],
        [float(value) for value in image_std],
        float(rescale_factor),
    )


class IdentityInputNorm(nn.Module):
    """Cast already normalized patches to the vision tower's input dtype."""

    @property
    def input_dtype(self) -> torch.dtype | None:
        return None

    def forward(self, pixel_values: torch.Tensor, visual_dtype: torch.dtype):
        return pixel_values.to(visual_dtype, copy=False)


def _normalization_provider():
    from vllm_fl.utils import use_flaggems_op

    if not use_flaggems_op("fused_mm_input_norm", default=False):
        raise ValueError(
            "Hy4 device image normalization requires the fused_mm_input_norm "
            "FlagGems provider; otherwise keep processor-side normalization"
        )
    from flaggems_vllm import fused_mm_input_norm

    return fused_mm_input_norm


@CustomOp.register("hy4_mm_input_norm")
class FusedMMInputNorm(CustomOp):
    """Own normalization parameters; delegate device arithmetic to the provider."""

    def __init__(self, image_mean, image_std, rescale_factor, channel=3):
        super().__init__()
        if len(image_mean) != channel or len(image_std) != channel:
            raise ValueError("Hy4 image normalization channel counts must match")
        if rescale_factor == 0 or any(std == 0 for std in image_std):
            raise ValueError(
                "Hy4 image rescaling and standard deviations must be nonzero"
            )
        self.channel = channel
        # Allocate parameters during model construction, before budgeting and
        # capture. Preserve the original FP32 parameter arithmetic and rounding.
        mean = torch.tensor(image_mean, dtype=torch.float32, device="cpu")
        std = torch.tensor(image_std, dtype=torch.float32, device="cpu")
        device = torch.get_default_device()
        self.register_buffer("weight", (rescale_factor / std).to(device))
        self.register_buffer("bias", (-mean / std).to(device))

    @classmethod
    def enabled(cls):
        return True

    @property
    def input_dtype(self):
        return torch.uint8

    def forward_native(self, pixel_values, visual_dtype):
        if pixel_values.ndim != 2:
            raise ValueError(
                "Hy4 image patches must have shape [patches, patch_elements]"
            )
        patches, size = pixel_values.shape
        if size % self.channel:
            raise ValueError(
                "Hy4 patch elements must be divisible by the channel count"
            )
        provider = _normalization_provider()
        inputs = pixel_values.reshape(patches, self.channel, size // self.channel)
        outputs = torch.empty(
            (patches, size), dtype=visual_dtype, device=pixel_values.device
        )
        provider(
            inputs.contiguous(),
            outputs.view(patches, self.channel, size // self.channel),
            self.weight.contiguous(),
            self.bias.contiguous(),
        )
        return outputs

    forward_cuda = forward_native
    forward_oot = forward_native


def build_mm_input_norm(model_config: ModelConfig) -> nn.Module:
    mm_config = getattr(model_config, "multimodal_config", None)
    if not getattr(mm_config, "mm_device_do_normalize", False):
        return IdentityInputNorm()
    params = _load_norm_params(model_config)
    if params.is_identity:
        return IdentityInputNorm()
    # Fail at construction, before allocating the vision tower, when users
    # explicitly request a device path without its selected provider.
    _normalization_provider()
    return FusedMMInputNorm(
        params.image_mean,
        params.image_std,
        params.rescale_factor,
        channel=len(params.image_mean),
    )
