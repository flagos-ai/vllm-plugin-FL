# Copyright (c) 2026 BAAI. All rights reserved.

import torch


@torch.library.custom_op("vllm_fl::iluvatar_gems_silu_and_mul", mutates_args=())
def _iluvatar_gems_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    # Keep FlagGems Python dtype inference outside Dynamo tracing.
    from flaggems_vllm import silu_and_mul

    d = x.shape[-1] // 2
    return silu_and_mul(x[..., :d], x[..., d:])


@_iluvatar_gems_silu_and_mul.register_fake
def _iluvatar_gems_silu_and_mul_fake(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))
