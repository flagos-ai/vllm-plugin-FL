"""Local TP8 MLA output gate, using a qualified small-M split-K projection."""

import torch
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
)
from vllm_fl.dispatch import CachedOp

small_split_projection = CachedOp("kimi_k3_small_projection")


class _SmallSplitProjection:
    """Same BF16 products and output rounding; FP32 K-sum is partitioned."""

    def forward(self, x):
        w = self.weight
        if not (
            self.bias is None
            and self.return_bias
            and not self.gather_output
            and x.shape == (8, 7168)
            and w.shape == (1536, 7168)
            and x.dtype == w.dtype == torch.bfloat16
            and x.is_cuda
            and x.device == w.device
            and x.is_contiguous()
            and w.is_contiguous()
        ):
            return super().forward(x)
        split = self._k_splits
        stages = self._pipeline_stages
        return small_split_projection(x, w, split, stages)


class Tp8GateProjection(_SmallSplitProjection, ColumnParallelLinear):
    """MLA output sigmoid gate; original column-parallel weight loader."""

    _k_splits = 8
    _pipeline_stages = 3


class Tp8SharedGateUpProjection(_SmallSplitProjection, MergedColumnParallelLinear):
    """Shared MLP gate/up; retain the merged checkpoint weight loader."""

    _k_splits = 16
    _pipeline_stages = 4
