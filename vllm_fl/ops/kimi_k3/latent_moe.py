"""Store K3 latent projections as TP shards while retaining the original tail."""

import torch
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.models.common.ops.sequence_parallel import sp_all_gather, sp_shard
from vllm.models.kimi_k3.nvidia.latent_moe_runner import LatentMoERunner, LatentTailTier

from vllm_fl.dispatch import CachedOp

project_grouped = CachedOp("kimi_k3_latent_projection")
accumulate_latent_up = CachedOp("kimi_k3_latent_up")





class LatentColumnParallelLinear(ColumnParallelLinear):
    """Use FlagGems mm with row tiling and vLLM's canonical weight loader."""

    def __init__(self, *args, sequence_parallel: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sequence_parallel = sequence_parallel
        if sequence_parallel and not self.gather_output:
            raise ValueError("SP latent projection must restore the full feature axis")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if x.ndim != 2 or self.bias is not None or not self.return_bias:
            raise ValueError("K3 latent projections require a bias-free 2D input")
        # Different TP ranks own different token rows under SP. Gather rows
        # first, then gather projected columns, and restore the owned rows.
        if self.sequence_parallel:
            x = sp_all_gather(x)
        output = self.project_local(x)
        if self.gather_output and self.tp_size > 1:
            output = tensor_model_parallel_all_gather(output)
        if self.sequence_parallel:
            output = sp_shard(output)
        return output, None

    def project_local(
        self, x: torch.Tensor, *, out_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        return project_grouped(x, self.weight, out_dtype=out_dtype)


class ShardedLatentMoERunner(LatentMoERunner):
    """Use the existing column tail with a physically sharded up weight."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sequence_parallel = self.moe_config.is_sequence_parallel
        if (
            sequence_parallel
            or self.moe_config.ep_size != 1
            or self.moe_config.dp_size != 1
            or self.moe_config.tp_size != 8
            or not self._use_fused_path()
        ):
            raise NotImplementedError(
                "K3 latent tail requires TP8/DP1 with EP/SP disabled"
            )
        transform = self.routed_output_transform
        if transform is None or transform.norm is None:
            raise ValueError("The K3 latent output transform requires RMSNorm")
        projection = transform.up_proj
        if (
            (not sequence_parallel and projection.tp_size != self.moe_config.tp_size)
            or projection.gather_output != sequence_parallel
            or projection.weight.shape[0] * projection.tp_size != projection.output_size
        ):
            raise ValueError("Expected a column-parallel up-projection shard")

    def _select_tail_tier(
        self, fused_output: torch.Tensor, shared_output: torch.Tensor
    ) -> LatentTailTier:
        return LatentTailTier.COLUMN_PARALLEL

    def _shard_up_proj_tail(
        self,
        fused_output: torch.Tensor,
        shared_output: torch.Tensor,
        trunc_size: int | None,
    ) -> torch.Tensor:
        transform = self.routed_output_transform
        assert transform is not None and transform.norm is not None
        latent = self.allreduce_norm_latent_out(fused_output, transform.norm)
        weight = transform.up_proj.weight
        shard_start = get_tensor_model_parallel_rank() * weight.shape[0]
        hidden_shard = shared_output.narrow(-1, shard_start, weight.shape[0])
        accumulate_latent_up(hidden_shard, latent, weight)
        return self._maybe_reduce_final_output(
            shared_output, trunc_size, output_is_reduced=False
        )
