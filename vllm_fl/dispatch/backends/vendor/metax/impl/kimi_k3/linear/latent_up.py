"""TP8 U-shard accumulation with the unchanged pinned FlagGems addmm kernel."""
import torch
from flag_gems import runtime
from flag_gems.ops.addmm import addmm_kernel
from vllm_fl.dispatch.backends.flaggems.impl.kimi_k3_mm import raw_kernel

_LATENT_UP_KERNEL = raw_kernel(addmm_kernel)

def accumulate_latent_up(hidden_shard, latent, weight):
    """Direct U accumulation for verified TP8 layouts and disjoint inputs."""
    m = latent.shape[0]
    if (
        runtime.device.vendor_name == "metax"
        and m in (1, 2, 4, 8, 16, 32, 64, 128, 512)
        and latent.shape == (m, 3584)
        and weight.shape == (896, 3584)
        and hidden_shard.shape == (m, 896)
        and latent.dtype == weight.dtype == hidden_shard.dtype == torch.bfloat16
        and latent.is_cuda
        and latent.device == weight.device == hidden_shard.device
        and latent.is_contiguous()
        and weight.is_contiguous()
        and hidden_shard.stride(1) == 1
        and (m == 1 or hidden_shard.stride(0) == 7168)
        and hidden_shard.untyped_storage().data_ptr()
        not in (
            latent.untyped_storage().data_ptr(),
            weight.untyped_storage().data_ptr(),
        )
    ):
        # Same canonical addmm arithmetic, now tiled for the TP8 U shard.
        bm, bn, bk, warps, stages = (
            (32, 64, 64, 4, 2) if m == 512 else (16, 32, 32, 4, 4)
        )
        wt = weight.t()
        _LATENT_UP_KERNEL[((m + bm - 1) // bm, (896 + bn - 1) // bn)](
            latent,
            wt,
            hidden_shard,
            hidden_shard,
            1,
            1,
            m,
            896,
            3584,
            latent.stride(0),
            latent.stride(1),
            wt.stride(0),
            wt.stride(1),
            hidden_shard.stride(0),
            hidden_shard.stride(1),
            hidden_shard.stride(0),
            hidden_shard.stride(1),
            BIAS_IS_VECTOR=False,
            BIAS_IS_SCALAR=False,
            HAS_K=True,
            IS_FP64=False,
            BLOCK_SIZE_M=bm,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=bk,
            num_warps=warps,
            num_stages=stages,
        )
        return hidden_shard
    return hidden_shard.addmm_(latent, weight.t())
