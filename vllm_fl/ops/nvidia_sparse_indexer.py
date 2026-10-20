# Copyright (c) 2026 BAAI. All rights reserved.

"""Initialize upstream sparse indexers backed by real EMPTY FP8 kernels."""

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.custom_op import CustomOp, op_registry_oot
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.platforms import current_platform

from vllm_fl.nvidia_compat import has_native_vllm_kernels


class NvidiaSparseAttnIndexer(SparseAttnIndexer):
    """Keep upstream forward and state without requiring unused DeepGEMM.

    The selected model installer supplies the real FP8 scoring/cache kernels.
    This constructor changes only the upstream native-library requirement;
    other architectures and full native wheels retain upstream initialization.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        compress_ratio: int = 1,
        candidate_blocks: torch.Tensor | None = None,
        candidate_block_size: int = 0,
        candidate_write: bool = False,
    ):
        config = get_current_vllm_config()
        architectures = (
            config.model_config.architectures if config.model_config is not None else ()
        )
        is_dsv41 = "DeepseekV41ForCausalLM" in architectures
        is_hy4 = "HYV4ForCausalLM" in architectures
        if (
            has_native_vllm_kernels()
            or not current_platform.is_cuda()
            or not (is_dsv41 or is_hy4)
        ):
            super().__init__(
                k_cache,
                quant_block_size,
                scale_fmt,
                topk_tokens,
                head_dim,
                max_model_len,
                max_total_seq_len,
                topk_indices_buffer,
                skip_k_cache_insert,
                use_fp4_cache,
                compress_ratio,
                candidate_blocks,
                candidate_block_size,
                candidate_write,
            )
            return
        parallel = config.parallel_config
        capability = current_platform.get_device_capability()
        if (
            capability is None
            or capability.major != 9
            or quant_block_size != 128
            or head_dim != 128
            or scale_fmt != "ue8m0"
            or use_fp4_cache
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
        ):
            raise NotImplementedError(
                "EMPTY sparse indexer supports Hopper FP8 128-dim TP-only routing"
            )
        if is_dsv41:
            if topk_tokens != 512 or not skip_k_cache_insert or compress_ratio < 1:
                raise NotImplementedError(
                    "DSV4.1 EMPTY indexer requires topk=512 and separately inserted FP8 keys"
                )
            # Candidate selection/masking remains in the upstream forward,
            # after dense FP8 logits. It does not require sparse DeepGEMM.
            if candidate_blocks is not None and candidate_block_size <= 0:
                raise ValueError(
                    "DSV4.1 candidate blocks require a positive block size"
                )
        elif (
            topk_tokens not in (512, 1024, 2048)
            or compress_ratio != 1
            or candidate_blocks is not None
            or candidate_block_size
            or candidate_write
        ):
            raise NotImplementedError(
                "HY V4 EMPTY indexer supports uncompressed FP8 routing without candidates"
            )
        CustomOp.__init__(self)
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.compress_ratio = compress_ratio
        self.candidate_blocks = candidate_blocks
        self.candidate_block_size = candidate_block_size
        self.candidate_write = candidate_write
        self.dense_mha_metadata_layer_name = ""
        self.topk_backend = config.kernel_config.sparse_indexer_topk_backend
        self._parallel_config = parallel
        self.dcp_world_size = 1
        self.dcp_rank = 0
        self.use_pcp = False
        self.pcp_shard_decode_requests = parallel.pcp_shard_decode_requests
        self._cp_kv_cache_interleave_size = None
        if config.kernel_config.enable_jit_warmup:
            from vllm.v1.attention.ops.common import (
                _PACK_SEQ_TRITON_KERNEL,
                _UNPACK_SEQ_TRITON_KERNEL,
            )

            _PACK_SEQ_TRITON_KERNEL.register_warmup(
                dtype=current_platform.fp8_dtype(), pad_value=-float("inf")
            )
            _UNPACK_SEQ_TRITON_KERNEL.register_warmup()


def register_nvidia_sparse_indexer() -> bool:
    """Select the shared provider without replacing native or user providers."""
    if has_native_vllm_kernels() or not current_platform.is_cuda():
        return False
    existing = op_registry_oot.get("SparseAttnIndexer")
    if existing is not None:
        return existing is NvidiaSparseAttnIndexer
    CustomOp.register_oot(name="SparseAttnIndexer")(NvidiaSparseAttnIndexer)
    NvidiaSparseAttnIndexer.name = SparseAttnIndexer.name
    return True
