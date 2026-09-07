# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlagGems-backed sparse MLA for out-of-tree accelerators.

The metadata and cache contract match vLLM 0.24's sparse MLA backend.  Kernel
selection is backend-neutral: use FlagGems when its sparse MLA/cache writer is
available, otherwise retain a slow PyTorch correctness implementation.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import re
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
    SparseMLAAttentionImpl,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_fl.kernels.glm5_next.provider import (
    get_glm5_provider,
    use_nvidia_reference,
)

logger = init_logger(__name__)


def _flagtree_version_tuple() -> tuple[int, ...] | None:
    """Return the numeric FlagTree release tuple when package metadata exists."""
    try:
        version = importlib.metadata.version("flagtree")
    except importlib.metadata.PackageNotFoundError:
        return None
    numbers = re.match(r"^(\d+(?:\.\d+)*)", version)
    if numbers is None:
        return None
    return tuple(int(part) for part in numbers.group(1).split("."))


def _configure_sparse_mla_tle(loaded: Any) -> None:
    """Select the known-good FlagGems sparse-MLA implementation.

    FlagTree 0.6.1 exposes the complete TLE API, but its select-encodings pass
    rejects the FlagGems sparse-MLA kernel at the scalar ``local_ptr`` load.
    Keep using FlagGems' non-TLE Triton implementation for released versions
    through 0.6.1.  The override is intentionally plugin-scoped so an upstream
    fix can be qualified without changing the process-wide FlagGems setting.
    """
    if not getattr(loaded, "HAS_TLE_FLASHMLA_SPARSE", False):
        return

    mode = os.getenv("VLLM_FL_GLM5_SPARSE_MLA_TLE", "auto").strip().lower()
    if mode not in {"auto", "0", "1", "false", "true", "off", "on"}:
        raise ValueError(
            "VLLM_FL_GLM5_SPARSE_MLA_TLE must be auto, on/1/true, or "
            f"off/0/false; got {mode!r}"
        )
    if mode in {"1", "true", "on"}:
        return

    tle = getattr(loaded, "tle", None)
    tle_gpu = getattr(tle, "gpu", None)
    required_gpu_apis = ("alloc", "copy", "local_ptr", "warp_specialize")
    missing_api = not hasattr(tle, "pipe") or any(
        not hasattr(tle_gpu, api) for api in required_gpu_apis
    )
    flagtree_version = _flagtree_version_tuple()
    known_bad_release = flagtree_version is not None and flagtree_version <= (0, 6, 1)
    explicitly_disabled = mode in {"0", "false", "off"}
    if explicitly_disabled or missing_api or known_bad_release:
        loaded.HAS_TLE_FLASHMLA_SPARSE = False
        if explicitly_disabled:
            reason = "disabled by VLLM_FL_GLM5_SPARSE_MLA_TLE"
        elif missing_api:
            reason = "required FlagTree pipe/GPU APIs are unavailable"
        else:
            reason = (
                "FlagTree <= 0.6.1 fails TritonTleSelectEncodings for the "
                "FlagGems sparse-MLA kernel"
            )
        logger.warning_once(
            "FlagGems sparse MLA TLE path is %s; using its non-TLE Triton kernel",
            reason,
        )


def _flag_op(module: str, name: str):
    try:
        loaded = importlib.import_module(f"flag_gems.fused.{module}")
        if module == "flashmla_sparse":
            _configure_sparse_mla_tle(loaded)
        return getattr(loaded, name)
    except (ImportError, AttributeError, OSError):
        return None


@dataclass
class FlagGemsSparseMLAMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int = 64
    topk_tokens: int = 2048


class FlagGemsSparseMLAMetadataBuilder(
    AttentionMetadataBuilder[FlagGemsSparseMLAMetadata]
):
    # H100 A/B explicitly exercises the FlagGems kernels under graph capture.
    # Real non-NVIDIA bring-up remains conservative because a missing per-op
    # kernel can fall back to code containing host scalar reads.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
        if current_platform.is_cuda() and not use_nvidia_reference()
        else AttentionCGSupport.NEVER
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.req_id_per_token_buffer = torch.empty(
            max_tokens, dtype=torch.int32, device=device
        )
        self.topk_tokens = vllm_config.model_config.hf_config.index_topk

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlagGemsSparseMLAMetadata:
        del common_prefix_len, fast_build
        metadata = common_attn_metadata
        starts = np.asarray(metadata.query_start_loc_cpu, dtype=np.int32)
        segment_lengths = np.diff(starts)
        request_ids = np.repeat(
            np.arange(segment_lengths.shape[0], dtype=np.int32), segment_lengths
        )
        self.req_id_per_token_buffer.fill_(0)
        if request_ids.size:
            request_ids_tensor = torch.from_numpy(request_ids)
            self.req_id_per_token_buffer[: request_ids.size].copy_(
                request_ids_tensor, non_blocking=True
            )
        return FlagGemsSparseMLAMetadata(
            num_reqs=metadata.num_reqs,
            max_query_len=metadata.max_query_len,
            max_seq_len=metadata.max_seq_len,
            num_actual_tokens=metadata.num_actual_tokens,
            query_start_loc=metadata.query_start_loc,
            slot_mapping=metadata.slot_mapping,
            block_table=metadata.block_table_tensor,
            req_id_per_token=self.req_id_per_token_buffer[: metadata.num_actual_tokens],
            block_size=self.kv_cache_spec.block_size,
            topk_tokens=self.topk_tokens,
        )


class FlagGemsSparseMLABackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]
    # vLLM 0.24 resolves cudagraph support through the metadata builder, but
    # newer runners also inspect the backend class. Keep both declarations in
    # sync so the portable GLM5 path can participate in piecewise capture.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
        if current_platform.is_cuda() and not use_nvidia_reference()
        else AttentionCGSupport.NEVER
    )

    @staticmethod
    def get_name() -> str:
        return "FLAGGEMS_MLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type[FlagGemsSparseMLAMetadataBuilder]:
        return FlagGemsSparseMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[FlagGemsSparseMLAImpl]:
        return FlagGemsSparseMLAImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # GLM5-Next is 512-NoPE; DeepSeek-V3.2 is 512-NoPE + 64-RoPE.
        return [512, 576]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        del capability
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        if num_kv_heads != 1:
            raise ValueError("Sparse MLA requires one latent KV head")
        return (num_blocks, block_size, head_size)


def _convert_request_to_physical_indices(
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_indices_i64 = token_indices.to(torch.int64)
    request_ids_i64 = request_ids.to(torch.int64).reshape(-1, 1)
    block_ids = torch.div(token_indices_i64, block_size, rounding_mode="floor")
    valid = (token_indices_i64 >= 0) & (block_ids < block_table.shape[1])
    safe_blocks = block_ids.clamp(min=0, max=block_table.shape[1] - 1)
    rows = request_ids_i64.expand_as(safe_blocks)
    physical_blocks = block_table[rows, safe_blocks].to(torch.int64)
    valid = valid & (physical_blocks >= 0)
    physical = physical_blocks * block_size + torch.remainder(
        token_indices_i64, block_size
    )
    valid_lengths = valid.sum(dim=-1).to(torch.int32)

    # ``flash_mla_sparse_fwd`` consumes only ``indices[:topk_length]``.  A
    # missing logical block can create an invalid entry in the middle of the
    # indexer's result, so merely replacing invalid entries with -1 would both
    # expose -1 to the kernel and drop later valid tokens.  Compact valid
    # physical slots to a contiguous prefix, matching vLLM's reference
    # ``COMPACT_TO_FRONT`` conversion contract.  The extra column is a sink for
    # every invalid entry and keeps this path free of host synchronization.
    width = physical.shape[-1]
    compact_dst = valid.to(torch.int64).cumsum(dim=-1) - 1
    compact_dst = torch.where(valid, compact_dst, width)
    compact = torch.full(
        (*physical.shape[:-1], width + 1),
        -1,
        dtype=physical.dtype,
        device=physical.device,
    )
    compact.scatter_(dim=-1, index=compact_dst, src=physical)
    return compact[..., :width].to(torch.int32), valid_lengths


def _sparse_mla_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    value_dim: int,
    valid_lengths: torch.Tensor,
) -> torch.Tensor:
    rows, heads, _ = q.shape
    output = torch.zeros(rows, heads, value_dim, dtype=q.dtype, device=q.device)
    for row in range(rows):
        length = int(valid_lengths[row].item())
        if length == 0:
            continue
        selected = indices[row, :length].to(torch.int64)
        selected_kv = kv.index_select(0, selected).squeeze(1).float()
        scores = torch.einsum("hd,kd->hk", q[row].float(), selected_kv) * scale
        probabilities = torch.softmax(scores, dim=-1)
        output[row] = torch.einsum(
            "hk,kv->hv", probabilities, selected_kv[:, :value_dim]
        ).to(output.dtype)
    return output


class FlagGemsSparseMLAImpl(SparseMLAAttentionImpl[FlagGemsSparseMLAMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: Any | None = None,
        **mla_args,
    ) -> None:
        del (
            alibi_slopes,
            sliding_window,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
        )
        if kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "FlagGems sparse MLA portable path currently requires BF16 KV cache"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.softmax_scale = float(scale)
        self.kv_lora_rank = int(mla_args["kv_lora_rank"])
        self.topk_indices_buffer = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        fn = _flag_op("concat_and_cache_mla", "concat_and_cache_mla")
        if fn is not None:
            flag_cache_dtype = (
                "auto" if kv_cache_dtype == "bfloat16" else kv_cache_dtype
            )
            try:
                fn(
                    kv_c_normed,
                    k_pe.squeeze(1),
                    kv_cache,
                    slot_mapping.flatten(),
                    kv_cache_dtype=flag_cache_dtype,
                    scale=k_scale,
                )
                return
            except (NotImplementedError, RuntimeError) as exc:
                logger.warning(
                    "FlagGems MLA cache writer rejected this workload; using "
                    "the PyTorch correctness fallback: %s",
                    exc,
                )
        if kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError("Portable MLA cache write only supports BF16")
        source = (
            kv_c_normed
            if k_pe.shape[-1] == 0
            else torch.cat((kv_c_normed, k_pe.squeeze(1)), dim=-1)
        )
        slots = slot_mapping.flatten().to(torch.int64)
        valid = slots >= 0
        cache_flat = kv_cache.view(-1, kv_cache.shape[-1])
        cache_flat[slots[valid]] = source[valid]

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlagGemsSparseMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del layer
        if isinstance(q, tuple):
            q_nope, q_pe = q
            q = q_nope if q_pe.shape[-1] == 0 else torch.cat((q_nope, q_pe), dim=-1)
        num_tokens = q.shape[0]
        if self.topk_indices_buffer is None:
            raise RuntimeError("Sparse MLA requires the indexer's top-k buffer")
        request_topk = self.topk_indices_buffer[:num_tokens]
        physical_topk, valid_lengths = _convert_request_to_physical_indices(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            request_topk,
            attn_metadata.block_size,
        )
        cache = kv_c_and_k_pe_cache.contiguous().view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        actual_heads = q.shape[1]
        padded_heads = 64 if actual_heads <= 64 else 128
        if actual_heads not in (64, 128):
            q_padded = q.new_zeros(num_tokens, padded_heads, q.shape[-1])
            q_padded[:, :actual_heads].copy_(q)
        else:
            q_padded = q

        sparse_fn = _flag_op("flashmla_sparse", "flash_mla_sparse_fwd")
        if sparse_fn is not None:
            try:
                output = sparse_fn(
                    q_padded.contiguous(),
                    cache,
                    physical_topk.unsqueeze(1).contiguous(),
                    self.softmax_scale,
                    d_v=self.kv_lora_rank,
                    topk_length=valid_lengths.contiguous(),
                )[0]
            except (NotImplementedError, RuntimeError) as exc:
                logger.warning(
                    "FlagGems sparse MLA rejected this workload; using the "
                    "PyTorch correctness fallback: %s",
                    exc,
                )
                output = _sparse_mla_torch(
                    q_padded,
                    cache,
                    physical_topk,
                    self.softmax_scale,
                    self.kv_lora_rank,
                    valid_lengths,
                )
        else:
            output = _sparse_mla_torch(
                q_padded,
                cache,
                physical_topk,
                self.softmax_scale,
                self.kv_lora_rank,
                valid_lengths,
            )
        return output[:, :actual_heads], None


__all__ = [
    "FlagGemsSparseMLABackend",
    "FlagGemsSparseMLAImpl",
    "FlagGemsSparseMLAMetadata",
    "FlagGemsSparseMLAMetadataBuilder",
]
