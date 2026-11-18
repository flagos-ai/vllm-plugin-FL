# Copyright 2026 FlagOS Contributors
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
#
# The online-softmax algorithm is adapted from the Apache-2.0 vLLM Kimi K3
# AttnRes implementation, which includes MIT-licensed work from the
# flash-linear-attention project (Songlin Yang, Yu Zhang, and Zhiyuan Li).

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import has_triton_tle, libentry, libtuner
from flaggems_vllm.utils.device_info import get_device_capability, get_sm_count

from .configs import post_configs

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        _HAS_TLE_LOAD = True
    except ImportError:
        tle = None
        _HAS_TLE_LOAD = False
else:
    tle = None
    _HAS_TLE_LOAD = False

_MAX_BLOCKS = 8
_PERSISTENT_CTA_COUNT_PER_SM = 2
_MIN_TOKENS_FOR_PERSISTENT_LOOP = 256
_MIN_BLOCKS_FOR_PERSISTENT_LOOP = 3
_MAX_BLOCKS_FOR_WIDE_SOURCE_TILE = 3
_LARGE_TOKEN_COUNT_THRESHOLD = 4096


def _token_count_bucket(num_tokens: int) -> int:
    if num_tokens < _MIN_TOKENS_FOR_PERSISTENT_LOOP:
        return 0
    if num_tokens < _LARGE_TOKEN_COUNT_THRESHOLD:
        return 1
    return 2


class _FixedLaunchConfig(NamedTuple):
    use_persistent_token_loop: bool
    use_compact_source_reduction: bool
    source_tile_size: int
    num_warps: int
    num_stages: int
    enable_pdl: bool


def _select_fixed_launch_config(
    num_tokens: int,
    num_blocks: int,
) -> _FixedLaunchConfig:
    """Select a fixed launch plan for cases that update operator state."""
    if (
        num_tokens < _MIN_TOKENS_FOR_PERSISTENT_LOOP
        and 1 < num_blocks <= _MAX_BLOCKS_FOR_WIDE_SOURCE_TILE
    ):
        return _FixedLaunchConfig(
            use_persistent_token_loop=False,
            use_compact_source_reduction=True,
            source_tile_size=4,
            num_warps=8,
            num_stages=2,
            enable_pdl=True,
        )

    use_persistent_token_loop = (
        num_tokens >= _MIN_TOKENS_FOR_PERSISTENT_LOOP
        and num_blocks >= _MIN_BLOCKS_FOR_PERSISTENT_LOOP
    )
    if use_persistent_token_loop:
        source_tile_size, num_warps, num_stages = 1, 4, 1
    elif num_blocks <= 1:
        source_tile_size, num_warps, num_stages = 1, 4, 2
    elif num_blocks <= 3:
        source_tile_size, num_warps, num_stages = 4, 4, 2
    elif num_blocks < _MAX_BLOCKS:
        source_tile_size, num_warps, num_stages = 8, 8, 2
    else:
        source_tile_size, num_warps, num_stages = 4, 8, 2
    return _FixedLaunchConfig(
        use_persistent_token_loop=use_persistent_token_loop,
        use_compact_source_reduction=False,
        source_tile_size=source_tile_size,
        num_warps=num_warps,
        num_stages=num_stages,
        enable_pdl=False,
    )


def _prune_post_configs(configs, named_args, **kwargs):
    del named_args
    if not _HAS_TLE_LOAD:
        configs = [
            config
            for config in configs
            if not config.kwargs.get("USE_TLE_ASYNC_LOAD", False)
        ]
    token_count_bucket = kwargs["TOKEN_COUNT_BUCKET"]
    num_blocks = kwargs["NUM_BLOCKS"]
    compact_shape = (
        token_count_bucket == 0 and 1 < num_blocks <= _MAX_BLOCKS_FOR_WIDE_SOURCE_TILE
    )
    if not compact_shape:
        configs = [
            config
            for config in configs
            if not config.kwargs["USE_COMPACT_SOURCE_REDUCTION"]
        ]
    if (
        token_count_bucket == 0
        and _MAX_BLOCKS_FOR_WIDE_SOURCE_TILE < num_blocks < _MAX_BLOCKS
    ):
        return [
            config
            for config in configs
            if not config.kwargs["USE_COMPACT_SOURCE_REDUCTION"]
            and not config.kwargs["MERGE_PREFIX_INTO_SOURCE_LOOP"]
            and not config.kwargs["USE_TLE_ASYNC_LOAD"]
            and config.kwargs["SOURCE_TILE_SIZE"] == 8
            and config.num_warps == 8
            and config.num_stages == 2
        ]
    # Two-source tiles have a stable large-token layout: even block counts reduce
    # the prefix separately; odd block counts merge it into the source loop so
    # every tile contains two valid sources.
    if token_count_bucket == 2 and num_blocks >= _MIN_BLOCKS_FOR_PERSISTENT_LOOP:
        merge_prefix_into_source_loop = num_blocks % 2 == 1
        return [
            config
            for config in configs
            if not config.kwargs["USE_COMPACT_SOURCE_REDUCTION"]
            and config.kwargs["MERGE_PREFIX_INTO_SOURCE_LOOP"]
            == merge_prefix_into_source_loop
            and not config.kwargs["USE_TLE_ASYNC_LOAD"]
            and config.kwargs["SOURCE_TILE_SIZE"] == 2
            and not config.kwargs["launch_pdl"]
            and config.num_warps == 4
            and config.num_stages == 2
        ]
    if token_count_bucket == 0 and num_blocks < _MAX_BLOCKS:
        configs = [
            config
            for config in configs
            if config.kwargs["USE_COMPACT_SOURCE_REDUCTION"]
            or (
                config.kwargs["MERGE_PREFIX_INTO_SOURCE_LOOP"] and config.num_stages > 1
            )
        ]
    return configs


if _HAS_TLE_LOAD:

    @triton.jit
    def _load_source_vectors(
        value_ptrs,
        value_mask,
        USE_TLE_ASYNC_LOAD: tl.constexpr,
    ):
        return tle.load(
            value_ptrs,
            mask=value_mask,
            other=0.0,
            eviction_policy="evict_first",
            is_async=USE_TLE_ASYNC_LOAD,
        )

else:

    @triton.jit
    def _load_source_vectors(
        value_ptrs,
        value_mask,
        USE_TLE_ASYNC_LOAD: tl.constexpr,
    ):
        return tl.load(
            value_ptrs,
            mask=value_mask,
            other=0.0,
            eviction_policy="evict_first",
        )


# Step 1: update prefix state and optionally copy it into a residual block.
@triton.jit
def _update_prefix_and_block(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    token_idx,
    hidden_offsets,
    hidden_mask,
    stride_prefix_m,
    stride_delta_m,
    stride_block_m,
    stride_block_r,
    BLOCK_WRITE_INDEX: tl.constexpr,
    ADD_DELTA_TO_PREFIX: tl.constexpr,
    STORE_PREFIX_TO_BLOCK: tl.constexpr,
    USE_ALIGNED_MEMORY_HINTS: tl.constexpr,
):
    token_idx = token_idx.to(tl.int64)
    prefix_value_ptrs = prefix_ptr + token_idx * stride_prefix_m + hidden_offsets
    if USE_ALIGNED_MEMORY_HINTS:
        prefix_value_ptrs = tl.multiple_of(prefix_value_ptrs, 16)
    updated_prefix = tl.load(
        prefix_value_ptrs,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)

    if ADD_DELTA_TO_PREFIX:
        delta_value_ptrs = delta_ptr + token_idx * stride_delta_m + hidden_offsets
        if USE_ALIGNED_MEMORY_HINTS:
            delta_value_ptrs = tl.multiple_of(delta_value_ptrs, 16)
        delta = tl.load(
            delta_value_ptrs,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        updated_prefix = (updated_prefix + delta).to(prefix_ptr.dtype.element_ty)
        updated_prefix = updated_prefix.to(tl.float32)
        tl.store(prefix_value_ptrs, updated_prefix, mask=hidden_mask)

    if STORE_PREFIX_TO_BLOCK:
        block_write_ptrs = (
            blocks_ptr
            + token_idx * stride_block_m
            + BLOCK_WRITE_INDEX * stride_block_r
            + hidden_offsets
        )
        if USE_ALIGNED_MEMORY_HINTS:
            block_write_ptrs = tl.multiple_of(block_write_ptrs, 16)
        tl.store(block_write_ptrs, updated_prefix, mask=hidden_mask)

    return updated_prefix


# Step 2: compute the softmax-weighted sum of prefix and residual blocks.
@triton.jit
def _compute_attention_weighted_sum(
    prefix_ptr,
    blocks_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    preloaded_prefix,
    preloaded_norm_qk_weight,
    token_idx,
    hidden_offsets,
    hidden_mask,
    stride_prefix_m,
    stride_block_m,
    stride_block_r,
    eps,
    NUM_BLOCKS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    NEEDS_SOURCE_WRITE_BARRIER: tl.constexpr,
    USE_COMPACT_SOURCE_REDUCTION: tl.constexpr,
    MERGE_PREFIX_INTO_SOURCE_LOOP: tl.constexpr,
    USE_PRELOADED_PREFIX: tl.constexpr,
    USE_PRELOADED_NORM_QK_WEIGHT: tl.constexpr,
    USE_TLE_ASYNC_LOAD: tl.constexpr,
    USE_ALIGNED_MEMORY_HINTS: tl.constexpr,
    SOURCE_TILE_SIZE: tl.constexpr,
    HIDDEN_TILE_SIZE: tl.constexpr,
):
    token_idx = token_idx.to(tl.int64)

    if USE_PRELOADED_PREFIX:
        prefix = preloaded_prefix
    elif not USE_COMPACT_SOURCE_REDUCTION and not MERGE_PREFIX_INTO_SOURCE_LOOP:
        prefix_value_ptrs = prefix_ptr + token_idx * stride_prefix_m + hidden_offsets
        if USE_ALIGNED_MEMORY_HINTS:
            prefix_value_ptrs = tl.multiple_of(prefix_value_ptrs, 16)
        prefix = tl.load(
            prefix_value_ptrs,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        prefix = tl.zeros((HIDDEN_TILE_SIZE,), tl.float32)

    if USE_PRELOADED_NORM_QK_WEIGHT:
        norm_qk_weight = preloaded_norm_qk_weight
    elif NUM_BLOCKS > 0 or MERGE_PREFIX_INTO_SOURCE_LOOP:
        norm_weight_ptrs = norm_weight_ptr + hidden_offsets
        qk_weight_ptrs = qk_weight_ptr + hidden_offsets
        if USE_ALIGNED_MEMORY_HINTS:
            norm_weight_ptrs = tl.multiple_of(norm_weight_ptrs, 16)
            qk_weight_ptrs = tl.multiple_of(qk_weight_ptrs, 16)
        norm_weight = tl.load(
            norm_weight_ptrs,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        qk_weight = tl.load(
            qk_weight_ptrs,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        norm_qk_weight = norm_weight * qk_weight
    else:
        norm_qk_weight = tl.zeros((HIDDEN_TILE_SIZE,), tl.float32)

    if USE_COMPACT_SOURCE_REDUCTION:
        if NEEDS_SOURCE_WRITE_BARRIER:
            tl.debug_barrier()

        running_max = tl.full((), -float("inf"), tl.float32)
        softmax_denominator = tl.zeros((), tl.float32)
        weighted_sum_numerator = tl.zeros((HIDDEN_TILE_SIZE,), tl.float32)
        source_count: tl.constexpr = NUM_BLOCKS + 1
        for source_tile_index in range(tl.cdiv(source_count, SOURCE_TILE_SIZE)):
            source_indices = source_tile_index * SOURCE_TILE_SIZE + tl.arange(
                0, SOURCE_TILE_SIZE
            )
            valid_sources = source_indices < source_count
            prefix_lane = source_indices == NUM_BLOCKS
            block_value_ptrs = (
                blocks_ptr
                + token_idx * stride_block_m
                + source_indices[:, None] * stride_block_r
                + hidden_offsets[None, :]
            )
            prefix_value_ptrs = (
                prefix_ptr
                + token_idx * stride_prefix_m
                + source_indices[:, None] * 0
                + hidden_offsets[None, :]
            )
            value_ptrs = tl.where(
                prefix_lane[:, None],
                prefix_value_ptrs,
                block_value_ptrs,
            )
            if USE_ALIGNED_MEMORY_HINTS:
                value_ptrs = tl.multiple_of(value_ptrs, (1, 16))
            source_values = tl.load(
                value_ptrs,
                mask=valid_sources[:, None] & hidden_mask[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

            reciprocal_rms = tl.rsqrt(
                tl.sum(source_values * source_values, axis=1) * (1.0 / HIDDEN_SIZE)
                + eps
            )
            logits = (
                tl.sum(source_values * norm_qk_weight[None, :], axis=1) * reciprocal_rms
            )
            masked_logits = tl.where(valid_sources, logits, -float("inf"))
            tile_max = tl.max(masked_logits, axis=0)
            new_running_max = tl.maximum(running_max, tile_max)
            previous_scale = tl.exp(running_max - new_running_max)
            source_weights = tl.exp(masked_logits - new_running_max)
            softmax_denominator = softmax_denominator * previous_scale + tl.sum(
                source_weights, axis=0
            )
            weighted_sum_numerator = weighted_sum_numerator * previous_scale + tl.sum(
                source_weights[:, None] * source_values, axis=0
            )
            running_max = new_running_max
    elif not MERGE_PREFIX_INTO_SOURCE_LOOP and NUM_BLOCKS == 0:
        weighted_sum_numerator = prefix
        softmax_denominator = tl.full((), 1.0, tl.float32)
    elif MERGE_PREFIX_INTO_SOURCE_LOOP:
        source_count = NUM_BLOCKS + 1
        running_max = tl.full((), -float("inf"), tl.float32)
        softmax_denominator = tl.zeros((), tl.float32)
        weighted_sum_numerator = tl.zeros((HIDDEN_TILE_SIZE,), tl.float32)

        for source_tile_index in range(tl.cdiv(source_count, SOURCE_TILE_SIZE)):
            source_indices = source_tile_index * SOURCE_TILE_SIZE + tl.arange(
                0, SOURCE_TILE_SIZE
            )
            valid_sources = source_indices < source_count
            block_source_ptrs = (
                blocks_ptr
                + token_idx * stride_block_m
                + source_indices * stride_block_r
            )
            prefix_source_ptrs = (
                prefix_ptr + token_idx * stride_prefix_m + source_indices * 0
            )
            source_base_ptrs = tl.where(
                source_indices == NUM_BLOCKS,
                prefix_source_ptrs,
                block_source_ptrs,
            )
            if USE_ALIGNED_MEMORY_HINTS:
                source_base_ptrs = tl.multiple_of(source_base_ptrs, 16)
            value_ptrs = source_base_ptrs[:, None] + hidden_offsets[None, :]
            if USE_ALIGNED_MEMORY_HINTS:
                value_ptrs = tl.multiple_of(value_ptrs, (1, 16))
            source_values = _load_source_vectors(
                value_ptrs,
                valid_sources[:, None] & hidden_mask[None, :],
                USE_TLE_ASYNC_LOAD,
            ).to(tl.float32)

            reciprocal_rms = tl.rsqrt(
                tl.sum(source_values * source_values, axis=1) * (1.0 / HIDDEN_SIZE)
                + eps
            )
            logits = (
                tl.sum(source_values * norm_qk_weight[None, :], axis=1) * reciprocal_rms
            )
            masked_logits = tl.where(
                valid_sources,
                logits,
                -float("inf"),
            )
            tile_max = tl.max(masked_logits, axis=0)
            if source_tile_index == 0:
                running_max = tile_max
                source_weights = tl.exp(masked_logits - running_max)
                softmax_denominator = tl.sum(source_weights, axis=0)
                weighted_sum_numerator = tl.sum(
                    source_weights[:, None] * source_values,
                    axis=0,
                )
            else:
                new_running_max = tl.maximum(running_max, tile_max)
                previous_scale = tl.exp(running_max - new_running_max)
                source_weights = tl.exp(masked_logits - new_running_max)
                softmax_denominator = softmax_denominator * previous_scale + tl.sum(
                    source_weights, axis=0
                )
                weighted_sum_numerator = (
                    weighted_sum_numerator * previous_scale
                    + tl.sum(
                        source_weights[:, None] * source_values,
                        axis=0,
                    )
                )
                running_max = new_running_max
    else:
        if NEEDS_SOURCE_WRITE_BARRIER:
            tl.debug_barrier()

        prefix_fits_in_source_tile: tl.constexpr = NUM_BLOCKS % SOURCE_TILE_SIZE != 0
        if prefix_fits_in_source_tile:
            running_max = tl.full((), -float("inf"), tl.float32)
            softmax_denominator = tl.zeros((), tl.float32)
            weighted_sum_numerator = tl.zeros((HIDDEN_TILE_SIZE,), tl.float32)
        else:
            prefix_reciprocal_rms = tl.rsqrt(
                tl.sum(prefix * prefix, axis=0) * (1.0 / HIDDEN_SIZE) + eps
            )
            running_max = (
                tl.sum(prefix * norm_qk_weight, axis=0) * prefix_reciprocal_rms
            )
            softmax_denominator = tl.full((), 1.0, tl.float32)
            weighted_sum_numerator = prefix

        for source_tile_index in range(tl.cdiv(NUM_BLOCKS, SOURCE_TILE_SIZE)):
            source_indices = source_tile_index * SOURCE_TILE_SIZE + tl.arange(
                0, SOURCE_TILE_SIZE
            )
            valid_blocks = source_indices < NUM_BLOCKS
            prefix_lane = source_indices == NUM_BLOCKS
            valid_sources = valid_blocks
            if prefix_fits_in_source_tile:
                valid_sources |= prefix_lane

            block_value_ptrs = (
                blocks_ptr
                + token_idx * stride_block_m
                + source_indices[:, None] * stride_block_r
                + hidden_offsets[None, :]
            )
            if USE_ALIGNED_MEMORY_HINTS:
                block_value_ptrs = tl.multiple_of(block_value_ptrs, (1, 16))
            source_values = _load_source_vectors(
                block_value_ptrs,
                valid_blocks[:, None] & hidden_mask[None, :],
                False,
            ).to(tl.float32)
            if prefix_fits_in_source_tile:
                source_values = tl.where(
                    prefix_lane[:, None],
                    prefix[None, :],
                    source_values,
                )

            reciprocal_rms = tl.rsqrt(
                tl.sum(source_values * source_values, axis=1) * (1.0 / HIDDEN_SIZE)
                + eps
            )
            logits = (
                tl.sum(source_values * norm_qk_weight[None, :], axis=1) * reciprocal_rms
            )
            masked_logits = tl.where(
                valid_sources,
                logits,
                -float("inf"),
            )
            tile_max = tl.max(masked_logits, axis=0)
            if prefix_fits_in_source_tile and source_tile_index == 0:
                running_max = tile_max
                source_weights = tl.exp(masked_logits - running_max)
                softmax_denominator = tl.sum(source_weights, axis=0)
                weighted_sum_numerator = tl.sum(
                    source_weights[:, None] * source_values,
                    axis=0,
                )
            else:
                new_running_max = tl.maximum(running_max, tile_max)
                previous_scale = tl.exp(running_max - new_running_max)
                source_weights = tl.exp(masked_logits - new_running_max)
                softmax_denominator = softmax_denominator * previous_scale + tl.sum(
                    source_weights, axis=0
                )
                weighted_sum_numerator = (
                    weighted_sum_numerator * previous_scale
                    + tl.sum(
                        source_weights[:, None] * source_values,
                        axis=0,
                    )
                )
                running_max = new_running_max

    return weighted_sum_numerator, softmax_denominator


# Step 3: apply the optional output RMS normalization and store the result.
@triton.jit
def _normalize_and_store_output(
    output_norm_weight_ptr,
    output_ptr,
    weighted_sum_numerator,
    softmax_denominator,
    token_idx,
    hidden_offsets,
    hidden_mask,
    stride_output_m,
    output_norm_eps,
    HIDDEN_SIZE: tl.constexpr,
    APPLY_OUTPUT_NORM: tl.constexpr,
    USE_ALIGNED_MEMORY_HINTS: tl.constexpr,
):
    if APPLY_OUTPUT_NORM:
        output_reciprocal_std = tl.rsqrt(
            tl.sum(weighted_sum_numerator * weighted_sum_numerator, axis=0)
            * (1.0 / HIDDEN_SIZE)
            + output_norm_eps * softmax_denominator * softmax_denominator
        )
        output_norm_weight_ptrs = output_norm_weight_ptr + hidden_offsets
        if USE_ALIGNED_MEMORY_HINTS:
            output_norm_weight_ptrs = tl.multiple_of(output_norm_weight_ptrs, 16)
        output_norm_weight = tl.load(
            output_norm_weight_ptrs,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        output = weighted_sum_numerator * output_reciprocal_std * output_norm_weight
    else:
        output = weighted_sum_numerator * (1.0 / softmax_denominator)

    output_value_ptrs = output_ptr + token_idx * stride_output_m + hidden_offsets
    if USE_ALIGNED_MEMORY_HINTS:
        output_value_ptrs = tl.multiple_of(output_value_ptrs, 16)
    tl.store(output_value_ptrs, output, mask=hidden_mask)


@triton.jit
def _execute_attn_res_steps(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    output_norm_weight_ptr,
    output_ptr,
    preloaded_norm_qk_weight,
    token_idx,
    hidden_offsets,
    hidden_mask,
    stride_prefix_m,
    stride_delta_m,
    stride_block_m,
    stride_block_r,
    stride_output_m,
    eps,
    output_norm_eps,
    NUM_BLOCKS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_WRITE_INDEX: tl.constexpr,
    ADD_DELTA_TO_PREFIX: tl.constexpr,
    STORE_PREFIX_TO_BLOCK: tl.constexpr,
    APPLY_OUTPUT_NORM: tl.constexpr,
    USE_PRELOADED_NORM_QK_WEIGHT: tl.constexpr,
    USE_COMPACT_SOURCE_REDUCTION: tl.constexpr,
    USE_ALIGNED_MEMORY_HINTS: tl.constexpr,
    SOURCE_TILE_SIZE: tl.constexpr,
    HIDDEN_TILE_SIZE: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    updated_prefix = _update_prefix_and_block(
        prefix_ptr,
        delta_ptr,
        blocks_ptr,
        token_idx,
        hidden_offsets,
        hidden_mask,
        stride_prefix_m,
        stride_delta_m,
        stride_block_m,
        stride_block_r,
        BLOCK_WRITE_INDEX,
        ADD_DELTA_TO_PREFIX,
        STORE_PREFIX_TO_BLOCK,
        USE_ALIGNED_MEMORY_HINTS,
    )
    weighted_sum_numerator, softmax_denominator = _compute_attention_weighted_sum(
        prefix_ptr,
        blocks_ptr,
        norm_weight_ptr,
        qk_weight_ptr,
        updated_prefix,
        preloaded_norm_qk_weight,
        token_idx,
        hidden_offsets,
        hidden_mask,
        stride_prefix_m,
        stride_block_m,
        stride_block_r,
        eps,
        NUM_BLOCKS,
        HIDDEN_SIZE,
        (
            (USE_COMPACT_SOURCE_REDUCTION and ADD_DELTA_TO_PREFIX)
            or (STORE_PREFIX_TO_BLOCK and BLOCK_WRITE_INDEX < NUM_BLOCKS)
        ),
        USE_COMPACT_SOURCE_REDUCTION,
        False,
        True,
        USE_PRELOADED_NORM_QK_WEIGHT,
        False,
        USE_ALIGNED_MEMORY_HINTS,
        SOURCE_TILE_SIZE,
        HIDDEN_TILE_SIZE,
    )

    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    _normalize_and_store_output(
        output_norm_weight_ptr,
        output_ptr,
        weighted_sum_numerator,
        softmax_denominator,
        token_idx,
        hidden_offsets,
        hidden_mask,
        stride_output_m,
        output_norm_eps,
        HIDDEN_SIZE,
        APPLY_OUTPUT_NORM,
        USE_ALIGNED_MEMORY_HINTS,
    )


@libentry()
@libtuner(
    configs=post_configs(),
    key=[
        "TOKEN_COUNT_BUCKET",
        "NUM_BLOCKS",
        "HIDDEN_SIZE",
        "APPLY_OUTPUT_NORM",
        "USE_ALIGNED_MEMORY_HINTS",
    ],
    prune_configs_by={"early_config_prune": _prune_post_configs},
    use_cuda_graph=True,
    rep=20,
)
@triton.jit(do_not_specialize=["eps", "output_norm_eps"])
def _attn_res_post_kernel(
    prefix_ptr,
    blocks_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    output_norm_weight_ptr,
    output_ptr,
    stride_prefix_m,
    stride_block_m,
    stride_block_r,
    stride_output_m,
    eps,
    output_norm_eps,
    TOKEN_COUNT_BUCKET: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    APPLY_OUTPUT_NORM: tl.constexpr,
    USE_COMPACT_SOURCE_REDUCTION: tl.constexpr,
    MERGE_PREFIX_INTO_SOURCE_LOOP: tl.constexpr,
    USE_TLE_ASYNC_LOAD: tl.constexpr,
    USE_ALIGNED_MEMORY_HINTS: tl.constexpr,
    SOURCE_TILE_SIZE: tl.constexpr,
    HIDDEN_TILE_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_offsets = tl.max_contiguous(
        tl.arange(0, HIDDEN_TILE_SIZE),
        HIDDEN_TILE_SIZE,
    )
    hidden_mask = hidden_offsets < HIDDEN_SIZE

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    unused_prefix = tl.zeros((HIDDEN_TILE_SIZE,), tl.float32)
    unused_norm_qk_weight = tl.zeros((HIDDEN_TILE_SIZE,), tl.float32)
    weighted_sum_numerator, softmax_denominator = _compute_attention_weighted_sum(
        prefix_ptr,
        blocks_ptr,
        norm_weight_ptr,
        qk_weight_ptr,
        unused_prefix,
        unused_norm_qk_weight,
        token_idx,
        hidden_offsets,
        hidden_mask,
        stride_prefix_m,
        stride_block_m,
        stride_block_r,
        eps,
        NUM_BLOCKS,
        HIDDEN_SIZE,
        False,
        USE_COMPACT_SOURCE_REDUCTION,
        MERGE_PREFIX_INTO_SOURCE_LOOP,
        False,
        False,
        USE_TLE_ASYNC_LOAD,
        USE_ALIGNED_MEMORY_HINTS,
        SOURCE_TILE_SIZE,
        HIDDEN_TILE_SIZE,
    )

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    _normalize_and_store_output(
        output_norm_weight_ptr,
        output_ptr,
        weighted_sum_numerator,
        softmax_denominator,
        token_idx,
        hidden_offsets,
        hidden_mask,
        stride_output_m,
        output_norm_eps,
        HIDDEN_SIZE,
        APPLY_OUTPUT_NORM,
        USE_ALIGNED_MEMORY_HINTS,
    )


@libentry()
@triton.jit(do_not_specialize=["num_tokens", "eps", "output_norm_eps"])
def _attn_res_fixed_config_kernel(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    output_norm_weight_ptr,
    output_ptr,
    stride_prefix_m,
    stride_delta_m,
    stride_block_m,
    stride_block_r,
    stride_output_m,
    num_tokens,
    eps,
    output_norm_eps,
    NUM_BLOCKS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_WRITE_INDEX: tl.constexpr,
    ADD_DELTA_TO_PREFIX: tl.constexpr,
    STORE_PREFIX_TO_BLOCK: tl.constexpr,
    APPLY_OUTPUT_NORM: tl.constexpr,
    USE_PERSISTENT_TOKEN_LOOP: tl.constexpr,
    USE_COMPACT_SOURCE_REDUCTION: tl.constexpr,
    USE_ALIGNED_MEMORY_HINTS: tl.constexpr,
    CTA_COUNT: tl.constexpr,
    SOURCE_TILE_SIZE: tl.constexpr,
    HIDDEN_TILE_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    cta_index = tl.program_id(0)
    hidden_offsets = tl.max_contiguous(
        tl.arange(0, HIDDEN_TILE_SIZE),
        HIDDEN_TILE_SIZE,
    )
    hidden_mask = hidden_offsets < HIDDEN_SIZE

    if NUM_BLOCKS == 0 or not USE_PERSISTENT_TOKEN_LOOP:
        preloaded_norm_qk_weight = tl.zeros((HIDDEN_TILE_SIZE,), tl.float32)
    else:
        norm_weight_ptrs = norm_weight_ptr + hidden_offsets
        qk_weight_ptrs = qk_weight_ptr + hidden_offsets
        if USE_ALIGNED_MEMORY_HINTS:
            norm_weight_ptrs = tl.multiple_of(norm_weight_ptrs, 16)
            qk_weight_ptrs = tl.multiple_of(qk_weight_ptrs, 16)
        norm_weight = tl.load(
            norm_weight_ptrs,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        qk_weight = tl.load(
            qk_weight_ptrs,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        preloaded_norm_qk_weight = norm_weight * qk_weight

    if USE_PERSISTENT_TOKEN_LOOP:
        for token_idx in range(cta_index, num_tokens, CTA_COUNT):
            _execute_attn_res_steps(
                prefix_ptr,
                delta_ptr,
                blocks_ptr,
                norm_weight_ptr,
                qk_weight_ptr,
                output_norm_weight_ptr,
                output_ptr,
                preloaded_norm_qk_weight,
                token_idx,
                hidden_offsets,
                hidden_mask,
                stride_prefix_m,
                stride_delta_m,
                stride_block_m,
                stride_block_r,
                stride_output_m,
                eps,
                output_norm_eps,
                NUM_BLOCKS,
                HIDDEN_SIZE,
                BLOCK_WRITE_INDEX,
                ADD_DELTA_TO_PREFIX,
                STORE_PREFIX_TO_BLOCK,
                APPLY_OUTPUT_NORM,
                True,
                USE_COMPACT_SOURCE_REDUCTION,
                USE_ALIGNED_MEMORY_HINTS,
                SOURCE_TILE_SIZE,
                HIDDEN_TILE_SIZE,
                launch_pdl,
            )
    else:
        if launch_pdl:
            tl.extra.cuda.gdc_wait()

        _execute_attn_res_steps(
            prefix_ptr,
            delta_ptr,
            blocks_ptr,
            norm_weight_ptr,
            qk_weight_ptr,
            output_norm_weight_ptr,
            output_ptr,
            preloaded_norm_qk_weight,
            cta_index,
            hidden_offsets,
            hidden_mask,
            stride_prefix_m,
            stride_delta_m,
            stride_block_m,
            stride_block_r,
            stride_output_m,
            eps,
            output_norm_eps,
            NUM_BLOCKS,
            HIDDEN_SIZE,
            BLOCK_WRITE_INDEX,
            ADD_DELTA_TO_PREFIX,
            STORE_PREFIX_TO_BLOCK,
            APPLY_OUTPUT_NORM,
            False,
            USE_COMPACT_SOURCE_REDUCTION,
            USE_ALIGNED_MEMORY_HINTS,
            SOURCE_TILE_SIZE,
            HIDDEN_TILE_SIZE,
            launch_pdl,
        )


def _validate_tensor(
    name: str,
    tensor: torch.Tensor,
    prefix: torch.Tensor,
) -> None:
    if tensor.device != prefix.device:
        raise ValueError(f"{name} must be on {prefix.device}, got {tensor.device}")
    if tensor.dtype != prefix.dtype:
        raise TypeError(f"{name} must have dtype {prefix.dtype}, got {tensor.dtype}")
    if tensor.stride(-1) != 1:
        raise ValueError(f"the last dimension of {name} must be contiguous")

    if 0 in tensor.shape:
        return
    inner_span = 1
    for size, stride in zip(reversed(tensor.shape), reversed(tensor.stride())):
        if stride < inner_span:
            raise ValueError(f"{name} must have a non-overlapping row-major layout")
        inner_span += (size - 1) * stride


def _memory_interval(tensor: torch.Tensor) -> tuple[int, int] | None:
    if 0 in tensor.shape:
        return None
    max_offset = sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )
    start = tensor.data_ptr()
    return start, start + (max_offset + 1) * tensor.element_size()


def _tensors_overlap(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    lhs_interval = _memory_interval(lhs)
    rhs_interval = _memory_interval(rhs)
    if lhs_interval is None or rhs_interval is None:
        return False
    return lhs_interval[0] < rhs_interval[1] and rhs_interval[0] < lhs_interval[1]


def _validate_mutation_aliases(
    tensors: dict[str, torch.Tensor],
    mutable_names: tuple[str, ...],
) -> None:
    checked_pairs = set()
    for mutable_name in mutable_names:
        mutable_tensor = tensors[mutable_name]
        for other_name, other_tensor in tensors.items():
            if other_name == mutable_name:
                continue
            pair = tuple(sorted((mutable_name, other_name)))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            if _tensors_overlap(mutable_tensor, other_tensor):
                raise ValueError(
                    f"{mutable_name} must not overlap {other_name} when mutated"
                )


def _validate_attn_res_inputs(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> None:
    required_tensors = {
        "prefix": prefix,
        "blocks": blocks,
        "norm_weight": norm_weight,
        "qk_weight": qk_weight,
    }
    optional_tensors = {
        "delta": delta,
        "output_norm_weight": output_norm_weight,
    }
    for name, tensor in required_tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    for name, tensor in optional_tensors.items():
        if tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None")

    if prefix.device.type != "cuda":
        raise NotImplementedError("attn_res currently supports CUDA Hopper only")
    if prefix.dtype != torch.bfloat16:
        raise NotImplementedError("attn_res currently supports bfloat16 only")
    if prefix.ndim != 2 or prefix.shape[1] == 0:
        raise ValueError(
            "prefix must have shape [tokens, hidden_size] with hidden_size > 0, "
            f"got {tuple(prefix.shape)}"
        )

    num_tokens = prefix.shape[0]
    hidden_size = prefix.shape[1]
    if blocks.shape != (num_tokens, _MAX_BLOCKS, hidden_size):
        raise ValueError(
            "blocks must have shape "
            f"[tokens, {_MAX_BLOCKS}, {hidden_size}], got {tuple(blocks.shape)}"
        )
    if delta is not None and delta.shape != prefix.shape:
        raise ValueError(
            f"delta must match prefix shape {tuple(prefix.shape)}, "
            f"got {tuple(delta.shape)}"
        )
    for name, weight in (
        ("norm_weight", norm_weight),
        ("qk_weight", qk_weight),
    ):
        if weight.shape != (hidden_size,):
            raise ValueError(
                f"{name} must have shape [{hidden_size}], got {tuple(weight.shape)}"
            )
    if output_norm_weight is not None and output_norm_weight.shape != (hidden_size,):
        raise ValueError(
            "output_norm_weight must have shape "
            f"[{hidden_size}], got {tuple(output_norm_weight.shape)}"
        )

    for name, tensor in required_tensors.items():
        _validate_tensor(name, tensor, prefix)
    for name, tensor in optional_tensors.items():
        if tensor is not None:
            _validate_tensor(name, tensor, prefix)

    all_tensors = {
        **required_tensors,
        **{
            name: tensor
            for name, tensor in optional_tensors.items()
            if tensor is not None
        },
    }
    for name, tensor in all_tensors.items():
        if torch.is_grad_enabled() and tensor.requires_grad:
            raise NotImplementedError(
                f"attn_res is forward-only; {name} must not require gradients"
            )

    if not isinstance(num_blocks, int) or isinstance(num_blocks, bool):
        raise TypeError("num_blocks must be an int")
    if not 0 <= num_blocks <= _MAX_BLOCKS:
        raise ValueError(f"num_blocks must be in [0, {_MAX_BLOCKS}]")
    if not isinstance(block_write_idx, int) or isinstance(block_write_idx, bool):
        raise TypeError("block_write_idx must be an int")
    if not -1 <= block_write_idx < _MAX_BLOCKS:
        raise ValueError(f"block_write_idx must be in [-1, {_MAX_BLOCKS - 1}]")

    mutable_names = []
    if delta is not None:
        mutable_names.append("prefix")
    if block_write_idx >= 0:
        mutable_names.append("blocks")
    _validate_mutation_aliases(all_tensors, tuple(mutable_names))

    for name, value in (("eps", eps), ("output_norm_eps", output_norm_eps)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real scalar")


def attn_res(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> torch.Tensor:
    """Run the forward-only Kimi-K3 attention-residual operator on Hopper.

    ``prefix`` is updated in place when ``delta`` is present. ``blocks`` is
    updated in place when ``block_write_idx`` is non-negative. The attended
    sources are ``blocks[:, :num_blocks]`` followed by the updated prefix.
    Inputs must use non-overlapping row-major layouts with a contiguous hidden
    dimension. Mutated inputs must not overlap any other input tensor.

    Post does not mutate inputs and may therefore autotune safely. Cases that
    update prefix or blocks use fixed per-token or persistent configurations.
    """
    _validate_attn_res_inputs(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        eps,
        output_norm_eps,
    )

    output = prefix.new_empty(prefix.shape)
    with torch_device_fn.device(prefix.device):
        if get_device_capability() != (9, 0):
            raise NotImplementedError("attn_res currently supports SM90 only")
        num_tokens = prefix.shape[0]
        if num_tokens == 0:
            return output
        hidden_size = prefix.shape[1]
        hidden_tile_size = triton.next_power_of_2(hidden_size)

        output_norm_weight_ptr = (
            norm_weight if output_norm_weight is None else output_norm_weight
        )
        aligned_tensors = [
            prefix,
            blocks,
            norm_weight,
            qk_weight,
            output_norm_weight_ptr,
            output,
        ]
        if delta is not None:
            aligned_tensors.append(delta)
        rows_are_16b_aligned = (
            all(tensor.data_ptr() % 16 == 0 for tensor in aligned_tensors)
            and prefix.stride(0) % 16 == 0
            and blocks.stride(0) % 16 == 0
            and blocks.stride(1) % 16 == 0
            and output.stride(0) % 16 == 0
            and (delta is None or delta.stride(0) % 16 == 0)
        )
        is_post = delta is None and block_write_idx == -1
        if is_post:
            _attn_res_post_kernel[(num_tokens,)](
                prefix,
                blocks,
                norm_weight,
                qk_weight,
                output_norm_weight_ptr,
                output,
                prefix.stride(0),
                blocks.stride(0),
                blocks.stride(1),
                output.stride(0),
                eps,
                output_norm_eps,
                TOKEN_COUNT_BUCKET=_token_count_bucket(num_tokens),
                NUM_BLOCKS=num_blocks,
                HIDDEN_SIZE=hidden_size,
                APPLY_OUTPUT_NORM=output_norm_weight is not None,
                USE_ALIGNED_MEMORY_HINTS=rows_are_16b_aligned,
                HIDDEN_TILE_SIZE=hidden_tile_size,
            )
        else:
            config = _select_fixed_launch_config(num_tokens, num_blocks)
            cta_count = (
                min(num_tokens, _PERSISTENT_CTA_COUNT_PER_SM * get_sm_count())
                if config.use_persistent_token_loop
                else num_tokens
            )
            delta_ptr = prefix if delta is None else delta
            _attn_res_fixed_config_kernel[(cta_count,)](
                prefix,
                delta_ptr,
                blocks,
                norm_weight,
                qk_weight,
                output_norm_weight_ptr,
                output,
                prefix.stride(0),
                0 if delta is None else delta.stride(0),
                blocks.stride(0),
                blocks.stride(1),
                output.stride(0),
                num_tokens,
                eps,
                output_norm_eps,
                NUM_BLOCKS=num_blocks,
                HIDDEN_SIZE=hidden_size,
                BLOCK_WRITE_INDEX=block_write_idx,
                ADD_DELTA_TO_PREFIX=delta is not None,
                STORE_PREFIX_TO_BLOCK=block_write_idx >= 0,
                APPLY_OUTPUT_NORM=output_norm_weight is not None,
                USE_PERSISTENT_TOKEN_LOOP=config.use_persistent_token_loop,
                USE_COMPACT_SOURCE_REDUCTION=config.use_compact_source_reduction,
                USE_ALIGNED_MEMORY_HINTS=rows_are_16b_aligned,
                CTA_COUNT=cta_count,
                SOURCE_TILE_SIZE=config.source_tile_size,
                HIDDEN_TILE_SIZE=hidden_tile_size,
                num_warps=config.num_warps,
                num_stages=config.num_stages,
                launch_pdl=config.enable_pdl,
            )
    return output


__all__ = ["attn_res"]
