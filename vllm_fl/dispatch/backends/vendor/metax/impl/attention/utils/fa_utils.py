# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os

import torch

from vllm.platforms import current_platform

logger = logging.getLogger(__name__)

if current_platform.is_out_of_tree():
    # /------------------------  Metax Modification -------------------------\
    # ops.reshape_and_cache_flash is a CUDA-compiled C++ op that does not
    # exist on MetaX MACA devices.  Two backends are available:
    #
    #   USE_FLAGGEMS=1 (default)  — FlagGems Triton/MACA kernel (production)
    #   USE_FLAGGEMS=0            — Pure-PyTorch reference implementation
    #                               (useful for debugging numerical issues)
    #
    # The op backend preference is also declared in metax.yaml under
    # op_backends.reshape_and_cache_flash so the dispatch layer routes here.

    def _pytorch_reshape_and_cache_flash(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        """Pure-PyTorch reference implementation of reshape_and_cache_flash.

        Writes key/value tokens into the paged KV cache according to
        slot_mapping.  slot_mapping[i] = block_idx * block_size + block_offset
        for the i-th token.

        Only supports float16/bfloat16 (no FP8 quantisation).
        """
        num_tokens = slot_mapping.shape[0]
        block_size = key_cache.shape[1]   # (num_blocks, block_size, num_heads, head_size)

        block_idx = slot_mapping // block_size    # [num_tokens]
        block_off = slot_mapping % block_size     # [num_tokens]

        # key/value shape: [num_tokens, num_heads, head_size]
        key_cache[block_idx, block_off] = key[:num_tokens]
        value_cache[block_idx, block_off] = value[:num_tokens]

    def get_reshape_and_cache_flash():
        """Return the reshape_and_cache_flash function to use.

        Controlled by the USE_FLAGGEMS environment variable:
          USE_FLAGGEMS=0 → pure-PyTorch reference
          USE_FLAGGEMS=1 (default) → FlagGems Triton/MACA kernel
        """
        use_flaggems = os.environ.get("USE_FLAGGEMS", "1") != "0"
        if use_flaggems:
            try:
                from flag_gems.fused.reshape_and_cache_flash import (
                    reshape_and_cache_flash as _fg_impl,
                )
                logger.info(
                    "reshape_and_cache_flash: using FlagGems implementation"
                )
                return _fg_impl
            except ImportError as e:
                raise RuntimeError(
                    "FlagGems is required on MetaX MACA devices for "
                    "reshape_and_cache_flash but could not be imported. "
                    f"Set USE_FLAGGEMS=0 to use the PyTorch reference "
                    f"implementation instead. Original error: {e}"
                ) from e
        else:
            logger.info(
                "reshape_and_cache_flash: USE_FLAGGEMS=0, "
                "using pure-PyTorch reference implementation"
            )
            return _pytorch_reshape_and_cache_flash

    # Module-level binding so existing `from .utils.fa_utils import
    # reshape_and_cache_flash` callers keep working.
    reshape_and_cache_flash = get_reshape_and_cache_flash()

    from flash_attn import flash_attn_varlen_func  # noqa: F401
    from flash_attn import flash_attn_with_kvcache  # noqa: F401
    # \------------------------- Metax Modification -------------------------/

    get_scheduler_metadata = None


def get_flash_attn_version(requires_alibi: bool = False) -> int | None:
    logger.info_once(
        "Using Maca version of flash attention, which only supports version 2."
    )
    # Note: In maca this needs to be None since the MetaX flash_attn API
    # does not have a parameter for `fa_version`.
    return None


def flash_attn_supports_fp8() -> bool:
    logger.info_once(
        "Using Maca version of flash attention, which does not support FP8"
    )
    return False


def flash_attn_supports_sinks() -> bool:
    # MetaX fa2 supports attention sinks
    return True


def flash_attn_supports_mla():
    return False


def is_flash_attn_with_kvcache_available() -> bool:
    return False


def is_flash_attn_varlen_func_available() -> bool:
    return True
