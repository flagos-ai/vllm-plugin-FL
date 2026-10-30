# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned cache leaves for the DeepSeek V4 reference path."""

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec

from vllm_fl.attention.deepseek_v4 import (
    DeepseekV4CacheBackend,
    validate_cache_spec,
)
from vllm_fl.dispatch import CachedOp

_write = CachedOp("deepseek_v4_cache_write")


def create_deepseek_v4_cache_layers(config, cache_config, prefix):
    """Build scheduler leaves for plain reference storage, not packed OPP pages.

    Small state/SWA pages share the main C4 page size. Their logical windows
    are unchanged; compression applies only to the full compact-cache specs.
    """
    block = cache_config.block_size
    if block <= 0 or block % 128:
        raise ValueError(
            "Plain DeepSeek V4 caches require a block size divisible by 128"
        )
    if cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise NotImplementedError("Plain DeepSeek V4 main caches require BF16")
    ratio = config.compress_ratios[extract_layer_index(prefix)]
    if ratio not in (0, 4, 128):
        raise ValueError("DeepSeek V4 compression ratio must be 0, 4 or 128")
    width = config.head_dim
    page_bytes = (block // 4) * width * 2
    common = dict(num_kv_heads=1, model_version="deepseek_v4")
    specs = {
        "swa": (
            "swa_cache",
            SlidingWindowMLASpec(
                block_size=block // 4,
                head_size=width,
                dtype=torch.bfloat16,
                sliding_window=config.sliding_window,
                **common,
            ),
        ),
    }
    if ratio:
        coff = 2 if ratio == 4 else 1
        state_block = block // (32 if ratio == 4 else 16)
        specs.update(
            {
                "state": (
                    "compressor.state_cache",
                    SlidingWindowMLASpec(
                        block_size=state_block,
                        head_size=2 * coff * width,
                        dtype=torch.float32,
                        sliding_window=coff * ratio,
                        **common,
                    ),
                ),
                "compact": (
                    "compact_cache",
                    MLAAttentionSpec(
                        block_size=block,
                        head_size=width,
                        dtype=torch.bfloat16,
                        compress_ratio=ratio,
                        page_size_padded=page_bytes,
                        **common,
                    ),
                ),
            }
        )
    if ratio == 4:
        index_width = config.index_head_dim
        specs.update(
            {
                "index_state": (
                    "indexer.compressor.state_cache",
                    SlidingWindowMLASpec(
                        block_size=block // 32,
                        head_size=4 * index_width,
                        dtype=torch.float32,
                        sliding_window=8,
                        **common,
                    ),
                ),
                "index_key": (
                    "indexer.k_cache",
                    MLAAttentionSpec(
                        block_size=block,
                        head_size=index_width,
                        dtype=torch.int8,
                        compress_ratio=4,
                        **common,
                    ),
                ),
                "index_scale": (
                    "indexer.scale_cache",
                    MLAAttentionSpec(
                        block_size=block,
                        head_size=1,
                        dtype=torch.float16,
                        compress_ratio=4,
                        **common,
                    ),
                ),
            }
        )
    return {
        name: DeepseekV4CacheLayer(f"{prefix}.{suffix}", spec)
        for name, (suffix, spec) in specs.items()
    }


class DeepseekV4CacheLayer(nn.Module, AttentionLayerBase):
    _fl_deepseek_v4_cache = True

    def __init__(self, prefix, spec, *, vllm_config=None):
        super().__init__()
        validate_cache_spec(spec)
        config = vllm_config or get_current_vllm_config()
        context = config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate cache layer: {prefix}")
        self.prefix = prefix
        self.spec = spec
        self.kv_cache = torch.tensor([])
        context[prefix] = self

    def get_kv_cache_spec(self, vllm_config):
        return self.spec

    def get_attn_backend(self):
        return DeepseekV4CacheBackend

    def store(self, values, metadata):
        slots = (
            metadata.output_slots
            if self.spec.compress_ratio > 1
            else metadata.slot_mapping
        )
        _write(values, slots, self.kv_cache)
