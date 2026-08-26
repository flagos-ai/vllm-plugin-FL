# Copyright (c) 2026 BAAI. All rights reserved.

"""Stable FL runner interface for the locally owned Ascend attention."""

from .attention import AscendAttentionBackend


class AscendAttentionBackendFL(AscendAttentionBackend):
    """Accept FL/vLLM's cache dtype keyword and delegate to the vendor code."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return AscendAttentionBackend.get_kv_cache_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str,
        )


__all__ = ["AscendAttentionBackendFL"]
