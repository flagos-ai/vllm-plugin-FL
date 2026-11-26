# Copyright (c) 2026 BAAI. All rights reserved.

"""Use the MetaX launch parameters for the vLLM Triton backend's KV writer."""

from inspect import signature

from vllm.v1.attention.backends import triton_attn

from ..impl.attention.ops.reshape_and_cache import triton_reshape_and_cache_flash

if signature(triton_attn.triton_reshape_and_cache_flash) != signature(
    triton_reshape_and_cache_flash
):
    raise RuntimeError("Unexpected vLLM Triton KV cache writer signature")

triton_attn.triton_reshape_and_cache_flash = triton_reshape_and_cache_flash
