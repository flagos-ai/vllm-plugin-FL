# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
"""Use the vLLM 0.32 MLA layer and metadata contracts.

The model owns the native MLAAttention layer, including KV-cache updates and
the separately selected prefill backend. Reusing these types keeps metadata
builders aligned with the native layer and its context-parallel managers.
"""

from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)

__all__ = [
    "MLACommonBackend",
    "MLACommonDecodeMetadata",
    "MLACommonImpl",
    "MLACommonMetadata",
    "MLACommonMetadataBuilder",
    "QueryLenSupport",
]
