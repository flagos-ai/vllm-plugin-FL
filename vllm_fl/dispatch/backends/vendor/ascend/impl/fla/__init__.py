# Copyright (c) 2026 BAAI. All rights reserved.
from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.chunk import (
    chunk_gated_delta_rule as chunk_gated_delta_rule_npu,
)

__all__ = ["chunk_gated_delta_rule_npu"]
