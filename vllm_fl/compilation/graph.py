# Copyright (c) 2025 BAAI. All rights reserved.
"""Public graph API and backend implementation routing."""

from typing import Any

from vllm.platforms import current_platform

from vllm_fl.compilation.graph_base import (
    CUDAGraphStat as CUDAGraphStat,
    Graph as Graph,
    GraphEntry as GraphEntry,
    GraphOptions as GraphOptions,
    GraphWrapper as GraphWrapper,
    _graph_capture_stream as _graph_capture_stream,
    _weak_ref_npu_tensor as _weak_ref_npu_tensor,
    _weak_ref_npu_tensors as _weak_ref_npu_tensors,
    weak_ref_tensors as _weak_ref_native_tensors,
)
from vllm_fl.compilation.musa_graph import (
    MUSAGraphWrapper,
    _weak_ref_musa_tensors,
)

# Keep legacy NPU/stream helpers available alongside the public graph API.
__all__ = [
    "CUDAGraphStat",
    "Graph",
    "GraphEntry",
    "GraphOptions",
    "GraphWrapper",
    "weak_ref_tensors",
    "_graph_capture_stream",
    "_weak_ref_npu_tensor",
    "_weak_ref_npu_tensors",
]

GraphWrapper._platform_wrappers["musa"] = MUSAGraphWrapper


def weak_ref_tensors(value: Any) -> Any:
    if current_platform.device_type == "musa":
        return _weak_ref_musa_tensors(value)
    return _weak_ref_native_tensors(value)
