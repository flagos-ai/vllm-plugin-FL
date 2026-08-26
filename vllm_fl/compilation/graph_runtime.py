# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-specific lifecycle hooks for static graph execution."""

from contextlib import contextmanager, nullcontext
from typing import Any

import torch

from vllm.distributed.parallel_state import (
    GraphCaptureContext,
)
from vllm.platforms import current_platform


_GRAPH_CLASS_NAMES = {
    "cuda": "CUDAGraph",
    "npu": "NPUGraph",
    "musa": "MUSAGraph",
    "ptpu": "PTPUGraph",
    "gcu": "GCUGraph",
}


@contextmanager
def _ascend_graph_capture(device: torch.device):
    """Capture an NPUGraph on a stream isolated from the default stream."""
    capture_context = GraphCaptureContext(
        current_platform.torch_device_fn.Stream(device=device)
    )
    stream = capture_context.stream
    current_stream = current_platform.torch_device_fn.current_stream()
    if current_stream != stream:
        stream.wait_stream(current_stream)

    with current_platform.torch_device_fn.stream(stream), nullcontext():
        yield capture_context


def get_graph_capture(default_capture: Any) -> Any:
    """Override graph capture only for Ascend; preserve other vendors."""
    if current_platform.device_type == "npu":
        return _ascend_graph_capture
    return default_capture


def get_graph_class(device_type: str | None = None) -> Any:
    """Resolve the torch graph class for the active accelerator."""
    device_type = device_type or current_platform.device_type
    if device_type == "txda":
        return None
    graph_class_name = _GRAPH_CLASS_NAMES.get(device_type)
    if graph_class_name is None:
        raise NotImplementedError(
            f"Static graph is not supported on device type {device_type!r}"
        )
    try:
        return getattr(getattr(torch, device_type), graph_class_name)
    except AttributeError as exc:
        raise NotImplementedError(
            f"Torch does not provide {graph_class_name} for {device_type!r}"
        ) from exc


class GraphRuntimeBackend:
    """No-op lifecycle hooks shared by graph-capable accelerators."""

    def prepare_forward_context(self, forward_context: Any) -> None:
        pass

    def begin_capture(self, forward_context: Any) -> None:
        pass

    def end_capture(self) -> None:
        pass

    def after_capture(self) -> None:
        pass

    def before_replay(self) -> None:
        pass


class AscendGraphRuntimeBackend(GraphRuntimeBackend):
    """Ascend synchronization around NPUGraph replay."""

    def before_replay(self) -> None:
        current_platform.torch_device_fn.synchronize()


_GRAPH_RUNTIME_BACKENDS: dict[str, type[GraphRuntimeBackend]] = {
    "npu": AscendGraphRuntimeBackend,
}


def get_graph_runtime_backend(
    device_type: str | None = None,
) -> GraphRuntimeBackend:
    """Create lifecycle hooks for the active accelerator."""
    device_type = device_type or current_platform.device_type
    backend_cls = _GRAPH_RUNTIME_BACKENDS.get(device_type, GraphRuntimeBackend)
    return backend_cls()
