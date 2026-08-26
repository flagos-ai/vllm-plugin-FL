# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-specific lifecycle hooks for static graph execution."""

from contextlib import contextmanager, nullcontext
from typing import Any

import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.parallel_state import GraphCaptureContext
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
    """Provide an isolated stream for Ascend NPUGraph capture.

    The capture stream waits for work already queued on the current stream
    before capture starts. Entering this context makes the capture stream
    current, while leaving it restores the previously active stream. The
    returned ``GraphCaptureContext`` exposes the capture stream to callers.
    """
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

    def prepare_model_compile(self) -> None:
        pass

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

    def prepare_graph_wrapper(self) -> None:
        pass

    def prepare_capture(self, capture_descs: Any) -> None:
        pass

    def after_model_forward(self, vllm_config: VllmConfig) -> None:
        pass


class AscendGraphRuntimeBackend(GraphRuntimeBackend):
    """Ascend attention task state and synchronization around NPUGraph."""

    def __init__(self) -> None:
        self._update_stream: Any | None = None

    def prepare_forward_context(self, forward_context: Any) -> None:
        forward_context.capturing = False

    def begin_capture(self, forward_context: Any) -> None:
        from vllm_fl.compilation.graph import set_ascend_graph_capturing

        forward_context.capturing = True
        set_ascend_graph_capturing(True)

    def end_capture(self) -> None:
        from vllm_fl.compilation.graph import set_ascend_graph_capturing

        set_ascend_graph_capturing(False)

    def prepare_model_compile(self) -> None:
        from vllm_fl.dispatch import prewarm_cached_ops

        prewarm_cached_ops()

    def after_capture(self) -> None:
        from vllm_fl.compilation.graph import weak_ref_ascend_graph_workspaces

        weak_ref_ascend_graph_workspaces()

    def before_replay(self) -> None:
        current_platform.torch_device_fn.synchronize()

    def prepare_graph_wrapper(self) -> None:
        self._update_stream = torch.npu.Stream()

    def prepare_capture(self, capture_descs: Any) -> None:
        from vllm_fl.compilation.graph import set_ascend_graph_params

        set_ascend_graph_params(
            [
                desc.num_tokens
                for _, descriptors in capture_descs
                for desc in descriptors
            ]
        )

    def after_model_forward(self, vllm_config: VllmConfig) -> None:
        if self._update_stream is None:
            return

        from vllm.forward_context import get_forward_context

        from vllm_fl.compilation.graph import update_ascend_full_graph_params

        forward_context = get_forward_context()
        if (
            forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL
            or getattr(forward_context, "capturing", False)
        ):
            return

        batch_descriptor = forward_context.batch_descriptor
        assert batch_descriptor is not None
        update_ascend_full_graph_params(
            self._update_stream,
            forward_context,
            batch_descriptor.num_tokens,
        )


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
