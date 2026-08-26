# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/compilation/cuda_graph.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import re
import weakref
from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from typing import Any, ClassVar
from unittest.mock import patch

import torch

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_fl.compilation.graph_runtime import (
    get_graph_class,
    get_graph_runtime_backend,
)

logger = init_logger(__name__)


# FL-specific: platform-agnostic weak_ref_tensors
def weak_ref_tensors(tensor: Any) -> Any:
    try:
        from vllm.utils.torch_utils import weak_ref_tensors
        return weak_ref_tensors(tensor)
    except Exception:
        return tensor


# FL-specific: platform-agnostic graph class selection
class Graph:
    @staticmethod
    def graph() -> Any:
        # Resolve lazily so importing worker modules does not require an
        # initialized accelerator runtime (for example in CPU-only unit tests).
        return get_graph_class()()


# Re-export CUDAGraphStat for compatibility
from vllm.compilation.cuda_graph import CUDAGraphStat  # noqa: F401, E402


@dataclasses.dataclass
class GraphEntry:
    batch_descriptor: BatchDescriptor
    graph: Any | None = None
    output: Any | None = None

    # for graph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: list[int] | None = None

@dataclasses.dataclass
class GraphOptions:
    debug_log_enable: bool = True
    gc_disable: bool = False
    weak_ref_output: bool = True


@dataclasses.dataclass
class AscendGraphParams:
    """Host-side task update state for a captured Ascend full graph."""

    events: dict[int, list[Any]]
    workspaces: dict[int, torch.Tensor | None]
    handles: dict[int, list[Any]]
    attention_params: dict[int, list[tuple[Any, ...]]]
    conv1d_events: dict[int, list[Any]]
    conv1d_handles: dict[int, list[Any]]
    conv1d_params: dict[int, list[tuple[Any, ...]]]
    task_order: dict[int, list["AscendGraphTaskDescriptor"]]


@dataclasses.dataclass(frozen=True)
class AscendGraphTaskDescriptor:
    """One captured dynamic task in model execution order."""

    kind: str
    index: int
    layer_name: str


_ascend_graph_params: AscendGraphParams | None = None
_ascend_graph_capturing = False


def set_ascend_graph_params(capture_sizes: list[int]) -> None:
    """Initialize per-shape task update storage before graph capture."""
    global _ascend_graph_params
    sizes = sorted(set(capture_sizes))
    _ascend_graph_params = AscendGraphParams(
        events={size: [] for size in sizes},
        workspaces={size: None for size in sizes},
        handles={size: [] for size in sizes},
        attention_params={size: [] for size in sizes},
        conv1d_events={size: [] for size in sizes},
        conv1d_handles={size: [] for size in sizes},
        conv1d_params={size: [] for size in sizes},
        task_order={size: [] for size in sizes},
    )


def get_ascend_graph_params() -> AscendGraphParams | None:
    return _ascend_graph_params


def update_ascend_graph_params_workspace(
    num_tokens: int,
    workspace: torch.Tensor,
) -> None:
    """Keep the per-shape attention workspace used by graph task updates."""
    if _ascend_graph_params is not None:
        _ascend_graph_params.workspaces[num_tokens] = workspace


def weak_ref_ascend_graph_workspaces() -> None:
    """Release strong Python references after NPUGraph owns workspaces."""
    if _ascend_graph_params is None:
        return
    for num_tokens, workspace in _ascend_graph_params.workspaces.items():
        if workspace is not None:
            _ascend_graph_params.workspaces[num_tokens] = weak_ref_tensors(
                workspace
            )


def set_ascend_graph_capturing(capturing: bool) -> None:
    global _ascend_graph_capturing
    _ascend_graph_capturing = capturing


def is_ascend_graph_capturing() -> bool:
    return _ascend_graph_capturing


def record_ascend_graph_task(
    num_tokens: int,
    kind: str,
    index: int,
    layer_name: str,
) -> None:
    """Record a task at its actual capture-time model position."""
    if _ascend_graph_params is None:
        return
    if num_tokens not in _ascend_graph_params.task_order:
        return
    _ascend_graph_params.task_order[num_tokens].append(
        AscendGraphTaskDescriptor(kind, index, layer_name)
    )


def _interleaved_graph_task_update_enabled(vllm_config: VllmConfig) -> bool:
    """Return whether the experimental hybrid decode-only path is eligible."""
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if not isinstance(additional_config, Mapping):
        return False
    compilation_config = getattr(vllm_config, "compilation_config", None)
    return (
        current_platform.device_type == "npu"
        and bool(
            additional_config.get(
                "enable_interleaved_graph_task_update", True
            )
        )
        and bool(
            getattr(getattr(vllm_config, "model_config", None),
                    "is_hybrid", False)
        )
        and getattr(vllm_config, "speculative_config", None) is None
        and getattr(compilation_config, "cudagraph_mode", None)
        == CUDAGraphMode.FULL_DECODE_ONLY
    )


_LAYER_INDEX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _validated_interleaved_task_order(
    graph_params: AscendGraphParams,
    num_tokens: int,
) -> list[AscendGraphTaskDescriptor] | None:
    """Validate exact coverage and monotonically increasing model layers.

    Returning ``None`` deliberately selects the established attention-then-GDN
    update path.  The strict one-task-per-layer contract matches the non-spec
    Qwen3.5/Qwen3.6 hybrid decode graph and prevents an experimental ordering
    from being used for unfamiliar graph layouts.
    """
    if num_tokens not in graph_params.task_order:
        return None
    order = graph_params.task_order[num_tokens]
    num_attention = len(graph_params.attention_params.get(num_tokens, ()))
    num_conv1d = len(graph_params.conv1d_params.get(num_tokens, ()))
    if not (
        num_attention == len(graph_params.handles.get(num_tokens, ()))
        == len(graph_params.events.get(num_tokens, ()))
        and num_conv1d
        == len(graph_params.conv1d_handles.get(num_tokens, ()))
        == len(graph_params.conv1d_events.get(num_tokens, ()))
    ):
        return None
    if len(order) != num_attention + num_conv1d:
        return None

    expected = {
        ("attention", index) for index in range(num_attention)
    } | {("conv1d", index) for index in range(num_conv1d)}
    actual = {(task.kind, task.index) for task in order}
    if len(actual) != len(order) or actual != expected:
        return None

    layer_indices: list[int] = []
    for task in order:
        if task.kind == "attention":
            params = graph_params.attention_params[num_tokens]
        elif task.kind == "conv1d":
            params = graph_params.conv1d_params[num_tokens]
        else:
            return None
        if task.index >= len(params):
            return None
        captured_layer_name = params[task.index][0 if task.kind == "attention" else 9]
        if task.layer_name != captured_layer_name:
            return None
        match = _LAYER_INDEX_PATTERN.search(task.layer_name)
        if match is None:
            return None
        layer_indices.append(int(match.group(1)))

    if (
        layer_indices != sorted(layer_indices)
        or len(set(layer_indices)) != len(layer_indices)
    ):
        return None
    return order


def _interleaved_forward_metadata_complete(
    task_order: list[AscendGraphTaskDescriptor],
    forward_context: Any,
    attention_metadata_type: type,
    conv1d_metadata_type: type,
) -> bool:
    """Reject the candidate before submitting any task on partial metadata."""
    metadata = getattr(forward_context, "attn_metadata", None)
    if not isinstance(metadata, dict):
        return False
    for task in task_order:
        expected_type = (
            attention_metadata_type
            if task.kind == "attention"
            else conv1d_metadata_type
        )
        if not isinstance(metadata.get(task.layer_name), expected_type):
            return False
    return True


def update_ascend_full_graph_params(
    update_stream: Any,
    forward_context: Any,
    num_tokens: int,
    vllm_config: VllmConfig,
) -> None:
    """Refresh host parameters consumed by task groups on the next replay."""
    from vllm_fl.dispatch.backends.vendor.ascend.impl.attention import (
        AscendAttentionBackendImpl,
        AscendMetadata,
        using_paged_attention,
    )
    from vllm_fl.dispatch.backends.vendor.ascend.impl.gdn import (
        GDNAttentionMetadata,
        _update_conv1d_graph_task,
        update_conv1d_graph_params,
    )

    graph_params = get_ascend_graph_params()
    if (
        graph_params is not None
        and _interleaved_graph_task_update_enabled(vllm_config)
        and (
            task_order := _validated_interleaved_task_order(
                graph_params, num_tokens
            )
        )
        is not None
        and _interleaved_forward_metadata_complete(
            task_order,
            forward_context,
            AscendMetadata,
            GDNAttentionMetadata,
        )
    ):
        attention_uses_paged = using_paged_attention(num_tokens, vllm_config)
        with torch.npu.stream(update_stream):
            for task in task_order:
                if task.kind == "attention":
                    AscendAttentionBackendImpl._update_graph_task(
                        update_stream,
                        forward_context,
                        num_tokens,
                        vllm_config,
                        task.index,
                        use_pa=attention_uses_paged,
                    )
                else:
                    _update_conv1d_graph_task(
                        update_stream,
                        forward_context,
                        num_tokens,
                        vllm_config,
                        task.index,
                    )
        return

    AscendAttentionBackendImpl.update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
    )
    update_conv1d_graph_params(
        update_stream, forward_context, num_tokens, vllm_config
    )


class GraphWrapper:
    """FL-specific graph wrapper that supports multiple device types (CUDA, NPU).
    Adapted from upstream CUDAGraphWrapper with platform-agnostic graph capture."""

    _all_instances: ClassVar[weakref.WeakSet["GraphWrapper"]] = weakref.WeakSet()

    @classmethod
    def clear_all_graphs(cls) -> None:
        """Clear captured graphs from all GraphWrapper instances."""
        for instance in list(cls._all_instances):
            instance.clear_graphs()

    def __init__(self,
                 runnable: Callable,
                 vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode,
                 cudagraph_options: GraphOptions | None = None):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config
        self.graph_runtime = get_graph_runtime_backend()

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self._runnable_str = str(runnable) if self.is_debugging_mode else None

        # assert runtime_mode is not NONE(no cudagraph), otherwise, we don't
        # need to initialize a GraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = GraphOptions()
        self.cudagraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # cudagraphs for.
        self.concrete_graph_entries: dict[BatchDescriptor, GraphEntry] = {}

        GraphWrapper._all_instances.add(self)

    def __getattr__(self, key: str) -> Any:
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        if self.is_debugging_mode:
            raise AttributeError(
                f"Attribute {key} not exists in the runnable of "
                f"cudagraph wrapper: {self._runnable_str}"
            )
        raise AttributeError

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    @property
    def cudagraph_wrapper(self) -> "GraphWrapper":
        return self

    def clear_graphs(self) -> None:
        self.concrete_graph_entries.clear()

    def __call__(self, *args, **kwargs):
        if not is_forward_context_available():
            # No forward context means we are outside the normal
            # inference path (e.g. a vision encoder forward pass).
            return self.runnable(*args, **kwargs)

        forward_context = get_forward_context()
        self.graph_runtime.prepare_forward_context(forward_context)
        batch_descriptor = forward_context.batch_descriptor
        graph_runtime_mode = forward_context.cudagraph_runtime_mode

        if (
            graph_runtime_mode == CUDAGraphMode.NONE
            or graph_runtime_mode != self.runtime_mode
        ):
            return self.runnable(*args, **kwargs)

        assert batch_descriptor is not None
        if batch_descriptor not in self.concrete_graph_entries:
            # create a new entry for this batch descriptor
            self.concrete_graph_entries[batch_descriptor] = GraphEntry(
                batch_descriptor=batch_descriptor
            )

        entry = self.concrete_graph_entries[batch_descriptor]

        if entry.graph is None:
            if self.cudagraph_options.debug_log_enable:
                logger.debug(
                    "Capturing a cudagraph on (%s,%s)",
                    self.runtime_mode.name,
                    entry.batch_descriptor,
                )
            # validate that cudagraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            graph = Graph.graph()

            with ExitStack() as stack:
                if self.cudagraph_options.gc_disable:
                    stack.enter_context(patch("gc.collect", lambda: None))
                    # FL-specific: patch our platform's empty_cache
                    stack.enter_context(
                        patch("vllm_fl.platform.PlatformFL.empty_cache",
                              lambda: None)
                    )

            if self.graph_pool is not None:
                set_graph_pool_id(self.graph_pool)
            else:
                set_graph_pool_id(current_platform.graph_pool_handle())

            # Sync offloader's copy stream before capture if available.
            try:
                from vllm.model_executor.offloader.base import get_offloader
                get_offloader().sync_prev_onload()
            except (ImportError, RuntimeError):
                pass

            self.graph_runtime.begin_capture(forward_context)
            try:
                # FL-specific: use platform-agnostic graph capture
                with current_platform.torch_device_fn.graph(
                    graph, pool=self.graph_pool
                ):
                    # `output` is managed by pytorch's graph pool
                    output = self.runnable(*args, **kwargs)
                    # Join offloader's copy stream after forward if available
                    try:
                        from vllm.model_executor.offloader.base import get_offloader
                        get_offloader().join_after_forward()
                    except (ImportError, RuntimeError):
                        pass
                    if self.cudagraph_options.weak_ref_output:
                        output = weak_ref_tensors(output)
            finally:
                self.graph_runtime.end_capture()

            self.graph_runtime.after_capture()

            entry.output = weak_ref_tensors(output)
            entry.graph = graph

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during graph capture
            return output

        if self.is_debugging_mode:
            # check if the input addresses are the same
            new_input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for cudagraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}"
            )

        # Sync offloader before replay if available
        try:
            from vllm.model_executor.offloader.base import get_offloader
            get_offloader().sync_prev_onload()
        except (ImportError, RuntimeError):
            pass

        self.graph_runtime.before_replay()
        entry.graph.replay()
        return entry.output
