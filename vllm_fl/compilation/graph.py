# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/compilation/cuda_graph.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import weakref
from collections import Counter
from collections.abc import Callable
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

logger = init_logger(__name__)


# FL-specific: platform-agnostic weak_ref_tensors
def weak_ref_tensors(tensor: Any) -> Any:
    if current_platform.device_type == "cuda":
        from vllm.utils.torch_utils import weak_ref_tensors
        return weak_ref_tensors(tensor)
    else:
        ### TODO: add csrc npu custom op
        return tensor


# FL-specific: platform-agnostic graph class selection
class Graph:
    if current_platform.device_type == "cuda":
        graph = torch.cuda.CUDAGraph
    elif current_platform.device_type == "npu":
        graph = torch.npu.NPUGraph
    elif current_platform.device_type == "musa":
        graph = torch.musa.MUSAGraph
    elif current_platform.device_type == "ptpu":
        graph = torch.ptpu.PTPUGraph
    else:
        raise NotImplementedError("not support graph")


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

            # FL-specific: use platform-agnostic graph capture
            with current_platform.torch_device_fn.graph(
                graph, pool=self.graph_pool
            ):
                # `output` is managed by pytorch's cudagraph pool
                output = self.runnable(*args, **kwargs)
                # Join offloader's copy stream after forward if available
                try:
                    from vllm.model_executor.offloader.base import get_offloader
                    get_offloader().join_after_forward()
                except (ImportError, RuntimeError):
                    pass
                if self.cudagraph_options.weak_ref_output:
                    output = weak_ref_tensors(output)

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

        current_platform.torch_device_fn.synchronize()
        entry.graph.replay()
        return entry.output
