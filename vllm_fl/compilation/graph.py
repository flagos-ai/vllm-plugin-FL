# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/compilation/cuda_graph.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any, Optional
from unittest.mock import patch

import torch

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# --------------------------------------------------------------------------- #
# Backend-specific extension registry for GraphWrapper.
#
# A vendor backend can register a mixin class that implements any subset of the
# hook methods below.  The mixin is instantiated once per GraphWrapper and
# receives the same hook calls during capture/replay.  This keeps the generic
# framework code free of device-specific details.
# --------------------------------------------------------------------------- #
_graph_wrapper_backend_registry: dict[str, type] = {}


def register_graph_wrapper_backend(device_type: str, backend_cls: type) -> None:
    """Register a backend-specific mixin for GraphWrapper.

    The mixin class may implement any of the following hook methods:

    - before_capture(self, entry: GraphEntry, args, kwargs) -> None
        Called after input_addresses are recorded but before the graph capture
        context is entered.  Useful for stream synchronization or patching the
        runtime environment.

    - wrap_capture_context(self, entry: GraphEntry, stack: ExitStack) -> None
        Called inside the ExitStack used around graph capture.  The backend can
        enter additional context managers (e.g. disable gc / empty_cache).

    - after_capture(self, entry: GraphEntry, output, args, kwargs) -> Any
        Called after the graph capture block exits but before the entry is
        stored.  The return value replaces `output` if not None.

    - before_replay(self, entry: GraphEntry, args, kwargs) -> None
        Called before graph replay.  Useful for device synchronization.

    - capture_error_handler(self, exc: BaseException) -> None
        Called when graph capture raises.  May translate the exception or raise
        a more informative error.

    - weak_ref_tensors(self, tensor: Any) -> Any
        Return a weak-ref version of tensors for the current backend.  If not
        implemented, falls back to the generic helper.
    """
    _graph_wrapper_backend_registry[device_type] = backend_cls
    logger.info_once("Registered graph wrapper backend for device_type=%s: %s",
                     device_type, backend_cls.__name__)


def get_graph_wrapper_backend(device_type: str) -> Optional[type]:
    return _graph_wrapper_backend_registry.get(device_type)


def weak_ref_tensors(tensor: Any) -> Any:
    backend = _get_active_backend()
    if backend is not None and hasattr(backend, "weak_ref_tensors"):
        return backend.weak_ref_tensors(tensor)
    if current_platform.device_type == "cuda":
        from vllm.utils.torch_utils import weak_ref_tensors
        return weak_ref_tensors(tensor)
    # TODO: add csrc npu custom op when available
    return tensor


# Per-wrapper active backend instance.  This is set by GraphWrapper.__init__ and
# used by the weak_ref_tensors helper so callers do not need to pass `self`.
_active_backend_instance: Optional[Any] = None


def _set_active_backend(backend: Optional[Any]) -> None:
    global _active_backend_instance
    _active_backend_instance = backend


def _get_active_backend() -> Optional[Any]:
    return _active_backend_instance


class Graph:
    if current_platform.device_type == "cuda":
        graph = torch.cuda.CUDAGraph
    elif current_platform.device_type == "npu":
        graph = torch.npu.NPUGraph
    elif current_platform.device_type == "musa":
        graph = torch.musa.MUSAGraph
    else:
        raise NotImplementedError("not support graph")


@dataclasses.dataclass
class GraphEntry:
    batch_descriptor: BatchDescriptor
    graph: Optional[Graph] = None
    output: Optional[Any] = None

    # for graph debugging, track the input addresses
    # during capture, and check if they are the same during replay
    input_addresses: Optional[list[int]] = None


@dataclasses.dataclass
class GraphOptions:
    debug_log_enable: bool = True
    gc_disable: bool = False
    weak_ref_output: bool = True


class GraphWrapper:
    def __init__(self,
                 runnable: Callable,
                 vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode,
                 cudagraph_options: Optional[GraphOptions] = None):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"

        # assert runtime_mode is not NONE(no cudagraph), otherwise, we don't
        # need to initialize a CUDAGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        self.graph_pool = current_platform.get_global_graph_pool()

        if cudagraph_options is None:
            cudagraph_options = GraphOptions()
        self.graph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # cudagraphs for.
        self.concrete_graph_entries: dict[BatchDescriptor, GraphEntry] = {}

        # Instantiate backend-specific mixin if one has been registered.
        backend_cls = get_graph_wrapper_backend(current_platform.device_type)
        if backend_cls is not None:
            self.backend = backend_cls(self)
        else:
            self.backend = None

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(
            f"Attribute {key} not exists in the runnable of "
            f"cudagraph wrapper: {self.runnable}"
        )

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        graph_runtime_mode = forward_context.cudagraph_runtime_mode

        if (
            graph_runtime_mode == CUDAGraphMode.NONE
            or graph_runtime_mode != self.runtime_mode
        ):
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without cudagraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)

        if batch_descriptor not in self.concrete_graph_entries:
            # create a new entry for this batch descriptor
            self.concrete_graph_entries[batch_descriptor] = GraphEntry(
                batch_descriptor=batch_descriptor
            )

        entry = self.concrete_graph_entries[batch_descriptor]

        if entry.graph is None:
            if self.graph_options.debug_log_enable:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
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

            _set_active_backend(self.backend)

            # Give the backend a chance to run pre-capture logic (e.g. stream
            # sync, offloader sync).
            if self.backend is not None and hasattr(self.backend,
                                                    "before_capture"):
                self.backend.before_capture(entry, args, kwargs)

            with ExitStack() as stack:
                if self.graph_options.gc_disable:
                    # during every model forward for piecewise graph
                    # mode, we will capture many pieces of graphs
                    # (roughly one per layer). running gc again and again
                    # across layers will make the graph capture very slow.
                    # therefore, we only run gc for the first graph,
                    # and disable gc for the rest of the graphs.
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(
                        patch("vllm_fl.platform.PlatformFL.empty_cache",
                              lambda: None)
                    )

                # Backend-specific context wrappers (e.g. disable NPU
                # empty_cache when it lives on a different module path).
                if self.backend is not None and hasattr(
                        self.backend, "wrap_capture_context"):
                    self.backend.wrap_capture_context(entry, stack)

                set_graph_pool_id(self.graph_pool)

                # mind-exploding: carefully manage the reference and memory.
                try:
                    with current_platform.torch_device_fn.graph(
                            graph, pool=self.graph_pool):
                        # `output` is managed by pytorch's cudagraph pool
                        output = self.runnable(*args, **kwargs)
                        if self.graph_options.weak_ref_output:
                            # by converting it to weak ref,
                            # the original `output` will immediately be released
                            # to save memory. It is only safe to do this for
                            # the last graph in piecewise cuadgraph mode, because
                            # the output of the last graph will not be used by
                            # any other cuda graph.
                            output = weak_ref_tensors(output)
                except BaseException as exc:
                    if self.backend is not None and hasattr(
                            self.backend, "capture_error_handler"):
                        self.backend.capture_error_handler(exc)
                    raise

            # Backend-specific post-capture logic (weak-ref workspaces,
            # offloader join, etc.).  The return value may replace `output`.
            if self.backend is not None and hasattr(self.backend,
                                                    "after_capture"):
                backend_output = self.backend.after_capture(
                    entry, output, args, kwargs)
                if backend_output is not None:
                    output = backend_output

            entry.output = weak_ref_tensors(output)
            entry.graph = graph

            compilation_counter.num_cudagraph_captured += 1

            _set_active_backend(None)

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

        _set_active_backend(self.backend)

        if self.backend is not None and hasattr(self.backend,
                                                "before_replay"):
            self.backend.before_replay(entry, args, kwargs)
        else:
            current_platform.torch_device_fn.synchronize()

        entry.graph.replay()

        _set_active_backend(None)
        return entry.output
