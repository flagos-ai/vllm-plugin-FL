# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/compilation/cuda_graph.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import inspect
import threading
import weakref
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, suppress
from contextvars import ContextVar
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
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)


# FL-specific: platform-agnostic weak_ref_tensors
def _weak_ref_npu_tensor(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.device.type == "npu":
        import torch_npu

        # Bound in torch_npu/csrc/npu/Module.cpp in the pinned 2.10 runtime.
        weak_ref_tensor = getattr(
            getattr(torch_npu, "_C", None), "_weak_ref_tensor", None
        )
        if not callable(weak_ref_tensor):
            raise RuntimeError(
                "NPU graph weak references require torch_npu._C._weak_ref_tensor. "
                "Use the documented torch-npu 2.10 runtime or disable graph mode."
            )
        return weak_ref_tensor(value)
    return value


def _weak_ref_npu_tensors(value: Any) -> Any:
    """Recursively replace NPU tensors without retaining graph-pool storage."""
    if isinstance(value, torch.Tensor):
        return _weak_ref_npu_tensor(value)
    if isinstance(value, list):
        return [_weak_ref_npu_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_weak_ref_npu_tensors(item) for item in value)
    if isinstance(value, dict):
        return {key: _weak_ref_npu_tensors(item) for key, item in value.items()}

    from vllm.sequence import IntermediateTensors

    if isinstance(value, IntermediateTensors):
        return IntermediateTensors(
            {key: _weak_ref_npu_tensors(item) for key, item in value.tensors.items()}
        )
    return value


def weak_ref_tensors(value: Any) -> Any:
    if current_platform.device_type == "npu":
        # The empty vLLM wheel registers torch.ops._C.weak_ref_tensor as an
        # identity. torch_npu's private primitive creates the real weak tensor
        # needed to release captured graph outputs.
        return _weak_ref_npu_tensors(value)

    try:
        from vllm.utils.torch_utils import (
            weak_ref_tensors as upstream_weak_ref_tensors,
        )

        return upstream_weak_ref_tensors(value)
    except Exception:
        return value


def _graph_capture_stream() -> Any:
    if current_platform.device_type in {"npu", "txda"}:
        # vLLM's current_stream helper maintains CUDA-specific thread-local
        # state and can remain on the default stream while torch.npu.stream()
        # has selected the side stream used for graph capture (also on TXDA).
        return current_platform.torch_device_fn.current_stream()
    return current_stream()


def _validate_txda_graph_runnable(runnable) -> tuple[bool, str]:
    from vllm_fl.compilation.txda_partition import validate_txda_graph_callable

    valid, reason = validate_txda_graph_callable(runnable)
    if valid:
        return valid, reason
    from vllm.compilation.piecewise_backend import PiecewiseBackend

    # Only this known eager adapter is transparent. Arbitrary objects exposing
    # .graph or .runnable are not evidence that their execution matches that FX.
    if type(runnable) is not PiecewiseBackend:
        return False, reason
    if (
        runnable.compilation_config.backend != "eager"
        or runnable.compiled_runnables is not None
        or not runnable.range_entries
    ):
        return False, "TXDA requires an uncached eager PiecewiseBackend"
    valid, reason = validate_txda_graph_callable(runnable.graph)
    if not valid:
        return valid, reason
    known_wrapper = PiecewiseBackend.get_compiled_graph_wrapper(runnable, None)
    for entry in runnable.range_entries.values():
        if not entry.compiled:
            return False, "TXDA piecewise range is not compiled"
        compiled = entry.runnable
        if inspect.isfunction(compiled) and compiled.__code__ is known_wrapper.__code__:
            closure = inspect.getclosurevars(compiled).nonlocals
            if closure.get("self") is not runnable:
                return False, "TXDA piecewise wrapper has a different owner"
            compiled = closure.get("compiled_graph")
        valid, reason = validate_txda_graph_callable(compiled)
        if not valid:
            return valid, reason
    return True, ""


class _TxdaGraphUnsupported(RuntimeError):
    pass


class _TxdaGraphInputs:
    """Own stable inputs; eager split operations may allocate on every call."""

    def __init__(self, args, kwargs):
        from torch.utils._pytree import tree_flatten, tree_unflatten

        leaves, self.spec = tree_flatten((args, kwargs))
        self.buffers = []
        self.layout = []
        seen = {}
        storage_owners = {}
        for value in leaves:
            if isinstance(value, torch.Tensor):
                if (
                    type(value) not in (torch.Tensor, torch.nn.Parameter)
                    or value.device.type != current_platform.device_type
                    or value.layout != torch.strided
                    or not value.is_contiguous()
                    or value.storage_offset() != 0
                ):
                    raise _TxdaGraphUnsupported(
                        "TXDA staging requires contiguous device tensors without storage offsets"
                    )
                identity = id(value)
                pointer = value.untyped_storage().data_ptr()
                if (
                    value.numel()
                    and pointer in storage_owners
                    and storage_owners[pointer] != identity
                ):
                    raise _TxdaGraphUnsupported(
                        "TXDA staging cannot preserve distinct input storage aliases"
                    )
                storage_owners[pointer] = identity
                if identity not in seen:
                    seen[identity] = torch.empty_like(value)
                    if seen[identity].stride() != value.stride():
                        raise _TxdaGraphUnsupported(
                            "TXDA staging cannot preserve input strides"
                        )
                    seen[identity].copy_(value)
                self.buffers.append(seen[identity])
                self.layout.append(
                    (tuple(value.shape), value.dtype, value.device, value.stride())
                )
            elif value is None or type(value) in (
                bool,
                int,
                float,
                str,
                torch.dtype,
                torch.device,
            ):
                self.buffers.append(value)
                self.layout.append(None)
            else:
                raise _TxdaGraphUnsupported(
                    f"TXDA staging cannot validate {type(value).__name__}"
                )
        self.args, self.kwargs = tree_unflatten(self.buffers, self.spec)
        # Retain only owned buffers, never the original request's input tensors.
        self.aliases = tuple(
            next((j for j in range(i) if leaves[j] is value), i)
            if isinstance(value, torch.Tensor)
            else None
            for i, value in enumerate(leaves)
        )

    def update(self, args, kwargs):
        from torch.utils._pytree import tree_flatten

        leaves, spec = tree_flatten((args, kwargs))
        if spec != self.spec or len(leaves) != len(self.buffers):
            raise _TxdaGraphUnsupported("TXDA graph input structure changed")
        owners = {}
        for i, (value, buffer, layout) in enumerate(
            zip(leaves, self.buffers, self.layout)
        ):
            if layout is None:
                if type(value) is not type(buffer) or value != buffer:
                    raise _TxdaGraphUnsupported("TXDA graph scalar input changed")
                continue
            if (
                type(value) not in (torch.Tensor, torch.nn.Parameter)
                or value.layout != torch.strided
                or (tuple(value.shape), value.dtype, value.device, value.stride())
                != layout
                or not value.is_contiguous()
                or value.storage_offset() != 0
            ):
                raise _TxdaGraphUnsupported("TXDA graph tensor layout changed")
            identity = id(value)
            pointer = value.untyped_storage().data_ptr()
            if value.numel() and pointer in owners and owners[pointer] != identity:
                raise _TxdaGraphUnsupported("TXDA graph input storage aliases changed")
            owners[pointer] = identity
            alias = next((j for j in range(i) if leaves[j] is value), i)
            if alias != self.aliases[i]:
                raise _TxdaGraphUnsupported("TXDA graph input object aliases changed")
        # Validate every argument before changing any buffer.
        for i, (value, buffer, layout) in enumerate(
            zip(leaves, self.buffers, self.layout)
        ):
            if layout is not None and self.aliases[i] == i:
                buffer.copy_(value)

    def aliases_output(self, output) -> bool:
        from torch.utils._pytree import tree_flatten

        pointers = {
            value.untyped_storage().data_ptr()
            for value in self.buffers
            if isinstance(value, torch.Tensor) and value.numel()
        }
        leaves, _ = tree_flatten(output)
        return any(
            isinstance(value, torch.Tensor)
            and value.numel()
            and value.untyped_storage().data_ptr() in pointers
            for value in leaves
        )


class _TxdaGraphSequence:
    """Replay individually captured Triton launches for an audited FX region.

    The current TXDA SDK silently fails to execute multi-kernel graphs. Native
    work is therefore never inferred from a successful capture: both FX and
    actual operator events must match the conservative add/mul profile.
    """

    _capture_lock = threading.RLock()
    _marker = "TXDA_SINGLE_KERNEL_CAPTURE"
    _metadata_ops = frozenset(
        {
            "aten::empty",
            "aten::empty_like",
            "aten::empty_strided",
            "aten::as_strided",
            "aten::view",
            "aten::_unsafe_view",
            "aten::alias",
            "aten::transpose",
            "aten::permute",
            "aten::t",
            "aten::unsqueeze",
            "aten::squeeze",
            "aten::expand",
            "aten::detach",
            "aten::size",
            "aten::stride",
            "aten::numel",
            "aten::dim",
            "aten::storage_offset",
            "aten::is_contiguous",
            "aten::sym_size",
            "aten::sym_stride",
            "aten::sym_numel",
            "aten::sym_storage_offset",
        }
    )

    def __init__(self):
        # TXDA marks a pool dead after its last graph is destroyed; reusing the
        # global handle after memory profiling/clear triggers an allocator
        # use_count assertion. Allocations here live outside native capture, so
        # each sequence can safely own a fresh pool for kernel graph bookkeeping.
        self.pool = current_platform.graph_pool_handle()
        self.graphs = []
        self.retained = []
        self.inputs = None

    @classmethod
    def _contains_launch(cls, event):
        return event.name == cls._marker or any(
            cls._contains_launch(child) for child in event.cpu_children
        )

    @classmethod
    def _audit(cls, profile):
        for event in profile.events():
            if event.name == cls._marker or "::" not in event.name:
                continue
            if event.name in cls._metadata_ops:
                continue
            if event.name in {"aten::add", "aten::mul"} and cls._contains_launch(event):
                continue
            raise _TxdaGraphUnsupported(
                f"unrecorded TXDA graph operation: {event.name}"
            )

    def capture(self, runnable, args, kwargs):
        from triton.backends.tsingmicro.driver import TXDALauncher

        if torch.autograd._profiler_enabled():
            raise _TxdaGraphUnsupported(
                "TXDA graph audit cannot nest an active profiler"
            )
        self.inputs = _TxdaGraphInputs(args, kwargs)
        runtime = current_platform.torch_device_fn
        stream = _graph_capture_stream()
        owner_thread = threading.get_ident()
        with self._capture_lock:
            original = TXDALauncher.__call__

            def capture_launch(
                launcher,
                grid_x,
                grid_y,
                grid_z,
                actual_stream,
                function,
                *launch_args,
                **launch_kwargs,
            ):
                if threading.get_ident() != owner_thread:
                    return original(
                        launcher,
                        grid_x,
                        grid_y,
                        grid_z,
                        actual_stream,
                        function,
                        *launch_args,
                        **launch_kwargs,
                    )
                if actual_stream != stream.txda_stream:
                    raise _TxdaGraphUnsupported(
                        "TXDA launch switched away from the capture stream"
                    )
                # Strong references keep intermediate allocations and wrappers
                # alive even after the FX callable discards its local tensors.
                self.retained.append((launch_args, launch_kwargs))
                graph = runtime.TXDAGraph()
                with torch.profiler.record_function(self._marker):
                    graph.capture_begin(pool=self.pool)
                    try:
                        result = original(
                            launcher,
                            grid_x,
                            grid_y,
                            grid_z,
                            actual_stream,
                            function,
                            *launch_args,
                            **launch_kwargs,
                        )
                    finally:
                        graph.capture_end()
                    if result.res != 0:
                        raise RuntimeError(f"TXDA Triton launch failed: {result.res}")
                    self.graphs.append(graph)
                    # Recording-time Python/native consumers must see the
                    # intermediate. Every later replay repeats this launch.
                    graph.replay()
                return result

            try:
                with (
                    torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU]
                    ) as profile,
                    patch.object(TXDALauncher, "__call__", capture_launch),
                ):
                    output = runnable(*self.inputs.args, **self.inputs.kwargs)
                self._audit(profile)
                if not self.graphs:
                    raise _TxdaGraphUnsupported(
                        "TXDA region recorded no Triton launches"
                    )
                if self.inputs.aliases_output(output):
                    raise _TxdaGraphUnsupported(
                        "TXDA region returns an alias of a staged input"
                    )
                return output
            except Exception:
                runtime.synchronize()
                self.clear()
                raise

    def replay(self):
        for graph in self.graphs:
            graph.replay()

    def clear(self):
        self.graphs.clear()
        self.retained.clear()
        self.inputs = None


# FL-specific: platform-agnostic graph class selection
class Graph:
    if current_platform.device_type == "cuda":
        graph = torch.cuda.CUDAGraph
    elif current_platform.device_type == "npu":
        graph = torch.npu.NPUGraph
    elif current_platform.device_type == "musa":
        graph = torch.musa.MUSAGraph
    elif current_platform.device_type == "gcu":
        graph = torch.gcu.GCUGraph
    elif current_platform.device_type == "ptpu":
        graph = torch.ptpu.PTPUGraph
    elif current_platform.device_type == "txda":
        graph = torch.txda.TXDAGraph
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
    _active_txda_capture_scope: ClassVar[ContextVar] = ContextVar(
        "txda_graph_capture_scope", default=None
    )

    @classmethod
    @contextmanager
    def txda_capture_scope(cls):
        """Observe only wrappers actually captured or replayed by this runner."""
        wrappers = weakref.WeakSet()
        token = cls._active_txda_capture_scope.set(wrappers)
        try:
            yield wrappers
        finally:
            cls._active_txda_capture_scope.reset(token)

    @classmethod
    def count_live_txda_graphs(cls, wrappers) -> int:
        """Count live native graphs, excluding cleared or eager-only wrappers."""
        return len(
            {
                id(graph)
                for wrapper in list(wrappers)
                if not wrapper._txda_graph_disabled_reason
                for entry in wrapper.concrete_graph_entries.values()
                if isinstance(entry.graph, _TxdaGraphSequence)
                for graph in entry.graph.graphs
            }
        )

    def _record_txda_capture_scope(self) -> None:
        wrappers = self._active_txda_capture_scope.get()
        if wrappers is not None:
            wrappers.add(self)

    @classmethod
    def clear_all_graphs(cls) -> None:
        """Clear captured graphs from all GraphWrapper instances."""
        for instance in list(cls._all_instances):
            instance.clear_graphs()

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: GraphOptions | None = None,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self._txda_graph_disabled_reason = None
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self._runnable_str = str(runnable) if self.is_debugging_mode else None

        # assert runtime_mode is not NONE(no cudagraph), otherwise, we don't
        # need to initialize a GraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        # TODO: in the future, if we want to use multiple
        # streams, it might not be safe to share a global pool.
        # only investigate this when we use multiple streams
        self.graph_pool = current_platform.get_global_graph_pool()
        if current_platform.device_type == "txda" and self.graph_pool is None:
            self.graph_pool = current_platform.graph_pool_handle()

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
        for entry in self.concrete_graph_entries.values():
            if isinstance(entry.graph, _TxdaGraphSequence):
                entry.graph.clear()
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
        if current_platform.device_type == "txda" and self._txda_graph_disabled_reason:
            return self.runnable(*args, **kwargs)
        if batch_descriptor not in self.concrete_graph_entries:
            # create a new entry for this batch descriptor
            self.concrete_graph_entries[batch_descriptor] = GraphEntry(
                batch_descriptor=batch_descriptor
            )

        entry = self.concrete_graph_entries[batch_descriptor]

        if current_platform.device_type == "txda":
            return self._call_txda(entry, args, kwargs)

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
                    stack.enter_context(
                        patch("gc.collect", lambda *args, **kwargs: None)
                    )
                    stack.enter_context(
                        patch(
                            "torch.accelerator.empty_cache",
                            lambda *args, **kwargs: None,
                        )
                    )
                    stack.enter_context(
                        patch(
                            "vllm_fl.platform.PlatformFL.empty_cache",
                            lambda *args, **kwargs: None,
                        )
                    )

                if self.graph_pool is not None:
                    set_graph_pool_id(self.graph_pool)
                else:
                    set_graph_pool_id(current_platform.graph_pool_handle())

                # Sync offloader's copy stream before capture if available.
                with suppress(ImportError, RuntimeError):
                    from vllm.model_executor.offloader.base import get_offloader

                    get_offloader().sync_prev_onload()

                with current_platform.torch_device_fn.graph(
                    graph,
                    pool=self.graph_pool,
                    stream=_graph_capture_stream(),
                ):
                    output = self.runnable(*args, **kwargs)
                    with suppress(ImportError, RuntimeError, UnboundLocalError):
                        get_offloader().join_after_forward()
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
        with suppress(ImportError, RuntimeError):
            from vllm.model_executor.offloader.base import get_offloader

            get_offloader().sync_prev_onload()

        if current_platform.device_type == "npu":
            current_platform.torch_device_fn.synchronize()
        entry.graph.replay()
        return entry.output

    def _call_txda(self, entry, args, kwargs):
        if entry.graph is None:
            valid, reason = _validate_txda_graph_runnable(self.runnable)
            if not valid:
                self._txda_graph_disabled_reason = reason
                logger.warning_once("TXDA region remains eager: %s", reason)
                return self.runnable(*args, **kwargs)
            validate_cudagraph_capturing_enabled()
            sequence = _TxdaGraphSequence()
            try:
                output = sequence.capture(self.runnable, args, kwargs)
            except _TxdaGraphUnsupported as error:
                self._txda_graph_disabled_reason = str(error)
                logger.warning_once("TXDA region remains eager: %s", error)
                return self.runnable(*args, **kwargs)
            entry.graph = sequence
            # Strong outputs and launch arguments are necessary because the
            # sequence uses allocations made outside each short native capture.
            entry.output = output
            compilation_counter.num_cudagraph_captured += len(sequence.graphs)
            logger.debug("TXDA captured %d single-kernel graphs", len(sequence.graphs))
            self._record_txda_capture_scope()
            return output
        try:
            entry.graph.inputs.update(args, kwargs)
        except _TxdaGraphUnsupported as error:
            logger.warning_once(
                "TXDA region runs eagerly for changed inputs: %s", error
            )
            return self.runnable(*args, **kwargs)
        entry.graph.replay()
        self._record_txda_capture_scope()
        return entry.output
