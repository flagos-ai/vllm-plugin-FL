# Copyright (c) 2025 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/compilation/cuda_graph.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import weakref
from collections.abc import Callable
from contextlib import ExitStack, suppress
from typing import Any, ClassVar
from unittest.mock import patch

import torch
from torch.utils._pytree import tree_flatten

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
    if current_platform.device_type in ("musa", "npu"):
        # Native accelerator contexts do not update vLLM's CUDA stream cache.
        # Read the active accelerator side stream directly during capture.
        return current_platform.torch_device_fn.current_stream()
    return current_stream()


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
        graph = None
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

    # Keep MUSA PIECEWISE inputs alive and stage fresh inter-piece allocations.
    musa_input_leaves: list[Any] | None = None
    musa_input_spec: Any | None = None
    musa_input_metadata: list[Any] | None = None


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

    @staticmethod
    def _musa_tensor_metadata(tensor):
        return (
            tensor.shape,
            tensor.stride(),
            tensor.dtype,
            tensor.device,
            tensor.layout,
            tensor.storage_offset(),
            tensor.requires_grad,
        )

    def _stage_musa_inputs(self, entry, args, kwargs):
        def require(condition, message):
            if not condition:
                raise ValueError(message)

        leaves, spec = tree_flatten((args, kwargs))
        require(spec == entry.musa_input_spec, "MUSA graph input structure changed")
        require(
            len(leaves) == len(entry.musa_input_leaves),
            "MUSA graph input count changed",
        )
        copies = []
        captured_tensors = []
        incoming_tensors = []
        incoming_by_destination = {}
        for captured, incoming, metadata in zip(
            entry.musa_input_leaves, leaves, entry.musa_input_metadata
        ):
            if isinstance(captured, torch.Tensor):
                require(
                    isinstance(incoming, torch.Tensor),
                    "MUSA graph tensor input type changed",
                )
                require(
                    self._musa_tensor_metadata(captured) == metadata,
                    "Captured MUSA graph input metadata was mutated",
                )
                require(
                    self._musa_tensor_metadata(incoming) == metadata,
                    "MUSA graph input tensor metadata changed",
                )
                captured_tensors.append(captured)
                incoming_tensors.append(incoming)
                previous = incoming_by_destination.setdefault(id(captured), incoming)
                require(
                    previous.data_ptr() == incoming.data_ptr(),
                    "A repeated MUSA graph input changed its alias relationship",
                )
                if captured.data_ptr() != incoming.data_ptr():
                    require(
                        not torch._C._is_alias_of(captured, incoming),
                        "MUSA graph staging source overlaps its destination",
                    )
                    require(
                        captured.device.type == "musa",
                        "A captured host tensor address changed",
                    )
                    copies.append((captured, incoming))
            else:
                require(
                    type(incoming) is type(captured)
                    and (incoming is captured or incoming == captured),
                    "MUSA graph non-tensor input changed",
                )

        # Validate the complete call before writing any captured storage.
        for index, captured in enumerate(captured_tensors):
            for other_index in range(index):
                require(
                    torch._C._is_alias_of(captured, captured_tensors[other_index])
                    == torch._C._is_alias_of(
                        incoming_tensors[index], incoming_tensors[other_index]
                    ),
                    "MUSA graph input tensor alias relationship changed",
                )
        # Distinct views or cross-input aliases require a separate staging policy.
        for captured, incoming in copies:
            for other in captured_tensors:
                if other is captured:
                    continue
                require(
                    not torch._C._is_alias_of(captured, other),
                    "MUSA graph changed input aliases another captured input",
                )
                require(
                    not torch._C._is_alias_of(incoming, other),
                    "MUSA graph input aliases another staging destination",
                )
        copied = set()
        for captured, incoming in copies:
            identity = id(captured)
            if identity in copied:
                continue
            captured.copy_(incoming)
            copied.add(identity)

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
            musa_inputs = None
            if (
                current_platform.device_type == "musa"
                and self.runtime_mode == CUDAGraphMode.PIECEWISE
            ):
                leaves, spec = tree_flatten((args, kwargs))
                metadata = [
                    self._musa_tensor_metadata(x)
                    if isinstance(x, torch.Tensor)
                    else None
                    for x in leaves
                ]
                musa_inputs = (leaves, spec, metadata)
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

                # FL-specific: use platform-agnostic graph capture while
                # preserving the explicit stream used by vLLM v0.28.0.
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
            if musa_inputs is not None:
                (
                    entry.musa_input_leaves,
                    entry.musa_input_spec,
                    entry.musa_input_metadata,
                ) = musa_inputs

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during graph capture
            return output

        if entry.musa_input_leaves is not None:
            self._stage_musa_inputs(entry, args, kwargs)

        if self.is_debugging_mode and entry.musa_input_leaves is None:
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
