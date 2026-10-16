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


def _weak_ref_musa_tensors(value: Any) -> Any:
    """Borrow captured outputs while their native graph pool owns the memory.

    These descriptors do not own allocations. Ordinary captured inputs remain
    strongly held by GraphEntry; this helper must not weaken those inputs.
    TorchMUSA 2.9 uses the same storage/metadata APIs in musagraph_trees.py.
    """
    tensors = {}
    storages = {}

    def convert(item):
        if isinstance(item, torch.Tensor):
            if item.device.type != "musa":
                return item
            if (
                item.layout != torch.strided
                or item.is_quantized
                or item.requires_grad
                or item.is_conj()
                or item.is_neg()
            ):
                raise ValueError(
                    "MUSA graph outputs require plain strided inference tensors"
                )
            identity = id(item)
            if identity in tensors:
                return tensors[identity]
            source = item.untyped_storage()
            if source.nbytes() == 0:
                return item
            if source.data_ptr() == 0:
                raise ValueError("Nonempty MUSA graph storage has a null address")
            import torch_musa

            native = getattr(
                getattr(torch_musa, "_MUSAC", None),
                "_construct_MUSA_Tensor_From_Storage_And_Metadata",
                None,
            )
            if not callable(native):
                raise RuntimeError(
                    "MUSA graph outputs require TorchMUSA's metadata tensor constructor"
                )
            key = source._cdata
            if key not in storages:
                storages[key] = torch._C._construct_storage_from_data_pointer(
                    source.data_ptr(), item.device, source.nbytes()
                )
            metadata = {
                "size": item.size(),
                "stride": item.stride(),
                "dtype": item.dtype,
                "device": item.device,
                "storage_offset": item.storage_offset(),
            }
            result = native(metadata, storages[key])
            tensors[identity] = result
            return result
        if isinstance(item, list):
            return [convert(child) for child in item]
        if isinstance(item, tuple):
            return tuple(convert(child) for child in item)
        if isinstance(item, dict):
            return {key: convert(child) for key, child in item.items()}

        from vllm.sequence import IntermediateTensors

        if isinstance(item, IntermediateTensors):
            return IntermediateTensors(
                {key: convert(child) for key, child in item.tensors.items()}
            )
        return item

    return convert(value)


def weak_ref_tensors(value: Any) -> Any:
    if current_platform.device_type == "musa":
        return _weak_ref_musa_tensors(value)

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
        from vllm_fl.compilation.musa_graph_exec_cache import musa_graph_class

        graph = musa_graph_class(torch.musa.MUSAGraph)
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

    # Keep MUSA PIECEWISE capture inputs alive for stable-buffer replay.
    musa_input_leaves: list[Any] | None = None
    musa_input_spec: Any | None = None
    musa_input_metadata: list[Any] | None = None
    musa_input_addresses: list[int | None] | None = None
    musa_last_input_tensor_refs: list[Any] | None = None


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

    @staticmethod
    def _musa_tensors_alias(left, right):
        if torch._C._is_alias_of(left, right):
            return True
        if left.device != right.device or left.device.type != "musa":
            return False
        # Borrowed descriptors have distinct StorageImpls. Compare complete
        # physical storage spans as well, conservatively retaining the existing
        # prohibition on staging writes across overlapping input allocations.
        left_storage = left.untyped_storage()
        right_storage = right.untyped_storage()
        left_start = left_storage.data_ptr()
        right_start = right_storage.data_ptr()
        left_size = left_storage.nbytes()
        right_size = right_storage.nbytes()
        return (
            left_size > 0
            and right_size > 0
            and left_start < right_start + right_size
            and right_start < left_start + left_size
        )

    def _stage_musa_inputs(self, entry, args, kwargs):
        """Validate the call and report whether captured buffers can be replayed.

        An arbitrary runnable may mutate inputs or return aliases of them.
        Replaying after copying a fresh allocation into captured storage would
        hide those mutations from the caller and return aliases of the wrong
        allocation. Without a functionalization contract, changed addresses
        therefore execute the runnable with its original incoming arguments.
        """

        def require(condition, message):
            if not condition:
                raise ValueError(message)

        leaves, spec = tree_flatten((args, kwargs))
        require(spec == entry.musa_input_spec, "MUSA graph input structure changed")
        require(
            len(leaves) == len(entry.musa_input_leaves),
            "MUSA graph input count changed",
        )
        stable_inputs = True
        distinct_empty_input = False
        copies = []
        captured_tensors = []
        incoming_tensors = []
        incoming_by_destination = {}
        for captured, incoming, metadata, address, previous_ref in zip(
            entry.musa_input_leaves,
            leaves,
            entry.musa_input_metadata,
            entry.musa_input_addresses,
            entry.musa_last_input_tensor_refs,
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
                if incoming is not captured:
                    require(
                        self._musa_tensor_metadata(incoming) == metadata,
                        "MUSA graph input tensor metadata changed",
                    )
                captured_address = captured.data_ptr()
                incoming_address = incoming.data_ptr()
                require(
                    captured_address == address,
                    "Captured MUSA graph input address changed",
                )
                # Empty tensors all report address zero, even when they have
                # unrelated StorageImpls. Only the original captured object has
                # a known identity/alias contract in that case.
                distinct_empty = incoming is not captured and incoming_address == 0
                distinct_empty_input = distinct_empty_input or distinct_empty
                stable_inputs = (
                    stable_inputs
                    and not distinct_empty
                    and incoming_address == address
                    and (
                        incoming is captured
                        or (previous_ref is not None and previous_ref() is incoming)
                    )
                )
                captured_tensors.append(captured)
                incoming_tensors.append(incoming)
                previous = incoming_by_destination.setdefault(id(captured), incoming)
                require(
                    previous.data_ptr() == incoming.data_ptr(),
                    "A repeated MUSA graph input changed its alias relationship",
                )
                if captured_address != incoming_address:
                    require(
                        not self._musa_tensors_alias(captured, incoming),
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

        # Unchanged objects, addresses and metadata preserve the alias facts
        # already checked at capture or the last stable-buffer call. Keep this
        # normal path linear, without storage-span queries or pairwise scans.
        if stable_inputs:
            return True

        # Validate the complete changed call before selecting graph or eager.
        for index, captured in enumerate(captured_tensors):
            for other_index in range(index):
                require(
                    self._musa_tensors_alias(captured, captured_tensors[other_index])
                    == self._musa_tensors_alias(
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
                    not self._musa_tensors_alias(captured, other),
                    "MUSA graph changed input aliases another captured input",
                )
                require(
                    not self._musa_tensors_alias(incoming, other),
                    "MUSA graph input aliases another staging destination",
                )
        # Keep all metadata and alias checks above active even for calls that
        # cannot replay. Do not stage or copy mutations back across allocations.
        if copies or distinct_empty_input:
            return False
        # A different descriptor can still refer to the same captured buffer.
        # Remember validated identities weakly, without retaining new owners.
        entry.musa_last_input_tensor_refs = [
            weakref.ref(x) if isinstance(x, torch.Tensor) else None for x in leaves
        ]
        return True

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
                addresses = [
                    x.data_ptr() if isinstance(x, torch.Tensor) else None
                    for x in leaves
                ]
                musa_inputs = (leaves, spec, metadata, addresses)
            graph = Graph.graph()

            with ExitStack() as stack:
                if current_platform.device_type == "musa":
                    from vllm_fl.compilation.musa_graph_exec_cache import (
                        musa_graph_capture,
                    )

                    stack.enter_context(musa_graph_capture(graph))
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
                    entry.musa_input_addresses,
                ) = musa_inputs
                entry.musa_last_input_tensor_refs = [
                    weakref.ref(x) if isinstance(x, torch.Tensor) else None
                    for x in entry.musa_input_leaves
                ]

            compilation_counter.num_cudagraph_captured += 1

            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during graph capture
            return output

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

        if current_platform.device_type == "musa":
            from vllm_fl.compilation.musa_graph_exec_cache import musa_graph_replay

            def prepare_replay():
                # The cache holds residency through validation and enqueue.
                # A nonresident graph never stages inputs or restores an exec.
                if entry.musa_input_leaves is not None and not self._stage_musa_inputs(
                    entry, args, kwargs
                ):
                    return False
                with suppress(ImportError, RuntimeError):
                    from vllm.model_executor.offloader.base import get_offloader

                    get_offloader().sync_prev_onload()
                return True

            if not musa_graph_replay(entry.graph, before_replay=prepare_replay):
                return self.runnable(*args, **kwargs)
            return entry.output

        # Sync offloader before replay if available.
        with suppress(ImportError, RuntimeError):
            from vllm.model_executor.offloader.base import get_offloader

            get_offloader().sync_prev_onload()
        if current_platform.device_type == "npu":
            current_platform.torch_device_fn.synchronize()
        entry.graph.replay()
        return entry.output
