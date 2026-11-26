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
# Support both newer and legacy vLLM module paths for InductorPass.
try:
    from vllm.compilation.passes.inductor_pass import InductorPass, get_pass_context
except ImportError:  # pragma: no cover
    from vllm.compilation.inductor_pass import InductorPass, get_pass_context  # type: ignore[no-redef]
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
    try:
        from vllm.utils.torch_utils import weak_ref_tensors
        return weak_ref_tensors(tensor)
    except Exception:
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

            # Explicitly bind capture to the active CUDA stream so PyTorch does
            # not switch capture onto an internal side stream and invalidate
            # stream ownership of warmup-allocated scratch buffers. Non-CUDA
            # platforms keep their native `graph(graph, pool=...)` signature.
            capture_kwargs = {"pool": self.graph_pool}
            if current_platform.device_type == "cuda":
                capture_kwargs["stream"] = (
                    current_platform.torch_device_fn.current_stream()
                )
            with current_platform.torch_device_fn.graph(graph, **capture_kwargs):
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

        if current_platform.device_type == "npu":
            current_platform.torch_device_fn.synchronize()
        entry.graph.replay()
        return entry.output


class CompositePostGradPass(InductorPass):
    """Sequential composition of orthogonal post-grad FX `InductorPass` passes.

    Inherits from vLLM's `InductorPass` so it satisfies `PostGradPassManager`
    type checks, participates in TorchInductor's persistent cache keying via
    `uuid()`, and forwards range-specific applicability checks (`compile_range`)
    to each child pass independently. Passes execute in registration order (for
    example, `QuantFusionPass` may rewrite attention outputs into
    `fl_unified_attention_quant` before `RopeKvFusionPass` fuses the upstream
    RoPE + KV-cache update).
    """

    def __init__(self, passes: tuple[InductorPass, ...] | list[InductorPass]) -> None:
        """Validate and store the ordered sequence of child `InductorPass` instances."""
        self.passes: tuple[InductorPass, ...] = tuple(passes)
        for p in self.passes:
            if not isinstance(p, InductorPass):
                raise TypeError(f"CompositePostGradPass expects InductorPass instances, got {type(p)!r}")

    def uuid(self) -> str:
        """Return a deterministic cache digest combining composite source and child pass UUIDs."""
        return InductorPass.hash_dict({
            "composite": InductorPass.hash_source(self),
            "passes": [p.uuid() for p in self.passes],
        })

    def is_applicable_for_range(self, compile_range: Any) -> bool:
        """Return True if at least one child pass applies to `compile_range`."""
        return any(p.is_applicable_for_range(compile_range) for p in self.passes)

    def __call__(self, graph: torch.fx.Graph) -> None:
        """Run each applicable child pass in order on `graph`."""
        try:
            compile_range = get_pass_context().compile_range
        except AssertionError:
            # Outside an active vLLM PassContext (e.g., standalone unit tests),
            # treat all passes as applicable.
            compile_range = None
        for p in self.passes:
            if compile_range is None or p.is_applicable_for_range(compile_range):
                p(graph)


def register_post_grad_passes(vllm_config: VllmConfig, *passes: InductorPass) -> None:
    """Register one or more post-grad FX passes onto `vllm_config.compilation_config`.

    Installs a single `InductorPass` directly or wraps multiple passes in a
    `CompositePostGradPass` under `inductor_compile_config["post_grad_custom_post_pass"]`.
    When `VLLM_COMPILE` is paired with a monolithic CUDA Graph mode (i.e., no
    piecewise CUDA graph splitting is requested) and `splitting_ops` was left
    unset (`None`), initializes `splitting_ops = []` so vLLM keeps the full
    model forward in a single FX graph, enabling cross-block fusions across
    attention and normalization boundaries.
    """
    from vllm.config.compilation import CompilationMode

    cc = getattr(vllm_config, "compilation_config", None)
    if cc is None or cc.mode != CompilationMode.VLLM_COMPILE or not passes:
        return
    cc.inductor_compile_config["post_grad_custom_post_pass"] = (
        passes[0] if len(passes) == 1 else CompositePostGradPass(passes)
    )
    if (
        cc.splitting_ops is None
        and cc.cudagraph_mode is not None
        and not cc.cudagraph_mode.has_piecewise_cudagraphs()
    ):
        cc.splitting_ops = []
