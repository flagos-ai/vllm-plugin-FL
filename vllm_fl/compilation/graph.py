import dataclasses
from contextlib import ExitStack
from typing import Any, Callable, Optional
from unittest.mock import patch

import torch

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.compilation.monitor import validate_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id)
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def weak_ref_tensors(tensor: Any) -> Any:
    if current_platform.device_type == "cuda":
        from vllm.utils import weak_ref_tensors
        return weak_ref_tensors(tensor)
    else:
        ### TODO: add csrc npu custom op
        return tensor


class Graph:
    if current_platform.device_type == "cuda":
        graph = torch.cuda.CUDAGraph
    elif current_platform.device_type == "npu":
        graph = torch.npu.NPUGraph
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


class GraphWrapper:
    def __init__(self,
                 runnable: Callable,
                 vllm_config: VllmConfig,
                 runtime_mode: CUDAGraphMode,
                 graph_pool: Any = None,
                 cudagraph_options: Optional[CUDAGraphOptions] = None):
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
            cudagraph_options = CUDAGraphOptions()
        self.graph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # cudagraphs for.
        self.concrete_graph_entries: dict[BatchDescriptor, GraphEntry]\
                                                                        = {}
        
    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not exists in the runnable of "
                             f"cudagraph wrapper: {self.runnable}")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        graph_runtime_mode = forward_context.cudagraph_runtime_mode

        if graph_runtime_mode == CUDAGraphMode.NONE or \
                            graph_runtime_mode != self.runtime_mode:
            # CUDAGraphMode.NONE could mean the profile run, a warmup run, or
            # running without cudagraphs.
            # We do not trigger capture/replay if the runtime mode is not
            # matches. This enables properly dispatching to the correct
            # CUDAGraphWrapper when nesting multiple instances with different
            # runtime modes.
            return self.runnable(*args, **kwargs)
        
        if batch_descriptor not in self.concrete_graph_entries:
            # create a new entry for this batch descriptor
            self.concrete_graph_entries[batch_descriptor] = \
                GraphEntry(batch_descriptor=batch_descriptor)
            
        entry = self.concrete_graph_entries[batch_descriptor]

        if entry.graph is None:
            if self.graph_options.debug_log_enable:
                # Since we capture cudagraph for many different shapes and
                # capturing is fast, we don't need to log it for every
                # shape. E.g. we only log it for the first subgraph in
                # piecewise mode.
                logger.debug("Capturing a cudagraph on (%s,%s)",
                             self.runtime_mode.name, entry.batch_descriptor)
            # validate that cudagraph capturing is legal at this point.
            validate_cudagraph_capturing_enabled()

            input_addresses = [
                x.data_ptr() for x in args if isinstance(x, torch.Tensor)
            ]
            entry.input_addresses = input_addresses
            graph = Graph.graph()

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
                        patch("vllm_fl.platform.PlatformFL.empty_cache", lambda: None))

            set_graph_pool_id(self.graph_pool)

            # mind-exploding: carefully manage the reference and memory.
            with current_platform.torch_device_fn.graph(graph, pool=self.graph_pool):
                # `output` is managed by pytorch's cudagraph pool
                output = self.runnable(*args, **kwargs)

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
                f"got {new_input_addresses}")

        current_platform.torch_device_fn.synchronize()
        entry.graph.replay()
        return entry.output


