# Copyright (c) 2026 BAAI. All rights reserved.

"""Use GCUGraph while retaining the native V1 runner's wrapper contract."""

from vllm.compilation.cuda_graph import CUDAGraphWrapper

from vllm_fl.compilation.graph import GraphWrapper


class EnflameGraphWrapper(GraphWrapper, CUDAGraphWrapper):
    def __init__(self, runnable, vllm_config, runtime_mode, cudagraph_options=None):
        GraphWrapper.__init__(
            self, runnable, vllm_config, runtime_mode, cudagraph_options
        )
        # Native V1 discovers wrappers through this registry during memory
        # estimation and cleanup. GraphWrapper also tracks piecewise captures.
        CUDAGraphWrapper._all_instances.add(self)

    @property
    def concrete_cudagraph_entries(self):
        return self.concrete_graph_entries

    @concrete_cudagraph_entries.setter
    def concrete_cudagraph_entries(self, value):
        self.concrete_graph_entries = value


def adapt_enflame_graph_wrapper(model):
    if isinstance(model, EnflameGraphWrapper):
        return model
    if isinstance(model, CUDAGraphWrapper):
        return EnflameGraphWrapper(
            adapt_enflame_graph_wrapper(model.runnable),
            model.vllm_config,
            model.runtime_mode,
            model.cudagraph_options,
        )
    return model
