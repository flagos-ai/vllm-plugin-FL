"""MetaX execution adapters; native vLLM defaults remain unchanged."""

import torch
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager


class MetaXModelCudaGraphManager(ModelCudaGraphManager):
    def run_fullgraph(self, desc):
        result = super().run_fullgraph(desc)
        # MCCL requires completion before following eager collectives.
        # Identical stream boundary to the qualified runtime workaround.
        torch.cuda.current_stream().synchronize()
        return result
