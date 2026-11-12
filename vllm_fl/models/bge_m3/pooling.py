# Copyright (c) 2026 BAAI. All rights reserved.
"""BGE-M3 selected heads over one encoder output, using vLLM's tensor transport."""

import torch
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.layers.pooler.tokwise import AllPool


class BgeM3JointPooler(Pooler):
    def __init__(self, native_poolers):
        super().__init__()
        self.pooling = AllPool()
        self.dense_head = native_poolers["embed"].head
        self.sparse_head = native_poolers["token_classify"].pooler.head
        self.colbert_head = native_poolers["token_embed"].pooler.head

    def get_supported_tasks(self):
        return {"plugin"}

    def get_pooling_updates(self, task):
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(self, hidden_states, pooling_metadata):
        chunks = self.pooling(hidden_states, pooling_metadata)
        outputs = []
        for i, (hidden, params) in enumerate(
            zip(chunks, pooling_metadata.pooling_params)
        ):
            if hidden is None:
                outputs.append(None)
                continue
            selection = (params.extra_kwargs or {}).get("bge_m3", {})
            mask = selection.get("outputs", 1)
            single = selection.get("single_input", False)
            if type(mask) is not int or not 0 <= mask <= 7:
                raise ValueError("BGE-M3 output mask must be an integer in [0, 7]")
            # The two-value internal header makes postprocessing independent of
            # request ordering or shared mutable request queues. It is not public.
            parts = [hidden.new_tensor([mask, int(single)])]
            if mask & 1:
                dense = self.dense_head([hidden[0]], pooling_metadata[i : i + 1])[0]
                parts.append(dense.reshape(-1))
            if mask & 2:
                # Preserve every input position; official ID filtering/max is
                # done on CPU after the runner copies the result to the host.
                sparse = self.sparse_head.forward_chunk(hidden, params)
                parts.append(sparse.reshape(-1))
            if mask & 4:
                colbert = self.colbert_head.forward_chunk(hidden, params)[1:]
                parts.append(colbert.reshape(-1))
            outputs.append(torch.cat(parts))
        return outputs
