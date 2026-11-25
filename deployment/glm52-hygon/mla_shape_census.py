"""Opt-in diagnostic worker extension: runtime shapes and selected real inputs.
No kernel change. Captured clones add overhead; this run is NOT a performance run.
Only TP rank 0 writes, enabled by <run-root>/census/ENABLE after server readiness.
"""
import dataclasses
import json
import os
from pathlib import Path
import time

import torch
from vllm_fl.compilation.graph import GraphWrapper
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.distributed import get_tensor_model_parallel_rank
from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.glm_sparse import hygon_glm_sparse_mla


class CensusExtension:
    pass


root = Path(os.environ["GLM_MLA_CENSUS_ROOT"])
captures = {}
capture_shapes = {}
saved = set()
active_key = None
original_op = hygon_glm_sparse_mla._init_fn
original_call = GraphWrapper.__call__


def emit(record):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "shapes.jsonl").open("a") as f:
        f.write(json.dumps(dict(time_ns=time.time_ns(), pid=os.getpid(), **record)) + "\n")


def schema(q, kv, indices, scale):
    return dict(q=list(q.shape), q_stride=list(q.stride()), kv=list(kv.shape),
                kv_stride=list(kv.stride()), indices=list(indices.shape),
                indices_stride=list(indices.stride()), dtype=str(q.dtype),
                scale=scale, device=str(q.device))


def save_input(kind, q, kv, indices, scale):
    key = (kind, q.shape[0])
    if key in saved or len(saved) >= 5 or q.shape[0] not in (1, 64, 4096, 5089, 8192):
        return
    saved.add(key)
    flat = kv.reshape(-1, 1, 576)
    ids = indices.flatten().long()
    ids = ids[(ids >= 0) & (ids < flat.shape[0])].unique()
    payload = dict(q=q.cpu(), indices=indices.cpu(), kv_ids=ids.cpu(),
                   kv_values=flat[ids].cpu(), kv_shape=list(flat.shape),
                   schema=schema(q, kv, indices, scale), kind=kind)
    # Preserve physical offsets and selected values, not a compact-cache layout.
    # Reconstruction must allocate kv_shape and scatter kv_values at kv_ids.
    path = root / (kind + "-" + str(q.shape[0]) + ".pt")
    with path.open("xb") as f:
        torch.save(payload, f)
    emit(dict(event="saved", file=path.name, selected_kv=ids.numel(),
              shape=payload["schema"]))


def census_op(q, kv, indices, scale):
    if get_tensor_model_parallel_rank() != 0:
        return original_op(q, kv, indices, scale)
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and active_key is not None:
        capture_shapes.setdefault(active_key, []).append(schema(q, kv, indices, scale))
        if active_key not in captures:
            # Persistent copies are graph nodes, preserving the first layer's
            # real inputs before later layers can reuse temporary memory.
            captures[active_key] = (q.clone(), kv, indices.clone(), scale)
    output = original_op(q, kv, indices, scale)
    if not capturing and (root / "ENABLE").exists():
        emit(dict(event="op", shape=schema(q, kv, indices, scale)))
        save_input("op", q, kv, indices, scale)
    return output


def census_call(self, *args, **kwargs):
    global active_key
    if get_tensor_model_parallel_rank() != 0 or not is_forward_context_available():
        return original_call(self, *args, **kwargs)
    ctx = get_forward_context()
    if ctx.cudagraph_runtime_mode != self.runtime_mode or ctx.batch_descriptor is None:
        return original_call(self, *args, **kwargs)
    descriptor = ctx.batch_descriptor
    key = (id(self), descriptor)
    entry = self.concrete_graph_entries.get(descriptor)
    replay = entry is not None and entry.graph is not None
    old_key = active_key
    active_key = None if replay else key
    try:
        output = original_call(self, *args, **kwargs)
    finally:
        active_key = old_key
    if replay and (root / "ENABLE").exists():
        emit(dict(event="graph_replay", mode=str(self.runtime_mode),
                  batch=dataclasses.asdict(descriptor),
                  op_shapes=capture_shapes.get(key, [])))
        if key in captures:
            save_input("graph", *captures[key])
    return output


hygon_glm_sparse_mla.register_kernel("cuda", census_op)
GraphWrapper.__call__ = census_call
