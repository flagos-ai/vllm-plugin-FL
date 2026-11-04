"""Opt-in TP0 MoE census. Preserves backend/precision; timings are invalid.
Records every layer's real routing and bounded first-layer input snapshots.
Graph capture adds persistent input/route copies, replay reads the copies.
"""
import dataclasses
import json
import os
from pathlib import Path
import time

import torch
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.experts import triton_moe
from vllm_fl.dispatch.backends.vendor.hygon.impl.moe.triton_experts import HygonTritonExpertsFL
from vllm_fl.compilation.graph import GraphWrapper


class CensusExtension:
    pass


root = Path(os.environ["GLM_MOE_CENSUS_ROOT"])
original_apply = HygonTritonExpertsFL.apply
original_call = GraphWrapper.__call__
active_key = None
captures = {}
layers = {}
saved = set()


def emit(record):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "shapes.jsonl").open("a") as f:
        f.write(json.dumps(dict(time_ns=time.time_ns(), pid=os.getpid(), **record)) + "\n")


def record_inputs(kind, meta, ids, weights, x, w1, w2, s1, s2):
    ids_cpu = ids.cpu()
    counts = torch.bincount(ids_cpu.flatten().long(), minlength=meta["experts"])
    emit(dict(event="op", kind=kind, shape=meta, expert_counts=counts.tolist()))
    key = (kind, meta["tokens"])
    if meta["layer_index"] != 0 or key in saved or len(saved) >= 6:
        return
    saved.add(key)
    if not (root / "weights.pt").exists():
        with (root / "weights.pt").open("xb") as f:
            torch.save(dict(w1=w1.cpu(), w2=w2.cpu(), s1=s1.cpu(), s2=s2.cpu()), f)
    name = kind + "-" + str(meta["tokens"]) + ".pt"
    with (root / name).open("xb") as f:
        torch.save(dict(x=x.cpu(), topk_ids=ids_cpu, topk_weights=weights.cpu(),
                        schema=meta, kind=kind), f)
    emit(dict(event="saved", file=name, shape=meta))


def census_apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
                 activation, global_num_experts, expert_map, a1q_scale, a2_scale,
                 workspace13, workspace2, expert_tokens_meta,
                 apply_router_weight_on_input):
    args = (output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
            global_num_experts, expert_map, a1q_scale, a2_scale, workspace13,
            workspace2, expert_tokens_meta, apply_router_weight_on_input)
    if get_tensor_model_parallel_rank() != 0:
        return original_apply(self, *args)
    layer_index = layers.setdefault(w1.data_ptr(), len(layers))
    qc = self.quant_config
    config = triton_moe.try_get_optimal_moe_config(
        w1.size(), w2.size(), topk_ids.shape[1], qc.config_name(hidden_states.dtype),
        hidden_states.shape[0], block_shape=self.block_shape)
    meta = dict(layer_index=layer_index, tokens=hidden_states.shape[0],
                experts=w1.shape[0], topk=topk_ids.shape[1],
                x_shape=list(hidden_states.shape), x_stride=list(hidden_states.stride()),
                x_dtype=str(hidden_states.dtype), w1=list(w1.shape), w2=list(w2.shape),
                w_dtype=str(w1.dtype), scale1=list(self.w1_scale.shape),
                scale2=list(self.w2_scale.shape), config=config,
                int8_w8a8=qc.use_int8_w8a8, int8_w8a16=qc.use_int8_w8a16,
                quant_dtype=str(self.quant_dtype), activation=str(activation),
                apply_router_weight_on_input=apply_router_weight_on_input,
                experts_class=type(self).__module__ + "." + type(self).__name__)
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and active_key is not None:
        captures.setdefault(active_key, []).append((
            meta, topk_ids.clone(), topk_weights.clone(),
            hidden_states.clone() if layer_index == 0 else None,
            w1, w2, self.w1_scale, self.w2_scale))
    elif not capturing and (root / "ENABLE").exists():
        record_inputs("eager", meta, topk_ids, topk_weights, hidden_states,
                      w1, w2, self.w1_scale, self.w2_scale)
    return original_apply(self, *args)


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
        rows = captures.get(key, [])
        emit(dict(event="graph_replay", mode=str(self.runtime_mode),
                  batch=dataclasses.asdict(descriptor), layers=len(rows)))
        for row in rows:
            record_inputs("graph", *row)
    return output


HygonTritonExpertsFL.apply = census_apply
GraphWrapper.__call__ = census_call
