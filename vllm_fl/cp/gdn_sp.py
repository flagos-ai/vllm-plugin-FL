# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel GDN prefill for PR-style CP (state-corrected, two passes).

Each rank runs all GDN heads on its own zigzag rows instead of gathering the
whole sequence and computing 1/TP of the heads. Every request is split into
2*TP contiguous blocks; rank r owns blocks r and 2*TP-1-r.

* conv: a block's initial conv state is the last 3 pre-conv rows of the
  preceding block (tiny all-gather of every block's last rows);
* recurrence: the chunked delta rule is linear in its initial state with one
  K x K transition shared by all V rows, S_out = S_in @ T + L. Pass 1 gets
  each block's (L, T) in one fwd_h call (V concatenated with K: u=[u, 0],
  h0=[0; I]); (L, T) are all-gathered, chained per request, and pass 2 reruns
  fwd_h from the true start state, followed by chunk_fwd_o;
* output: gated norm and the full out_proj are local, so no reduce-scatter;
* cache: every rank derives each request's final conv/SSM state and writes its
  own native TP head shard, so decode and KV transfer see the native layout.

Only complete first chunks (no prior GDN state) with >= 2*TP*4 tokens per
request take this path; other steps keep the TP-head path.
"""

import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ENABLED = os.environ.get("VLLM_FL_GDN_SP", "1") == "1"
AUDIT_CALLS = int(os.environ.get("VLLM_FL_GDN_SP_AUDIT", "0"))
LOG_DIR = os.environ.get("VLLM_FL_GDN_SP_LOG")
MIN_BLOCK_ROWS = 4  # the conv halo needs the predecessor block's last 3 rows
CHUNK = 64  # FLA chunk size


class SPPlan:
    """Block bookkeeping shared by every rank (CPU, built once per step shape)."""

    def __init__(self, lengths, world, rank):
        nb = 2 * world
        self.world, self.rank, self.blocks = world, rank, nb
        self.num_requests = n = len(lengths)
        # Local segments in ZigZagPlan's local row order: block r of every
        # request, then block 2w-1-r of every request.
        self.local = []  # (request, block, local_offset, length)
        offset = 0
        for block in (rank, nb - 1 - rank):
            for req, q in enumerate(lengths):
                length = q * (block + 1) // nb - q * block // nb
                self.local.append((req, block, offset, length))
                offset += length
        self.rows = offset
        # All-gather order: rank-major, each rank's local segment order.
        self.gathered = {}
        for r in range(world):
            for i, (req, block) in enumerate(
                    (req, block) for block in (r, nb - 1 - r) for req in range(n)):
                self.gathered[(req, block)] = r * 2 * n + i
        self._tensors = {}

    def tensors(self, device):
        key = str(device)
        if key in self._tensors:
            return self._tensors[key]
        from vllm.model_executor.layers.fla.ops.index import (
            prepare_chunk_indices, prepare_chunk_offsets,
        )
        from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

        n, nb = self.num_requests, self.blocks
        lengths = [s[3] for s in self.local]
        cu_cpu = torch.tensor(np.concatenate(([0], np.cumsum(lengths))), dtype=torch.int32)
        cu = cu_cpu.to(device)
        nums, batch_ptr, offset_ptr = compute_causal_conv1d_metadata(cu_cpu, device=device)
        halo = [o + l - 3 + i for _, _, o, l in self.local for i in range(3)]
        pred = [self.gathered[(req, block - 1)] if block else 0
                for req, block, _, _ in self.local]
        has_init = [block > 0 for _, block, _, _ in self.local]
        t = SimpleNamespace(
            cu=cu,
            chunk_indices=prepare_chunk_indices(cu, CHUNK),
            chunk_offsets=prepare_chunk_offsets(cu, CHUNK),
            conv_meta=SimpleNamespace(nums_dict=nums, batch_ptr=batch_ptr,
                                      token_chunk_offset_ptr=offset_ptr),
            halo_rows=torch.tensor(halo, dtype=torch.long, device=device),
            pred=torch.tensor(pred, dtype=torch.long, device=device),
            has_init=torch.tensor(has_init, dtype=torch.bool, device=device),
            cache_slots=torch.arange(1, len(self.local) + 1, dtype=torch.int32, device=device),
            # gathered segment index for (request, block) in request-major order
            chain=torch.tensor([self.gathered[(q, b)] for q in range(n) for b in range(nb)],
                               dtype=torch.long, device=device),
            local_start=torch.tensor([req * nb + block for req, block, _, _ in self.local],
                                     dtype=torch.long, device=device),
            final_halo=torch.tensor([self.gathered[(q, nb - 1)] for q in range(n)],
                                    dtype=torch.long, device=device),
        )
        self._tensors[key] = t
        return t


_PLANS: dict = {}


def step_plan(context, common, lengths, zigzag_plan, group):
    """This step's SP plan for the local rank, or None to keep the TP path."""
    if context is None:  # SP reads GDN metadata from the forward context
        return None
    key = ("fl_gdn_sp_plan", id(common))
    if key in context.additional_kwargs:
        return context.additional_kwargs[key]
    plan = None
    world = group.world_size
    if (ENABLED and world > 1 and lengths
            and min(lengths) >= 2 * world * MIN_BLOCK_ROWS):
        cache_key = (tuple(lengths), world, group.rank_in_group)
        if cache_key not in _PLANS:
            if len(_PLANS) > 64:
                _PLANS.clear()
            _PLANS[cache_key] = SPPlan(lengths, world, group.rank_in_group)
        plan = _PLANS[cache_key]
        if plan.rows != len(zigzag_plan.rows):
            plan = None  # layout mismatch: never expected for these lengths
    context.additional_kwargs[key] = plan
    _record(group.rank_in_group, kind="step", sp=plan is not None, lengths=list(lengths))
    return plan


def _record(rank, **fields):
    if not LOG_DIR:
        return
    import json
    import time

    with open(os.path.join(LOG_DIR, f"gdn_sp_rank{rank}.jsonl"), "a") as f:
        f.write(json.dumps({"ns": time.perf_counter_ns(), **fields}) + "\n")


def _full_weights(layer, group):
    """Gather native TP shards once into full-head weights in global order."""
    cached = getattr(layer, "_fl_sp_weights", None)
    if cached is not None:
        return cached
    w = group.world_size
    kd, vd, hv = layer.key_dim // w, layer.value_dim // w, layer.num_v_heads // w
    qkvz = group.all_gather(layer.in_proj_qkvz.weight.data, dim=0)
    qkvz = torch.cat([p.flatten(0, 1) for p in
                      qkvz.view(w, -1, qkvz.shape[-1]).split([kd, kd, vd, vd], dim=1)])
    ba = group.all_gather(layer.in_proj_ba.weight.data, dim=0)
    ba = torch.cat([p.flatten(0, 1) for p in ba.view(w, -1, ba.shape[-1]).split([hv, hv], dim=1)])
    conv = group.all_gather(layer.conv1d.weight.data.flatten(1), dim=0)
    conv = torch.cat([p.flatten(0, 1) for p in
                      conv.view(w, -1, conv.shape[-1]).split([kd, kd, vd], dim=1)])
    bias = None
    if layer.conv1d.bias is not None:
        b = group.all_gather(layer.conv1d.bias.data, dim=0).view(w, -1)
        bias = torch.cat([p.flatten() for p in b.split([kd, kd, vd], dim=1)])
    full_kd = layer.key_dim
    shard = torch.cat([torch.arange(group.rank_in_group * kd, (group.rank_in_group + 1) * kd),
                       torch.arange(full_kd + group.rank_in_group * kd,
                                    full_kd + (group.rank_in_group + 1) * kd),
                       torch.arange(2 * full_kd + group.rank_in_group * vd,
                                    2 * full_kd + (group.rank_in_group + 1) * vd)])
    weights = SimpleNamespace(
        qkvz=qkvz.contiguous(), ba=ba.contiguous(), conv=conv.contiguous(), conv_bias=bias,
        A_log=group.all_gather(layer.A_log.data, dim=0).contiguous(),
        dt_bias=group.all_gather(layer.dt_bias.data, dim=0).contiguous(),
        out=group.all_gather(layer.out_proj.weight.data, dim=1).contiguous(),
        conv_shard=shard.to(qkvz.device),
    )
    layer._fl_sp_weights = weights
    return weights


def _eligible_layer(layer, metadata):
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    from .fused_gdn import _eligible

    if not isinstance(metadata, GDNAttentionMetadata):
        return False
    if (metadata.num_decodes or metadata.spec_sequence_masks is not None
            or metadata.non_spec_state_indices_tensor is None
            or metadata.non_spec_state_indices_tensor.ndim != 1):
        return False
    probe = torch.empty((1, layer.hidden_size), device=layer.A_log.device, dtype=torch.bfloat16)
    old = layer.out_proj.reduce_results
    layer.out_proj.reduce_results = False
    try:
        return _eligible(layer, probe, None)
    finally:
        layer.out_proj.reduce_results = old


def forward(layer, hidden_states, plan, group):
    """This rank's GDN output rows (local order), or None if the layer is not eligible."""
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fla.ops import chunk as fla_chunk
    from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o
    from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
    from vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv import (
        fused_post_conv_prep,
    )
    from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril
    from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    from .fused_gdn import strided_gated_norm

    metadata = get_forward_context().attn_metadata[layer.prefix]
    if not _eligible_layer(layer, metadata):
        return None
    W = _full_weights(layer, group)
    device = hidden_states.device
    t = plan.tensors(device)
    n, nb, world, rank = plan.num_requests, plan.blocks, plan.world, plan.rank
    H, Hk, K, V = layer.num_v_heads, layer.num_k_heads, layer.head_k_dim, layer.head_v_dim
    qkv_dim = 2 * layer.key_dim + layer.value_dim
    x = hidden_states.contiguous()
    rows = x.shape[0]

    # Projections on local rows with full weights.
    mixed, z = F.linear(x, W.qkvz).split([qkv_dim, layer.value_dim], dim=-1)
    b, a = F.linear(x, W.ba).chunk(2, dim=-1)

    # Conv: block start states are the predecessor blocks' last 3 pre-conv rows.
    halo = mixed.index_select(0, t.halo_rows).view(len(plan.local), 3, qkv_dim)
    halo_all = group.all_gather(halo.contiguous(), dim=0)
    conv_states = torch.zeros((len(plan.local) + 1, qkv_dim, 3), dtype=mixed.dtype, device=device)
    conv_states[1:] = halo_all.index_select(0, t.pred).transpose(1, 2)
    conv_out = causal_conv1d_fn(
        mixed.transpose(0, 1), W.conv, W.conv_bias, activation=layer.activation,
        conv_states=conv_states, has_initial_state=t.has_init,
        cache_indices=t.cache_slots,  # slot 0 is NULL_BLOCK_ID, which the kernel skips
        query_start_loc=t.cu, metadata=t.conv_meta,
    ).transpose(0, 1)
    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv_out, a=a.contiguous(), b=b.contiguous(), A_log=W.A_log,
        dt_bias=W.dt_bias, num_k_heads=Hk, head_k_dim=K, head_v_dim=V,
        apply_l2norm=True, output_g_exp=False,
    )
    q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    g, beta = g.unsqueeze(0), beta.unsqueeze(0)

    # WY representation per block (identical to chunk_gated_delta_rule_fwd).
    ci, co = t.chunk_indices, t.chunk_offsets
    gc = chunk_local_cumsum(g, chunk_size=CHUNK, cu_seqlens=t.cu, chunk_indices=ci)
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=gc, cu_seqlens=t.cu, chunk_indices=ci,
                                 output_dtype=torch.float32)
    A = solve_tril(A=A, cu_seqlens=t.cu, chunk_indices=ci, output_dtype=k.dtype)
    w_, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=gc, cu_seqlens=t.cu,
                              chunk_indices=ci)

    # Pass 1: zero-start final state L and transition T of every local block.
    u_cat = torch.cat([u, u.new_zeros(*u.shape[:-1], K)], dim=-1)
    h0 = torch.zeros((len(plan.local), H, V + K, K), dtype=torch.float32, device=device)
    h0[:, :, V:] = torch.eye(K, device=device)
    _, _, lt = fla_chunk.chunk_gated_delta_rule_fwd_h(
        k=k, w=w_, u=u_cat, g=gc, initial_state=h0, output_final_state=True,
        save_new_value=False, cu_seqlens=t.cu, chunk_indices=ci, chunk_offsets=co)
    lt_all = group.all_gather(lt.contiguous(), dim=0).index_select(0, t.chain)
    lt_all = lt_all.view(n, nb, H, V + K, K)
    state = torch.zeros((n, H, V, K), dtype=torch.float32, device=device)
    starts = []
    for block in range(nb):
        starts.append(state)
        state = torch.matmul(state, lt_all[:, block, :, V:]) + lt_all[:, block, :, :V]
    starts = torch.stack(starts, dim=1).flatten(0, 1).index_select(0, t.local_start)

    # Pass 2 from the true start states, then outputs.
    h, v_new, _ = fla_chunk.chunk_gated_delta_rule_fwd_h(
        k=k, w=w_, u=u, g=gc, initial_state=starts.contiguous(), output_final_state=False,
        cu_seqlens=t.cu, chunk_indices=ci, chunk_offsets=co)
    core = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=gc, scale=K ** -0.5, cu_seqlens=t.cu,
                       chunk_indices=ci)[0]
    normalized = strided_gated_norm(layer.norm, core, z.view(rows, H, V))
    out = F.linear(normalized.flatten(1), W.out)

    # Native TP-shard cache writes (identical inputs on every rank).
    slots = metadata.non_spec_state_indices_tensor[:n]
    hv = H // world
    ssm_cache = layer.kv_cache[1]
    ssm_cache[slots] = state[:, rank * hv:(rank + 1) * hv].to(ssm_cache.dtype)
    conv_cache = (layer.kv_cache[0] if is_conv_state_dim_first()
                  else layer.kv_cache[0].transpose(-1, -2))
    final_rows = halo_all.index_select(0, t.final_halo).index_select(2, W.conv_shard)
    conv_cache[slots] = final_rows.transpose(1, 2).to(conv_cache.dtype)
    return out


_AUDITED = [0]


def run(layer, hidden_states, plan, group, tp_forward):
    """SP forward; for the first AUDIT_CALLS layers also run TP and log differences."""
    if _AUDITED[0] >= AUDIT_CALLS:
        return forward(layer, hidden_states, plan, group)
    from vllm.forward_context import get_forward_context

    metadata = get_forward_context().attn_metadata[layer.prefix]
    if not _eligible_layer(layer, metadata):
        return None
    _AUDITED[0] += 1
    slots = metadata.non_spec_state_indices_tensor[:plan.num_requests]
    reference = tp_forward()
    conv_ref = layer.kv_cache[0][slots].clone()
    ssm_ref = layer.kv_cache[1][slots].clone()
    result = forward(layer, hidden_states, plan, group)

    def diff(x, y):
        x, y = x.float(), y.float()
        return (x - y).abs().max().item(), ((x - y).norm() / y.norm().clamp_min(1e-12)).item()

    _record(group.rank_in_group, kind="audit", layer=layer.prefix,
            requests=plan.num_requests, out=diff(result, reference),
            conv=diff(layer.kv_cache[0][slots], conv_ref),
            ssm=diff(layer.kv_cache[1][slots], ssm_ref))
    return result
