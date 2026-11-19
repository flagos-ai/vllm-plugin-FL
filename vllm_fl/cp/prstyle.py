# SPDX-License-Identifier: Apache-2.0
"""PR-style collocated sequence parallelism for Qwen3.5/3.6.

Prefill owns zigzag sequence rows at the layer boundary. Full attention keeps
the query local, reconstructs full K/V, and returns local rows. GDN runs its
native TP heads over a gathered sequence and reduce-scatters its partial
output. Decode uses the unmodified native TP path.
"""

from dataclasses import replace
from functools import lru_cache
from itertools import islice
from typing import Any

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.distributed import get_pp_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import unified_kv_cache_update
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)

from .layout import PagedKVPlan, ZigZagPlan
from .fused_norm import prefill_gemma_norm
from .fused_moe import _eligible, try_prefill_moe
from .fused_gdn import try_prefill_gdn
from . import gdn_sp

logger = init_logger(__name__)


def flash_attn_varlen_func(**kwargs):
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func as kernel,
    )

    return kernel(**kwargs)


def _context_metadata(context) -> Any | None:
    """Return the shared full-attention metadata used to build row plans."""
    raw = context.attn_metadata
    if raw is None or isinstance(raw, list):
        return None
    for metadata in raw.values():
        if (hasattr(metadata, "query_start_loc")
                and hasattr(metadata, "seq_lens")
                and hasattr(metadata, "block_table")):
            return metadata
    return None


def _is_sequence_prefill(metadata: Any | None) -> bool:
    if metadata is None:
        return False
    # Single-token batches do not need a device-to-host metadata copy.
    if metadata.max_query_len <= 1:
        return False
    context = get_forward_context()
    world_size = get_tp_group().world_size
    key = ("fl_prstyle_prefill", id(metadata), world_size)
    cached = context.additional_kwargs.get(key)
    if cached is not None:
        return cached
    starts = _query_starts(context, metadata)
    lengths = [b - a for a, b in zip(starts, starts[1:])]
    result = bool(lengths) and all(length >= 2 * world_size for length in lengths)
    if not result and lengths and all(length > 0 for length in lengths):
        # PD producers also schedule tiny prefill chunks: for hybrid models
        # the connector leaves the last prompt token to D, so a 64K budget
        # can contain [65535, 1]. Query length alone cannot identify decode.
        # All tokens still pass through native GDN in original sequence order;
        # only the stateless layer boundary is sequence sharded.
        result = any(getattr(layer, "_fl_prstyle_kv_producer", False)
                     for layer in context.no_compile_layers.values())
    context.additional_kwargs[key] = result
    return result


def _query_starts(context, metadata):
    key = ("fl_prstyle_query_starts", id(metadata))
    if key not in context.additional_kwargs:
        context.additional_kwargs[key] = tuple(
            metadata.query_start_loc.cpu().tolist()
        )
    return context.additional_kwargs[key]


@lru_cache(maxsize=8)
def _build_plans(starts, lengths, world_size):
    # Bounded reuse across forwards. Prefix lengths participate in the key,
    # so chunked prefill gets the correct causal metadata for each chunk.
    return ZigZagPlan.build_all(starts, lengths, world_size)


def _plans(context, metadata, group):
    key = ("fl_prstyle_plans", id(metadata), group.world_size)
    cached = context.additional_kwargs.get(key)
    if cached is None:
        starts = _query_starts(context, metadata)
        lengths = tuple(metadata.seq_lens.cpu().tolist())
        cached = _build_plans(starts, lengths, group.world_size)
        context.additional_kwargs[key] = cached
    return cached


def _select_positions(positions: torch.Tensor, plan: ZigZagPlan,
                      num_tokens: int) -> torch.Tensor:
    if positions.ndim == 1:
        return plan.select(positions)
    # Qwen supplies 3-D rotary positions as [3, num_tokens].  The metadata
    # token count can include an implementation-specific padded count, so use
    # the plan's packed row count when identifying the token dimension.
    if positions.shape[-1] == plan.num_tokens:
        return plan.select(positions, dim=-1)
    if positions.shape[0] == plan.num_tokens:
        return plan.select(positions, dim=0)
    raise RuntimeError(
        f"Cannot find token dimension in positions shape={tuple(positions.shape)} "
        f"plan_tokens={plan.num_tokens} metadata_tokens={num_tokens}"
    )


def _rank_major_partial(full: torch.Tensor, plans: tuple[ZigZagPlan, ...],
                        group) -> torch.Tensor:
    """Pack full-sequence TP partials for GroupCoordinator.reduce_scatter."""
    padded_rows = plans[0].padded_rows
    lengths = tuple(len(plan.rows) for plan in plans)
    source_rows = plans[0].rank_major_rows_tensor(plans, full.device)
    if all(length == padded_rows for length in lengths):
        # The common single-request case is already dense.  One gather kernel
        # produces the exact reduce-scatter layout without an intermediate copy.
        packed = full.index_select(0, source_rows)
    else:
        # Ragged requests still need zero padding, but use one gather and one
        # indexed write instead of one gather/copy pair per owner rank.
        selected = full.index_select(0, source_rows)
        packed = full.new_zeros((padded_rows * group.world_size, *full.shape[1:]))
        destinations = plans[0].rank_major_dest_tensor(plans, full.device)
        packed.index_copy_(0, destinations, selected)
    local = group.reduce_scatter(packed, dim=0)
    local_plan = plans[group.rank_in_group]
    return local[:len(local_plan.rows)]


def _flagcx_comm(group):
    """Return the native stream-capable communicator used by PCP, if present."""
    device_comm = getattr(group, "device_communicator", None)
    # Production runs currently use PyNCCL; FlagCX is used by deployments
    # that replace the device communicator.  Both expose the same stream API.
    for name in ("pynccl_comm", "pyflagcx_comm"):
        comm = getattr(device_comm, name, None)
        if (comm is not None and not getattr(comm, "disabled", True)
                and all(hasattr(comm, op) for op in
                        ("all_gather", "reduce_scatter"))):
            return comm
    return None


def _overlap_pr_moe(
    mlp: Qwen3NextSparseMoeBlock,
    local_input: torch.Tensor,
    plan: ZigZagPlan,
    group,
    chunk_rows: int = 8192,
) -> torch.Tensor | None:
    """Pipeline PCP MoE all-gather, experts, and reduce-scatter.

    Alternate two owned partial buffers so reduction of chunk i can overlap
    expert computation of chunk i+1, including reuse of native MoE workspace.
    Wait for reduction before reusing the same partial buffer at chunk i+2.
    Return ``None`` unless the stream-capable communicator and audited MoE path
    are present for every chunk, including a possibly shorter last chunk.
    """
    if (not plan.uniform_rows or group.world_size <= 1
            or local_input.ndim != 2 or local_input.shape[0] <= chunk_rows
            or not local_input.is_cuda or not _eligible(mlp, local_input)):
        return None
    comm = _flagcx_comm(group)
    if comm is None:
        return None

    local = (local_input
             if local_input.is_contiguous() else local_input.contiguous())
    local_rows = local.shape[0]
    chunks = tuple(
        (start, min(start + chunk_rows, local_rows))
        for start in range(0, local_rows, chunk_rows)
    )
    device = local.device
    compute_stream = torch.cuda.current_stream(device)
    comm_stream = torch.cuda.Stream(device=device)

    # Allocate on the caller's stream and keep tensors alive through the final
    # join. This also orders caching-allocator reuse before communication.
    gathered = [torch.empty(
        ((end - start) * group.world_size, local.shape[1]),
        dtype=local.dtype, device=device,
    ) for start, end in chunks]
    if not all(_eligible(mlp, value) for value in gathered):
        return None
    partial_buffers = [torch.empty_like(value) for value in gathered[:2]]
    output = torch.empty_like(local)

    input_ready = torch.cuda.Event()
    input_ready.record(compute_stream)
    comm_stream.wait_event(input_ready)

    gathered_done: list[torch.cuda.Event] = []
    reduced_done: list[torch.cuda.Event] = []

    def launch_gather(index: int):
        start, end = chunks[index]
        comm.all_gather(gathered[index], local[start:end], stream=comm_stream)
        done = torch.cuda.Event()
        done.record(comm_stream)
        gathered_done.append(done)

    def launch_reduce(partial: torch.Tensor, start: int, end: int):
        # Each rank's slice is contiguous and already in the local row order.
        comm.reduce_scatter(output[start:end], partial, stream=comm_stream)
        done = torch.cuda.Event()
        done.record(comm_stream)
        reduced_done.append(done)

    try:
        # Prefetch the next gather before waiting for this chunk's computation.
        with torch.cuda.stream(comm_stream):
            launch_gather(0)

        for index, (start, end) in enumerate(chunks):
            compute_stream.wait_event(gathered_done[index])
            if index >= 2:
                compute_stream.wait_event(reduced_done[index - 2])
            destination = partial_buffers[index % 2][:gathered[index].shape[0]]
            partial = try_prefill_moe(mlp, gathered[index], out=destination)
            if partial is None:
                # Eligibility was checked before launching any collectives.
                raise RuntimeError("MoE eligibility changed during overlap")
            computed = torch.cuda.Event()
            computed.record(compute_stream)
            with torch.cuda.stream(comm_stream):
                if index + 1 < len(chunks):
                    launch_gather(index + 1)
                comm_stream.wait_event(computed)
                launch_reduce(partial, start, end)
    finally:
        # All allocations and subsequent consumers are on compute_stream.
        # Join even on exceptions before releasing cross-stream tensor storage.
        compute_stream.wait_stream(comm_stream)
    return output


def _pr_moe(mlp: Qwen3NextSparseMoeBlock, local_input: torch.Tensor,
            plan: ZigZagPlan, plans: tuple[ZigZagPlan, ...], group) -> torch.Tensor:
    runner = mlp.experts.runner
    if runner._fused_output_is_reduced:
        full_input = plan.gather(local_input, group.all_gather)
        return plan.select(mlp(full_input))

    # MoE is token-wise: retain collective order through routing and reduction.
    # Padded rows are independent and discarded after reduce-scatter; they
    # never enter attention or the stateful GDN core.
    overlapped = _overlap_pr_moe(mlp, local_input, plan, group)
    if overlapped is not None:
        return overlapped

    full_input = plan.gather_rank_major(local_input, group.all_gather)
    partial = try_prefill_moe(mlp, full_input)
    if partial is not None:
        return group.reduce_scatter(partial, dim=0)[:len(plan.rows)]

    old_reduce = runner._maybe_reduce_final_output

    def reduce_local_rows(states, trunc_size):
        return group.reduce_scatter(states[..., :trunc_size], dim=0)[:len(plan.rows)]

    runner._maybe_reduce_final_output = reduce_local_rows
    try:
        if mlp.experts.is_internal_router:
            return mlp.experts(hidden_states=full_input,
                               router_logits=full_input)
        router_logits, _ = mlp.gate(full_input)
        return mlp.experts(hidden_states=full_input,
                           router_logits=router_logits)
    finally:
        runner._maybe_reduce_final_output = old_reduce


def _validate_config():
    config = get_current_vllm_config()
    model = config.model_config
    parallel = config.parallel_config
    hf = model.hf_text_config
    if hf.model_type not in {"qwen3_5_text", "qwen3_5_moe_text"}:
        raise ValueError("PR-style Qwen CP supports Qwen3.5/3.6 only")
    if parallel.tensor_parallel_size < 2:
        raise ValueError("PR-style Qwen CP requires tensor_parallel_size >= 2")
    unsupported = {
        "enforce_eager=False": not model.enforce_eager,
        "pipeline parallel": parallel.pipeline_parallel_size != 1,
        "data parallel": parallel.data_parallel_size != 1,
        "prefix caching": config.cache_config.enable_prefix_caching,
        "quantization": config.quant_config is not None,
        "sequence parallel MoE": getattr(parallel, "use_sequence_parallel_moe", False),
        "expert parallel": getattr(parallel, "enable_expert_parallel", False),
        "DBO": parallel.use_ubatching,
    }
    for feature, enabled in unsupported.items():
        if enabled:
            raise ValueError(f"PR-style Qwen CP does not support {feature}")
    if config.kernel_config.moe_backend != "triton" and hf.model_type == "qwen3_5_moe_text":
        raise ValueError("PR-style Qwen MoE CP requires kernel_config.moe_backend=triton")


class QwenPRStyleAttention(Qwen3NextAttention):
    """Full attention with local query rows and complete head semantics."""

    def __init__(self, config, model_config=None, cache_config=None,
                 quant_config=None, prefix=""):
        _validate_config()
        # MTP receives native replicated token rows, not the target CP layout.
        self._fl_mtp_native = "mtp" in prefix.split(".")
        super().__init__(config, model_config, cache_config, quant_config, prefix)
        if self.attn.get_attn_backend() is not FlashAttentionBackend:
            raise ValueError("PR-style Qwen CP requires FLASH_ATTN")
        # Experimental FA3 comparison. PCP rebuilds virtual-request metadata
        # and passes scheduler_metadata=None, so FA3 schedules these shapes.
        if self.attn.impl.vllm_flash_attn_version not in (2, 3):
            raise ValueError("PR-style Qwen CP requires FlashAttention v2 or v3")
        if quant_config is not None:
            raise ValueError("PR-style Qwen CP weight reconstruction needs BF16 weights")
        transfer = get_current_vllm_config().kv_transfer_config
        self.attn._fl_prstyle_kv_producer = (
            transfer is not None and transfer.kv_role == "kv_producer"
        )
        self.pr_group = get_tp_group()
        self.pr_prefill_calls = 0
        self.pr_prefill_tokens = 0
        self.tp_fallback_calls = 0
        self.tp_fallback_tokens = 0
        self.tp_decode_calls = 0
        self.tp_decode_tokens = 0
        self._full_qkv_weight = None
        self._full_qkv_bias = None
        self._full_o_weight = None

    @torch.no_grad()
    def _ensure_full_weights(self):
        if self._full_qkv_weight is not None:
            return
        group = self.pr_group
        local_qgate = self.num_heads * self.head_dim * (2 if self.attn_output_gate else 1)
        local_kv = self.num_kv_heads * self.head_dim
        local_width = local_qgate + 2 * local_kv
        weight = self.qkv_proj.weight.detach()
        if weight.shape[0] != local_width:
            raise ValueError(
                f"Unexpected QKV shard {tuple(weight.shape)}; expected {local_width} rows"
            )
        chunks = group.all_gather(weight.contiguous(), dim=0).split(local_width, dim=0)
        qgate = torch.cat([chunk[:local_qgate] for chunk in chunks], dim=0)
        if self.total_num_kv_heads >= group.world_size:
            kv_chunks = list(chunks)
        else:
            replicas = group.world_size // self.total_num_kv_heads
            kv_chunks = [chunks[i * replicas] for i in range(self.total_num_kv_heads)]
        key = torch.cat([chunk[local_qgate:local_qgate + local_kv]
                         for chunk in kv_chunks], dim=0)
        value = torch.cat([chunk[local_qgate + local_kv:]
                           for chunk in kv_chunks], dim=0)
        self._full_qkv_weight = torch.cat((qgate, key, value), dim=0).contiguous()

        bias = getattr(self.qkv_proj, "bias", None)
        if bias is not None:
            bias_chunks = group.all_gather(bias.detach().contiguous(), dim=0).split(
                local_width, dim=0
            )
            qbias = torch.cat([chunk[:local_qgate] for chunk in bias_chunks], dim=0)
            if self.total_num_kv_heads >= group.world_size:
                bkv_chunks = list(bias_chunks)
            else:
                replicas = group.world_size // self.total_num_kv_heads
                bkv_chunks = [bias_chunks[i * replicas]
                              for i in range(self.total_num_kv_heads)]
            self._full_qkv_bias = torch.cat(
                (qbias,
                 torch.cat([chunk[local_qgate:local_qgate + local_kv]
                            for chunk in bkv_chunks], dim=0),
                 torch.cat([chunk[local_qgate + local_kv:]
                            for chunk in bkv_chunks], dim=0)),
                dim=0,
            ).contiguous()

        # RowParallelLinear shards the input/head dimension. Rank order matches
        # the reconstructed query-head order above.
        self._full_o_weight = group.all_gather(
            self.o_proj.weight.detach().contiguous(), dim=1
        ).contiguous()

    def _native_kv_head_indices(self) -> torch.Tensor:
        if self.total_num_kv_heads >= self.pr_group.world_size:
            start = self.pr_group.rank_in_group * self.num_kv_heads
            return torch.arange(start, start + self.num_kv_heads,
                                device=self.q_norm.weight.device)
        replicas = self.pr_group.world_size // self.total_num_kv_heads
        head = self.pr_group.rank_in_group // replicas
        return torch.tensor([head], device=self.q_norm.weight.device, dtype=torch.long)

    @maybe_transfer_kv_layer
    def _pr_attention(self, q, k, v, gate, plan, plans, metadata, layer_name):
        # Restore the ordered sequence before writing each rank's native KV
        # heads. For complete requests, also reuse this gather
        # as the attention input instead of gathering heads from native cache.
        if self.total_num_kv_heads >= self.pr_group.world_size:
            head_start = self.pr_group.rank_in_group * self.num_kv_heads
        else:
            replicas = self.pr_group.world_size // self.total_num_kv_heads
            head_start = self.pr_group.rank_in_group // replicas
        cache = self.attn.kv_cache
        block_size = cache.shape[2]
        reuse_full_kv = (
            bool(plan.full_prefill_lengths)
            and k.dtype == v.dtype == cache.dtype == torch.bfloat16
            and self.attn.impl.vllm_flash_attn_version == 3
        )
        if reuse_full_kv:
            # The first sequence gather already contains every K/V head.
            # Restore all heads once and give each request its own range of
            # virtual pages, avoiding a second gather through native TP cache. Native
            # cache writes remain necessary for decode and KV transfer.
            full_k, full_v = plan.gather_native_kv(
                k, v, 0, self.total_num_kv_heads, self.pr_group.all_gather
            )
            native_k = full_k[:, head_start:head_start + self.num_kv_heads]
            native_v = full_v[:, head_start:head_start + self.num_kv_heads]
            unified_kv_cache_update(native_k, native_v, layer_name)
            pages = PagedKVPlan.contiguous_prefills(
                plan.full_prefill_lengths, block_size, allow_partial=True
            )
            full_cache = pages.prefill_kv(
                full_k, full_v, plan.full_prefill_lengths, block_size
            )
        else:
            native_k, native_v = plan.gather_native_kv(
                k, v, head_start, self.num_kv_heads, self.pr_group.all_gather
            )
            unified_kv_cache_update(native_k, native_v, layer_name)
            page_key = ("fl_prstyle_pages", id(metadata), block_size)
            context = get_forward_context()
            pages = context.additional_kwargs.get(page_key)
            if pages is None:
                pages = PagedKVPlan.build(
                    metadata.block_table.cpu().tolist(),
                    metadata.seq_lens.cpu().tolist(),
                    block_size,
                )
                context.additional_kwargs[page_key] = pages
            full_cache = pages.gather(
                cache,
                self.total_num_kv_heads,
                self.pr_group.world_size,
                self.pr_group.all_gather,
            )

        # FA writes every local query row. Empty owners still perform the KV
        # collectives/cache writes above, but have no output values to fill.
        local_output = torch.empty_like(q)
        if plan.rows:
            local_meta = plan.metadata(replace(
                metadata, block_table=pages.table(cache.device)
            ))
            flash_attn_varlen_func(
                q=q,
                k=full_cache[0],
                v=full_cache[1],
                out=local_output,
                cu_seqlens_q=local_meta.query_start_loc,
                max_seqlen_q=local_meta.max_query_len,
                seqused_k=local_meta.seq_lens,
                max_seqlen_k=local_meta.max_seq_len,
                block_table=local_meta.block_table,
                softmax_scale=self.scaling,
                causal=True,
                fa_version=self.attn.impl.vllm_flash_attn_version,
                scheduler_metadata=None,
                num_splits=metadata.max_num_splits,
            )
        # Flatten known head dimensions: reshape(0, -1) is ambiguous when a
        # tiny prefill chunk leaves this zigzag owner with no query rows.
        local_output = local_output.flatten(1)
        if gate is not None:
            local_output = local_output * torch.sigmoid(gate)
        return F.linear(local_output, self._full_o_weight)

    def forward(self, positions, output, hidden_states):
        if self._fl_mtp_native:
            return super().forward(positions, output, hidden_states)
        context = get_forward_context()
        metadata_map = context.attn_metadata
        if metadata_map is None or isinstance(metadata_map, list):
            self.tp_decode_calls += 1
            return super().forward(positions, output, hidden_states)
        metadata = metadata_map[self.attn.layer_name]
        if not isinstance(metadata, FlashAttentionMetadata) or not metadata.causal:
            return super().forward(positions, output, hidden_states)
        if metadata.max_query_len <= 1:
            self.tp_decode_calls += 1
            self.tp_decode_tokens += metadata.num_actual_tokens
            return super().forward(positions, output, hidden_states)

        common = _context_metadata(context)
        if not _is_sequence_prefill(common):
            self.tp_fallback_calls += 1
            self.tp_fallback_tokens += _query_starts(context, common)[-1]
            return super().forward(positions, output, hidden_states)
        plans = _plans(context, common, self.pr_group)
        plan = plans[self.pr_group.rank_in_group]
        self._ensure_full_weights()

        qgate, k, v = F.linear(
            hidden_states, self._full_qkv_weight, self._full_qkv_bias
        ).split([
            self.total_num_heads * self.head_dim * (2 if self.attn_output_gate else 1),
            self.total_num_kv_heads * self.head_dim,
            self.total_num_kv_heads * self.head_dim,
        ], dim=-1)
        if self.attn_output_gate:
            qgate = qgate.view(-1, self.total_num_heads, 2 * self.head_dim)
            q, gate = qgate.chunk(2, dim=-1)
            gate = gate.reshape(-1, self.total_num_heads * self.head_dim)
        else:
            q = qgate.view(-1, self.total_num_heads, self.head_dim)
            gate = None
        k = k.view(-1, self.total_num_kv_heads, self.head_dim)
        v = v.view(-1, self.total_num_kv_heads, self.head_dim)
        q = prefill_gemma_norm(self.q_norm, q)
        k = prefill_gemma_norm(self.k_norm, k)
        q, k = self.rotary_emb(
            positions,
            q.reshape(-1, self.total_num_heads * self.head_dim),
            k.reshape(-1, self.total_num_kv_heads * self.head_dim),
        )
        self.pr_prefill_calls += 1
        self.pr_prefill_tokens += plan.num_tokens
        output[:] = self._pr_attention(
            q.view(-1, self.total_num_heads, self.head_dim),
            k.view(-1, self.total_num_kv_heads, self.head_dim),
            v,
            gate,
            plan,
            plans,
            metadata,
            self.attn.layer_name,
        )


_NATIVE_LAYER_FORWARD = Qwen3NextDecoderLayer.forward


def _tp_gdn(linear_attn, hidden_states, plan, plans, group):
    # GDN retains native TP heads. Its RowParallel output is partial during
    # prefill, then sequence reduce-scatter restores this rank's rows.
    full_hidden = plan.gather(hidden_states, group.all_gather)
    out_proj = linear_attn.out_proj
    output_rows = plan.padded_rank_major_rows_tensor(plans, full_hidden.device)
    old_reduce = out_proj.reduce_results
    out_proj.reduce_results = False
    try:
        partial = try_prefill_gdn(linear_attn, full_hidden, output_rows)
        if partial is None:
            full_output = torch.empty_like(full_hidden)
            linear_attn(hidden_states=full_hidden, output=full_output)
    finally:
        out_proj.reduce_results = old_reduce
    if partial is not None and output_rows is not None:
        return group.reduce_scatter(partial, dim=0)[:len(plan.rows)]
    return _rank_major_partial(
        full_output if partial is None else partial, plans, group
    )


def _pr_layer_forward(self, hidden_states, residual, positions=None, **kwargs):
    if getattr(getattr(self, "self_attn", None), "_fl_mtp_native", False):
        return _NATIVE_LAYER_FORWARD(self, hidden_states, residual, positions, **kwargs)
    context = get_forward_context()
    common = _context_metadata(context)
    if not _is_sequence_prefill(common):
        return _NATIVE_LAYER_FORWARD(self, hidden_states, residual, positions, **kwargs)
    group = get_tp_group()
    plans = _plans(context, common, group)
    plan = plans[group.rank_in_group]

    if residual is None:
        residual = hidden_states
        hidden_states = prefill_gemma_norm(self.input_layernorm, hidden_states)
    else:
        hidden_states, residual = prefill_gemma_norm(
            self.input_layernorm, hidden_states, residual
        )

    if self.layer_type == "linear_attention":
        def tp_gdn():
            return _tp_gdn(self.linear_attn, hidden_states, plan, plans, group)

        sp_plan = gdn_sp.ENABLED and gdn_sp.step_plan(
            context, common, plan.full_prefill_lengths, plan, group
        )
        self_attention_output = None
        if sp_plan:
            self_attention_output = gdn_sp.run(
                self.linear_attn, hidden_states, sp_plan, group, tp_gdn
            )
        if self_attention_output is None:
            self_attention_output = tp_gdn()
    elif self.layer_type == "full_attention":
        self_attention_output = torch.empty_like(hidden_states)
        local_positions = _select_positions(positions, plan, common.num_actual_tokens)
        self.self_attn(
            hidden_states=hidden_states,
            output=self_attention_output,
            positions=local_positions,
        )
    else:
        raise ValueError("Invalid layer_type")

    if self.layer_scale:
        if len(self_attention_output.shape) == 2:
            self_attention_output = self_attention_output * (
                self.attn_layer_scale.to(self_attention_output.dtype)[0] + 1
            )
        else:
            self_attention_output = self_attention_output * (
                self.attn_layer_scale.to(self_attention_output.dtype) + 1
            )

    hidden_states, residual = prefill_gemma_norm(
        self.post_attention_layernorm, self_attention_output, residual
    )
    if isinstance(self.mlp, Qwen3NextSparseMoeBlock):
        hidden_states = _pr_moe(self.mlp, hidden_states, plan, plans, group)
    else:
        # Dense TP MLPs also require matching token rows on every rank.
        full_mlp_input = plan.gather(hidden_states, group.all_gather)
        hidden_states = plan.select(self.mlp(full_mlp_input))

    if self.layer_scale:
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states * (
                self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
            )
        else:
            hidden_states = hidden_states * (
                self.ffn_layer_scale.to(hidden_states.dtype) + 1
            )
    return hidden_states, residual


_NATIVE_MODEL_FORWARD = Qwen3NextModel.forward


def _pr_model_forward(self, input_ids, positions, intermediate_tensors=None,
                      inputs_embeds=None):
    context = get_forward_context()
    common = _context_metadata(context)
    sequence_prefill = _is_sequence_prefill(common) and get_pp_group().world_size == 1
    plan = None
    if sequence_prefill:
        plans = _plans(context, common, get_tp_group())
        plan = plans[get_tp_group().rank_in_group]

    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = None
        if plan is not None:
            hidden_states = plan.select(hidden_states)
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
    for layer_idx, layer in enumerate(
        islice(self.layers, self.start_layer, self.end_layer),
        start=self.start_layer,
    ):
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )
        self._maybe_add_hidden_state(
            aux_hidden_states, layer_idx + 1, hidden_states, residual
        )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
    if plan is not None:
        hidden_states = plan.gather(hidden_states, get_tp_group().all_gather)
        residual = plan.gather(residual, get_tp_group().all_gather)
    if plan is not None:
        hidden_states, _ = prefill_gemma_norm(self.norm, hidden_states, residual)
    else:
        hidden_states, _ = self.norm(hidden_states, residual)
    if aux_hidden_states:
        return hidden_states, aux_hidden_states
    return hidden_states


def patch_model():
    """Install the PR-style implementation in the current vLLM process."""
    from vllm.model_executor.models import qwen3_5
    from vllm.model_executor.models import qwen3_next

    qwen3_5.Qwen3NextAttention = QwenPRStyleAttention
    qwen3_next.Qwen3NextAttention = QwenPRStyleAttention
    qwen3_next.Qwen3NextDecoderLayer.forward = _pr_layer_forward
    qwen3_5.Qwen3NextDecoderLayer.forward = _pr_layer_forward
    qwen3_next.Qwen3NextModel.forward = _pr_model_forward
    qwen3_5.Qwen3NextModel.forward = _pr_model_forward
    from .fla_tune import install as install_fla_tune

    install_fla_tune()
    logger.info_once(
        "FL Qwen PR-style CP enabled: layer sequence shard, local full-attention "
        "query, GDN gather/reduce-scatter, native TP decode"
    )
