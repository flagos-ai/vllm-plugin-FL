"""Conditional sparse attention for a single long prefill request."""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

from vllm_fl import envs

logger = init_logger("vllm.vllm_fl.native_sparse_prefill")

_SWITCH = "VLLM_FL_NATIVE_SPARSE_PREFILL"
_PAGE = 16
_PHYSICAL_PAGE = 17
_Q_HEADS = 16
_KV_HEADS = 2
_HEAD_DIM = 128
_GROUP_TOKENS = 64
_DENSE_TAIL = 64
_MIN_HISTORY = 8192
_BUDGET_TOKENS = 2048
_BUDGET_PAGES = _BUDGET_TOKENS // _PAGE


@triton.jit
def _refresh_full_pages(
    K,
    KMin,
    KMax,
    Table,
    ks0,
    ks1,
    ks2,
    ks3,
    ms0,
    ms1,
    ms2,
    PAGE: tl.constexpr,
    D: tl.constexpr,
):
    page = tl.program_id(0)
    block = tl.load(Table + page).to(tl.int64)
    token = tl.arange(0, PAGE)
    feature = tl.arange(0, 2 * D)
    values = tl.load(
        K + block * ks0 + token[:, None] * ks1 + feature[None, :] * ks3
    ).to(tl.float32)
    lo = tl.min(values, 0).to(tl.bfloat16)
    hi = tl.max(values, 0).to(tl.bfloat16)
    tl.store(KMin + block * ms0 + feature * ms2, lo)
    tl.store(KMax + block * ms0 + feature * ms2, hi)


def refresh_full_pages(
    kcache: torch.Tensor,
    kmin: torch.Tensor,
    kmax: torch.Tensor,
    block_table_row: torch.Tensor,
    num_full_pages: int,
) -> None:
    """Recompute metadata for complete logical pages in one CTA per page."""
    count = int(num_full_pages)
    if count == 0:
        return
    if (
        kcache.dtype != torch.bfloat16
        or kcache.ndim != 4
        or tuple(kcache.shape[1:]) != (_PAGE, _KV_HEADS, _HEAD_DIM)
        or kcache.stride()[1:] != (_KV_HEADS * _HEAD_DIM, _HEAD_DIM, 1)
        or kmin.dtype != torch.bfloat16
        or kmax.dtype != torch.bfloat16
        or tuple(kmin.shape) != (kcache.shape[0], _KV_HEADS, _HEAD_DIM)
        or tuple(kmax.shape) != tuple(kmin.shape)
        or kmin.stride() != kmax.stride()
        or kmin.stride()[0] != kcache.stride(0)
        or kmin.stride()[1:] != (_HEAD_DIM, 1)
        or block_table_row.dtype != torch.int32
        or block_table_row.device != kcache.device
        or block_table_row.ndim != 1
        or block_table_row.stride(0) != 1
        or kmin.device != kcache.device
        or kmax.device != kcache.device
        or count > block_table_row.numel()
    ):
        raise ValueError(
            "native sparse prefill requires the 17-slot BF16 cache and contiguous int32 page row"
        )
    _refresh_full_pages[(count,)](
        kcache,
        kmin,
        kmax,
        block_table_row,
        *kcache.stride(),
        *kmin.stride(),
        PAGE=_PAGE,
        D=_HEAD_DIM,
        num_warps=4,
        num_stages=2,
        pipeline="basic",
    )


def global_query_groups(
    history: int, query_len: int
) -> tuple[list[tuple[int, int]], int]:
    """Split the prefix before its dense tail at global 64-token boundaries."""
    prefix_len = int(query_len) - _DENSE_TAIL
    if prefix_len <= 0:
        return [], 0
    groups = []
    start = 0
    while start < prefix_len:
        next_boundary = ((int(history) + start) // _GROUP_TOKENS + 1) * _GROUP_TOKENS
        end = min(prefix_len, next_boundary - int(history))
        if end <= start:
            raise ValueError("invalid global query group boundary")
        groups.append((start, end))
        start = end
    return groups, prefix_len


@dataclass
class SparsePrefillPlan:
    buffers: PlanBuffers
    num_groups: int
    max_group_q: int
    prefill_start: int
    query_len: int
    prefix_tokens: int
    history: int
    seq_len: int


class PlanBuffers:
    """Reusable pinned-host and device buffers for one metadata builder."""

    def __init__(self, device: torch.device | str, max_groups: int, max_pages: int):
        self.device = torch.device(device)
        self.max_groups = int(max_groups)
        self.max_pages = int(max_pages)
        total = 2 * (self.max_groups + 1) + 2 * self.max_groups + 4
        self.cpu = torch.empty((total,), dtype=torch.int32, pin_memory=True)
        self.device_buffer = torch.empty(
            (total,), dtype=torch.int32, device=self.device
        )
        cursor = 0
        self.cu_q_cpu = self.cpu[cursor : cursor + self.max_groups + 1]
        self.cu_q = self.device_buffer[cursor : cursor + self.max_groups + 1]
        cursor += self.max_groups + 1
        self.group_end_cpu = self.cpu[cursor : cursor + self.max_groups]
        self.group_end = self.device_buffer[cursor : cursor + self.max_groups]
        cursor += self.max_groups
        self.representative_cpu = self.cpu[cursor : cursor + self.max_groups]
        self.representative = self.device_buffer[cursor : cursor + self.max_groups]
        cursor += self.max_groups
        self.compact_cu_k_cpu = self.cpu[cursor : cursor + self.max_groups + 1]
        self.compact_cu_k = self.device_buffer[cursor : cursor + self.max_groups + 1]
        cursor += self.max_groups + 1
        self.tail_cu_q_cpu = self.cpu[cursor : cursor + 2]
        self.tail_cu_q = self.device_buffer[cursor : cursor + 2]
        self.tail_cu_k_cpu = self.cpu[cursor + 2 : cursor + 4]
        self.tail_cu_k = self.device_buffer[cursor + 2 : cursor + 4]
        self.tail_cu_q_cpu[:] = torch.tensor((0, _DENSE_TAIL), dtype=torch.int32)
        self.representative_q = torch.empty(
            (self.max_groups, _Q_HEADS, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=self.device,
        )

    def prepare(
        self,
        *,
        req_count: int,
        num_prefills: int,
        num_decodes: int,
        num_decode_tokens: int,
        query_start_loc_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        causal: bool,
        use_cascade: bool,
        dcp_world_size: int,
        dense_limit: int,
        block_table_width: int,
    ) -> SparsePrefillPlan | None:
        if (
            req_count <= 0
            or num_prefills != 1
            or query_start_loc_cpu.device.type != "cpu"
            or seq_lens_cpu.device.type != "cpu"
            or not causal
            or use_cascade
            or dcp_world_size != 1
            or int(block_table_width) != self.max_pages
        ):
            return None
        qstarts = query_start_loc_cpu.numpy()
        seqlens = seq_lens_cpu.numpy()
        row = int(num_decodes)
        if row < 0 or row + 1 >= qstarts.size or row >= seqlens.size:
            return None
        prefill_start = int(qstarts[row])
        if prefill_start != int(num_decode_tokens):
            return None
        query_len = int(qstarts[row + 1]) - prefill_start
        seq_len = int(seqlens[row])
        history = seq_len - query_len
        if query_len <= 128 or history < max(_MIN_HISTORY, int(dense_limit)):
            return None

        groups, prefix_len = global_query_groups(history, query_len)
        if not groups or len(groups) > self.max_groups:
            return None
        lengths = np.fromiter((end - start for start, end in groups), dtype=np.int32)
        np.cumsum(
            lengths, dtype=np.int32, out=self.cu_q_cpu.numpy()[1 : len(groups) + 1]
        )
        self.cu_q_cpu[0] = 0
        self.group_end_cpu.numpy()[: len(groups)] = np.fromiter(
            (history + end for _, end in groups), dtype=np.int32, count=len(groups)
        )
        self.representative_cpu.numpy()[: len(groups)] = np.fromiter(
            (prefill_start + start for start, _ in groups),
            dtype=np.int32,
            count=len(groups),
        )
        group_ends = self.group_end_cpu.numpy()[: len(groups)]
        tails = group_ends % _PAGE
        tails[tails == 0] = _PAGE
        compact_lengths = (_BUDGET_PAGES - 1) * _PAGE + tails
        self.compact_cu_k_cpu[0] = 0
        np.cumsum(
            compact_lengths,
            dtype=np.int32,
            out=self.compact_cu_k_cpu.numpy()[1 : len(groups) + 1],
        )
        self.tail_cu_k_cpu[0] = 0
        self.tail_cu_k_cpu[1] = seq_len

        count = len(groups)
        self.device_buffer.copy_(self.cpu, non_blocking=True)
        return SparsePrefillPlan(
            buffers=self,
            num_groups=count,
            max_group_q=max(end - start for start, end in groups),
            prefill_start=prefill_start,
            query_len=query_len,
            prefix_tokens=prefix_len,
            history=history,
            seq_len=seq_len,
        )


@dataclass
class _Runtime:
    enabled: bool = False
    reason: str = "not installed"
    workspace: object | None = None
    device: torch.device | None = None
    logged: bool = False


_RUNTIME = _Runtime()


def is_enabled() -> bool:
    return _RUNTIME.enabled


def install(vllm_config, device: torch.device, profile: dict, workspace) -> bool:
    """Enable only after a small selected-key startup check passes."""
    global _RUNTIME
    _RUNTIME = _Runtime(reason="switch disabled")
    auto = (
        profile.get("native_sparse_prefill") is True
        and profile.get("native_sparse_decode") is True
    )
    if not envs.switch(_SWITCH, auto=auto):
        return False
    try:
        from vllm_fl.attention import native_sparse_decode as core

        _startup_check(core, device, workspace)
        if not _benchmark_prefill_graphs(core, device, workspace):
            raise RuntimeError(
                "full-chain startup Graph cost gate did not beat dense prefill"
            )
        _RUNTIME = _Runtime(
            enabled=True,
            reason="startup selected-key and full-chain Graph checks passed",
            workspace=workspace,
            device=device,
        )
        logger.info(
            "Native sparse prefill enabled after startup correctness check and full-chain Graph cost gate"
        )
        return True
    except Exception as exc:
        _RUNTIME = _Runtime(reason=f"startup check failed: {type(exc).__name__}: {exc}")
        logger.warning(
            "Native sparse prefill disabled; retaining dense prefill: %s",
            _RUNTIME.reason,
            exc_info=True,
        )
        return False


def make_plan_buffers(
    device: torch.device, max_groups: int, max_pages: int
) -> PlanBuffers | None:
    if not is_enabled() or max_groups <= 0:
        return None
    return PlanBuffers(device, max_groups, max_pages)


def run(
    plan: SparsePrefillPlan,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    kmin: torch.Tensor,
    kmax: torch.Tensor,
    workspace,
    output: torch.Tensor,
    softmax_scale: float,
) -> None:
    """Run grouped sparse-prefix attention and the original dense final 64 queries."""
    from vllm_fl.attention import native_sparse_decode as core
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils.fa_utils import (
        flash_attn_varlen_func,
    )

    buffers = plan.buffers
    groups = plan.num_groups
    refresh_full_pages(key_cache, kmin, kmax, block_table_row, plan.seq_len // _PAGE)
    torch.index_select(
        query, 0, buffers.representative[:groups], out=buffers.representative_q[:groups]
    )
    grouped_table = block_table_row.view(1, -1).expand(groups, -1)
    selected, _compact_lengths = core.compact(
        buffers.representative_q[:groups],
        kmin,
        kmax,
        grouped_table,
        buffers.group_end[:groups],
        workspace,
    )
    prefill_start = plan.prefill_start
    prefix_end = prefill_start + plan.prefix_tokens
    prefix_result = flash_attn_varlen_func(
        q=query[prefill_start:prefix_end],
        k=key_cache,
        v=value_cache,
        block_table=selected[:groups],
        cu_seqlens_q=buffers.cu_q[: groups + 1],
        cu_seqlens_k=buffers.compact_cu_k[: groups + 1],
        max_seqlen_q=plan.max_group_q,
        max_seqlen_k=_BUDGET_TOKENS,
        softmax_scale=softmax_scale,
        causal=True,
    )
    prefix_output = (
        prefix_result[0] if isinstance(prefix_result, tuple) else prefix_result
    )
    output[prefill_start:prefix_end] = prefix_output

    tail_start = prefix_end
    tail_end = prefill_start + plan.query_len
    tail_result = flash_attn_varlen_func(
        q=query[tail_start:tail_end],
        k=key_cache,
        v=value_cache,
        block_table=block_table_row.view(1, -1),
        cu_seqlens_q=buffers.tail_cu_q,
        cu_seqlens_k=buffers.tail_cu_k,
        max_seqlen_q=_DENSE_TAIL,
        max_seqlen_k=plan.seq_len,
        softmax_scale=softmax_scale,
        causal=True,
    )
    tail_output = tail_result[0] if isinstance(tail_result, tuple) else tail_result
    output[tail_start:tail_end] = tail_output
    if _RUNTIME.enabled and not _RUNTIME.logged:
        _RUNTIME.logged = True
        logger.info(
            "Native sparse prefill route: history=%d qlen=%d groups=%d max_group_q=%d budget=%d",
            plan.history,
            plan.query_len,
            groups,
            plan.max_group_q,
            _BUDGET_TOKENS,
        )


def disable(reason: str = "disabled") -> None:
    global _RUNTIME
    _RUNTIME = _Runtime(reason=reason)


def _startup_check(core, device: torch.device, workspace) -> None:
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils.fa_utils import (
        flash_attn_varlen_func,
    )
    from vllm_fl.ops.native_sparse import (
        _assert_close,
        _cpu_attention,
        _gather_physical_cpu,
    )

    history, query_len = 8193, 129
    seq_len = history + query_len
    pages = math.ceil(seq_len / _PAGE)
    blocks = pages + 1
    if workspace.max_pages < pages or workspace.max_seqs < 2:
        raise RuntimeError(
            "selector workspace is too small for the sparse prefill startup fixture"
        )
    generator = torch.Generator(device=device).manual_seed(20261115)
    raw = torch.randn(
        (2, blocks, _PHYSICAL_PAGE, _KV_HEADS, _HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    kv = core.padded_view(raw, blocks)
    key_cache, value_cache = kv.unbind(0)
    kmin, kmax = core.metadata_views(kv)
    kmin.fill_(123)
    kmax.fill_(-123)
    physical = torch.arange(1, blocks, dtype=torch.int32, device=device)
    physical = torch.roll(physical, shifts=7)
    table = torch.zeros((1, workspace.max_pages), dtype=torch.int32, device=device)
    table[0, :pages] = physical[:pages]

    buffers = PlanBuffers(
        device, max_groups=workspace.max_seqs, max_pages=workspace.max_pages
    )
    qstarts = torch.tensor([0, query_len], dtype=torch.int32)
    seqlens = torch.tensor([seq_len], dtype=torch.int32)
    plan = buffers.prepare(
        req_count=1,
        num_prefills=1,
        num_decodes=0,
        num_decode_tokens=0,
        query_start_loc_cpu=qstarts,
        seq_lens_cpu=seqlens,
        causal=True,
        use_cascade=False,
        dcp_world_size=1,
        dense_limit=workspace.dense_limit,
        block_table_width=workspace.max_pages,
    )
    if plan is None:
        raise RuntimeError("startup fixture did not produce a sparse prefill plan")
    query = torch.randn(
        (query_len, _Q_HEADS, _HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    output = torch.empty_like(query)
    with torch.cuda.device(device):
        run(
            plan,
            query,
            key_cache,
            value_cache,
            table[0],
            kmin,
            kmax,
            workspace,
            output,
            1 / math.sqrt(_HEAD_DIM),
        )
        torch.cuda.synchronize(device)
    if not torch.isfinite(output).all():
        raise RuntimeError("native sparse prefill startup output is non-finite")

    q_cpu = query.float().cpu()
    table_cpu = table.cpu()[0]
    output_cpu = output.float().cpu()
    group_ranges, prefix_tokens = global_query_groups(history, query_len)
    selected, compact_lengths = core.compact(
        buffers.representative_q[: len(group_ranges)],
        kmin,
        kmax,
        table.expand(len(group_ranges), -1),
        buffers.group_end[: len(group_ranges)],
        workspace,
    )
    selected_cpu = selected[: len(group_ranges), :_BUDGET_PAGES].cpu()
    compact_lengths_cpu = compact_lengths[: len(group_ranges)].cpu()
    for group_index, (start, end) in enumerate(group_ranges):
        group_end = history + end
        logical_pages = math.ceil(group_end / _PAGE)
        page_table = table_cpu[:logical_pages].to(torch.int64)
        selected_len = int(compact_lengths_cpu[group_index])
        selected_count = math.ceil(selected_len / _PAGE)
        physical = selected_cpu[group_index, :selected_count]
        live_ids = set(page_table.tolist())
        if selected_count != _BUDGET_PAGES or any(
            physical_id <= 0 or physical_id not in live_ids
            for physical_id in physical.tolist()
        ):
            raise RuntimeError(
                "native sparse prefill startup selector returned invalid physical pages"
            )
        key, value = _gather_physical_cpu(kv, physical, selected_len)
        group_len = end - start
        for local_index in sorted({0, group_len - 1}):
            qindex = plan.prefill_start + start + local_index
            allowed_keys = selected_len - group_len + local_index + 1
            expected = _cpu_attention(
                q_cpu[qindex : qindex + 1], key, value, allowed_keys
            )[0]
            _assert_close(
                "native sparse prefill selected-key CPU FP32 attention",
                output_cpu[qindex],
                expected,
            )

    # Compare the dense tail with the same original-table varlen call.
    dense_tail_result = flash_attn_varlen_func(
        q=query[prefix_tokens:],
        k=key_cache,
        v=value_cache,
        block_table=table,
        cu_seqlens_q=torch.tensor([0, _DENSE_TAIL], dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
        max_seqlen_q=_DENSE_TAIL,
        max_seqlen_k=seq_len,
        softmax_scale=1 / math.sqrt(_HEAD_DIM),
        causal=True,
    )
    dense_tail = (
        dense_tail_result[0]
        if isinstance(dense_tail_result, tuple)
        else dense_tail_result
    )
    torch.testing.assert_close(output[prefix_tokens:], dense_tail, rtol=0, atol=0)


def _benchmark_prefill_graphs(core, device: torch.device, workspace) -> bool:
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils.fa_utils import (
        flash_attn_varlen_func,
    )
    from vllm_fl.ops.native_sparse import graph_pair_medians

    history, query_len = 8192, 2048
    seq_len = history + query_len
    pages = seq_len // _PAGE
    blocks = pages + 1
    if workspace.max_pages < pages or workspace.max_seqs < 1:
        raise RuntimeError(
            "selector workspace is too small for the sparse prefill Graph gate"
        )
    qstarts = torch.tensor([0, query_len], dtype=torch.int32)
    seqlens = torch.tensor([seq_len], dtype=torch.int32)
    buffers = PlanBuffers(
        device, max_groups=workspace.max_seqs, max_pages=workspace.max_pages
    )
    plan = buffers.prepare(
        req_count=1,
        num_prefills=1,
        num_decodes=0,
        num_decode_tokens=0,
        query_start_loc_cpu=qstarts,
        seq_lens_cpu=seqlens,
        causal=True,
        use_cascade=False,
        dcp_world_size=1,
        dense_limit=workspace.dense_limit,
        block_table_width=workspace.max_pages,
    )
    if plan is None:
        raise RuntimeError(
            "startup fixture did not produce a sparse prefill Graph plan"
        )

    dense_cu_q = torch.tensor([0, query_len], dtype=torch.int32, device=device)
    dense_cu_k = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    pools = []
    dense_graph = sparse_graph = None
    call = capture = graphs = None
    pool = storage = kv = key_cache = value_cache = physical = table = None
    kmin = kmax = generator = None
    try:
        for index in range(3):
            generator = torch.Generator(device=device).manual_seed(20261120 + index)
            storage = torch.randn(
                (2, blocks, _PHYSICAL_PAGE, _KV_HEADS, _HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            kv = core.padded_view(storage, blocks)
            key_cache, value_cache = kv.unbind(0)
            physical = torch.roll(
                torch.arange(1, blocks, dtype=torch.int32, device=device), index * 37
            )
            table = torch.zeros(
                (1, workspace.max_pages), dtype=torch.int32, device=device
            )
            table[0, :pages] = physical
            kmin, kmax = core.metadata_views(kv)
            pools.append(
                {
                    "storage": storage,
                    "key": key_cache,
                    "value": value_cache,
                    "kmin": kmin,
                    "kmax": kmax,
                    "table": table,
                    "query": torch.randn(
                        (query_len, _Q_HEADS, _HEAD_DIM),
                        dtype=torch.bfloat16,
                        device=device,
                        generator=generator,
                    ),
                    "output": torch.empty(
                        (query_len, _Q_HEADS, _HEAD_DIM),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                }
            )
        kv_bytes = pools[0]["storage"].numel() * pools[0]["storage"].element_size()
        if kv_bytes <= (8 << 20):
            raise RuntimeError("each rotating sparse prefill KV pool must exceed 8 MiB")

        def call(pool, sparse):
            if sparse:
                run(
                    plan,
                    pool["query"],
                    pool["key"],
                    pool["value"],
                    pool["table"][0],
                    pool["kmin"],
                    pool["kmax"],
                    workspace,
                    pool["output"],
                    1 / math.sqrt(_HEAD_DIM),
                )
            else:
                result = flash_attn_varlen_func(
                    q=pool["query"],
                    k=pool["key"],
                    v=pool["value"],
                    block_table=pool["table"],
                    cu_seqlens_q=dense_cu_q,
                    cu_seqlens_k=dense_cu_k,
                    max_seqlen_q=query_len,
                    max_seqlen_k=seq_len,
                    softmax_scale=1 / math.sqrt(_HEAD_DIM),
                    causal=True,
                )
                output = result[0] if isinstance(result, tuple) else result
                pool["output"].copy_(output)

        def capture(sparse):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for pool in pools * 4:
                    call(pool, sparse)
            return graph

        with torch.cuda.device(device):
            for sparse in (False, True):
                for pool in pools:
                    call(pool, sparse)
            torch.cuda.synchronize(device)
            dense_graph = capture(False)
            sparse_graph = capture(True)
            torch.cuda.synchronize(device)
            graphs = (dense_graph,), (sparse_graph,)
            graph_pair_medians(device, *graphs)  # Warmup.
            dense_ms, sparse_ms = graph_pair_medians(device, *graphs)

        calls = len(pools) * 4
        logger.info(
            "Native sparse prefill full-chain Graph ABBA: history=%d qlen=%d "
            "dense_median_ms_per_call=%.4f sparse_median_ms_per_call=%.4f",
            history,
            query_len,
            dense_ms / calls,
            sparse_ms / calls,
        )
        return sparse_ms < dense_ms
    finally:
        call = capture = graphs = plan = dense_graph = sparse_graph = pools = None
        buffers = qstarts = seqlens = dense_cu_q = dense_cu_k = None
        pool = storage = kv = key_cache = value_cache = physical = table = None
        kmin = kmax = generator = None
        gc.collect()
        torch.cuda.empty_cache()
