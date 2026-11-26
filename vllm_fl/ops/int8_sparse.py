"""Gated combined sparse decode/prefill path for the INT8 KV cache."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from vllm.logger import init_logger

from vllm_fl import device_profiles, envs

logger = init_logger("vllm.vllm_fl.int8_sparse")

_SWITCH = "VLLM_FL_KV_INT8_SPARSE"
_PREFILL_SWITCH = "VLLM_FL_KV_INT8_SPARSE_PREFILL"
_DENSE_T = "VLLM_FL_KV_INT8_SPARSE_DENSE_T"
_PAGE = 16
_BUDGET_TOKENS = 2048
_DEFAULT_DENSE_T = 8192
_PREFILL_HISTORY = 8192
_PREFILL_Q_MIN = 1024
_PREFILL_Q_MAX = 2048
_DENSE_TAIL = 64
_MAX_TOKENS = 8192 * _PAGE


@dataclass
class State:
    enabled: bool = False
    prefill_enabled: bool = False
    reason: str = "not installed"
    threshold: int = _DEFAULT_DENSE_T
    dense_limit: int = max(_DEFAULT_DENSE_T, _BUDGET_TOKENS)
    workspace: object | None = None
    planner: object | None = None
    prefill_buffers: object | None = None
    prefix_workspace: object | None = None
    tail_workspace: object | None = None
    decode_config: object | None = None
    refresher: object | None = None
    logged_decode_dense: bool = False
    logged_decode_sparse: bool = False
    logged_prefill: bool = False
    num_layers: int = 0
    max_pages: int = 0
    max_seqs: int = 0


_STATE = State()


def state() -> State:
    return _STATE


def is_enabled() -> bool:
    return _STATE.enabled and _STATE.workspace is not None


def prefill_enabled() -> bool:
    return is_enabled() and _STATE.prefill_enabled


def effective_dense_limit() -> int:
    return int(_STATE.dense_limit)


def log_runtime_route_once(
    sparse: bool, max_seq_len: int, dense_limit: int, runtime_mode
) -> None:
    if not is_enabled():
        return
    if sparse:
        if _STATE.logged_decode_sparse:
            return
        _STATE.logged_decode_sparse = True
        route = "sparse"
    else:
        if _STATE.logged_decode_dense:
            return
        _STATE.logged_decode_dense = True
        route = "dense"
    logger.info(
        "Sparse INT8 decode runtime route: path=%s max_seq_len=%d dense_limit=%d mode=%s",
        route,
        max_seq_len,
        dense_limit,
        runtime_mode,
    )


def workspace_nbytes() -> int:
    if not is_enabled():
        return 0
    total = int(_STATE.workspace.nbytes)
    if _STATE.planner is not None:
        total += (
            _STATE.planner.jobs_gpu.numel() * _STATE.planner.jobs_gpu.element_size()
        )
    if _STATE.prefill_buffers is not None:
        total += sum(
            x.numel() * x.element_size()
            for x in (
                _STATE.prefill_buffers.device_buffer,
                _STATE.prefill_buffers.representative_q,
            )
        )
    for workspace in (_STATE.prefix_workspace, _STATE.tail_workspace):
        if workspace is not None:
            total += sum(
                x.numel() * x.element_size() for x in workspace.buffers.values()
            )
    # MultiLayerMetadataRefresher binds this small pointer table after KV allocation.
    total += _STATE.num_layers * 4 * 8
    return total


def source_hash() -> str:
    """Key compilation on the candidate route, metadata ABI, and cache layout."""
    from vllm_fl.attention import (
        gqa_hybrid,
        int8_sparse as core,
        kv_int8,
        native_sparse_decode,
        native_sparse_prefill,
    )
    from vllm_fl.dispatch.backends.flaggems.impl import gqa_hybrid_attention

    paths = (
        Path(__file__),
        Path(core.__file__),
        Path(native_sparse_decode.__file__),
        Path(native_sparse_prefill.__file__),
        Path(kv_int8.__file__),
        Path(gqa_hybrid.__file__),
        Path(__file__).with_name("native_sparse.py"),
        Path(gqa_hybrid_attention.__file__),
        Path(__file__).parents[1] / "device_profiles.py",
        Path(__file__).parents[1] / "worker/model_runner.py",
        Path(__file__).parents[1] / "optimizations/runtime.py",
        Path(__file__).parents[1] / "attention/int8_decode_config.py",
        Path(__file__).parents[1] / "attention/decode_config.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.resolve().read_bytes())
    digest.update(
        json.dumps(
            {
                "threshold": os.environ.get(_DENSE_T, str(_DEFAULT_DENSE_T)),
                "switch": os.environ.get(_SWITCH, "auto"),
                "prefill_switch": os.environ.get(_PREFILL_SWITCH, "auto"),
                "budget": _BUDGET_TOKENS,
            },
            sort_keys=True,
        ).encode()
    )
    return digest.hexdigest()


def _threshold() -> int:
    value = int(os.environ.get(_DENSE_T, str(_DEFAULT_DENSE_T)))
    if value < 0 or value > _MAX_TOKENS:
        raise ValueError(f"{_DENSE_T} must be in [0,{_MAX_TOKENS}]")
    return value


def _validate(
    vllm_config, device: torch.device, profile: dict
) -> tuple[int, int, int, int]:
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    cache = vllm_config.cache_config
    scheduler = vllm_config.scheduler_config
    compilation = vllm_config.compilation_config
    from vllm_fl.attention import kv_int8

    if device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("sparse INT8 KV requires CUDA")
    if profile.get("int8_sparse") is not True or not kv_int8.kv_int8_enabled():
        raise ValueError("sparse INT8 KV requires the validated hybrid INT8 profile")
    if model.dtype != torch.bfloat16 or cache.cache_dtype != "auto":
        raise ValueError(
            "sparse INT8 KV requires BF16 model weights and automatic KV dtype"
        )
    if (cache.block_size or _PAGE) != _PAGE:
        raise ValueError(f"sparse INT8 KV requires block_size={_PAGE}")
    if (
        model.get_num_attention_heads(parallel),
        model.get_num_kv_heads(parallel),
        model.get_head_size(),
    ) != (16, 2, 128):
        raise ValueError("sparse INT8 KV requires local attention geometry (16,2,128)")
    if (
        parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.use_ubatching
    ):
        raise ValueError("sparse INT8 KV requires TP/PP/DCP=1 and no microbatching")
    if (
        vllm_config.speculative_config is not None
        or vllm_config.kv_transfer_config is not None
    ):
        raise ValueError(
            "sparse INT8 KV does not support speculative decoding or KV transfer"
        )
    if (
        cache.kv_offloading_size is not None
        or getattr(model, "is_encoder_decoder", False)
        or getattr(model, "is_hybrid", False)
    ):
        raise ValueError("sparse INT8 KV requires decoder-only, resident KV cache")
    if model.max_model_len < 16512 or model.max_model_len > _MAX_TOKENS:
        raise ValueError(
            "sparse INT8 KV startup gates require max_model_len in [16512,131072]"
        )
    max_pages = math.ceil(model.max_model_len / _PAGE)
    max_seqs = max(
        int(scheduler.max_num_seqs),
        int(getattr(compilation, "max_cudagraph_capture_size", 0) or 0),
    )
    if max_seqs < 64:
        raise ValueError(
            "sparse INT8 KV startup gates require at least 64 sequence slots"
        )
    max_layers = int(model.get_num_layers(parallel))
    if max_layers <= 0:
        raise ValueError("could not determine local attention layer count")
    threshold = _threshold()
    return threshold, max_pages, max_seqs, max_layers


def _raw_cache(device, blocks: int, generator=None):
    from vllm_fl.attention import kv_int8

    stride = kv_int8.SPARSE_PAGE_BYTES
    raw = torch.randint(
        -127,
        128,
        (blocks * stride,),
        dtype=torch.int8,
        device=device,
        generator=generator,
    )
    cache = torch.as_strided(
        raw,
        (blocks, 2, _PAGE, 2, 128),
        (stride, kv_int8.DATA_BYTES // 2, 256, 128, 1),
    )
    scales = kv_int8.scale_view(cache)
    scales.fill_(0.035)
    return cache


def _decode_fixture(device, batch: int, length: int, max_pages: int, *, seed: int):
    from vllm_fl.attention import int8_sparse as core

    pages = math.ceil(length / _PAGE)
    blocks = batch * pages + 1
    generator = torch.Generator(device=device).manual_seed(seed)
    cache = _raw_cache(device, blocks, generator)
    table_cpu = torch.zeros((batch, max_pages), dtype=torch.int32)
    for row in range(batch):
        ids = torch.arange(row * pages + 1, (row + 1) * pages + 1, dtype=torch.int32)
        table_cpu[row, :pages] = ids
    table = table_cpu.to(device)
    for row in range(batch):
        core.refresh_full_pages(cache, table[row], length // _PAGE)
    q = torch.randn(
        (batch, 16, 128), dtype=torch.bfloat16, device=device, generator=generator
    )
    seq = torch.full((batch,), length, dtype=torch.int32, device=device)
    cu = torch.arange(batch + 1, dtype=torch.int32, device=device)
    out = torch.empty_like(q)
    return cache, table, table_cpu, seq, cu, q, out


def _dequant_selected(
    cache, scale, physical: list[int], length: int, *, round_bf16=False
):
    ids = torch.as_tensor(physical, dtype=torch.int64, device=cache.device)
    raw = cache.detach().index_select(0, ids).cpu()
    scales = scale.detach().index_select(0, ids).cpu()
    keys, values = [], []
    left = length
    for i, block in enumerate(physical):
        take = min(_PAGE, left)
        k = raw[i, 0, :take].float()
        v = raw[i, 1, :take].float()
        ks = scales[i, 0].permute(1, 0)[:take].float()[..., None]
        vs = scales[i, 1].permute(1, 0)[:take].float()[..., None]
        key, value = k * ks, v * vs
        if round_bf16:
            key, value = (
                key.to(torch.bfloat16).float(),
                value.to(torch.bfloat16).float(),
            )
        keys.append(key)
        values.append(value)
        left -= take
        if left <= 0:
            break
    return torch.cat(keys, dim=0)[:length], torch.cat(values, dim=0)[:length]


def _check_decode_selection(
    cache, table, table_cpu, seq, query, output, workspace, config
):
    from vllm_fl.attention import int8_sparse as core, kv_int8, native_sparse_decode
    from vllm_fl.ops import native_sparse

    kmin, kmax = core.metadata_views(cache)
    selected, lengths = native_sparse_decode.compact(
        query[:1], kmin, kmax, table[:1], seq[:1], workspace
    )
    seq_len = int(seq[0].item())
    live = math.ceil(seq_len / _PAGE)
    ids = table_cpu[0, :live].to(device=query.device, dtype=torch.int64)
    expected_scores = native_sparse._cpu_page_scores(
        query[:1], kmin.index_select(0, ids), kmax.index_select(0, ids)
    )[0]
    expected_pages = native_sparse._cpu_expected_pages(
        expected_scores, live, seq_len, workspace.dense_limit, _BUDGET_TOKENS
    )
    length = int(lengths[0].item())
    count = math.ceil(length / _PAGE)
    expected_physical = table_cpu[0, expected_pages[:count]]
    if not torch.equal(selected[0, :count].cpu(), expected_physical):
        raise RuntimeError(
            "INT8 sparse selector differs from the independent CPU FP32 page ranking"
        )
    kv_int8.kv_int8_decode(
        query[:1],
        cache,
        kv_int8.half_base(cache),
        output[:1],
        torch.tensor([0, 1], dtype=torch.int32, device=query.device),
        lengths[:1],
        selected[:1],
        128**-0.5,
        kv_int8.DecodeWorkspace(query.device),
        config=config,
    )
    key, value = _dequant_selected(
        cache, kv_int8.scale_view(cache), expected_physical.tolist(), length
    )
    reference = native_sparse._cpu_attention(query[:1], key, value, length)[0]
    native_sparse._assert_close(
        "INT8 sparse selected-key CPU FP32 attention", output[:1], reference[None]
    )


def _capture_graph(device, fn):
    fn()
    fn()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize(device)
    return graph


def _decode_graph_gate(device, state: State) -> list[dict]:
    from vllm_fl.attention import int8_sparse as core, kv_int8, native_sparse_decode
    from vllm_fl.ops.native_sparse import graph_pair_medians

    results = []
    actual_dense_limit = int(state.dense_limit)
    performance_dense_limit = max(_DEFAULT_DENSE_T, actual_dense_limit)
    for batch, length in ((64, 4097), (32, 16512)):
        actual_sparse_route = length > actual_dense_limit
        contexts = [
            _decode_fixture(device, batch, length, state.max_pages, seed=20261115 + i)
            for i in range(4)
        ]
        config = state.decode_config
        refreshers = []
        jobs = []
        for cache, table, *_ in contexts:
            refreshers.append(core.MultiLayerMetadataRefresher([cache]))
            jobs.append(table[:, length // _PAGE - 1].contiguous())
        select_workspace = native_sparse_decode.Workspace(
            device,
            max_seqs=state.max_seqs,
            max_pages=state.max_pages,
            dense_threshold=state.threshold,
            budget_tokens=_BUDGET_TOKENS,
            pipeline=None,
        )
        _check_decode_selection(
            contexts[0][0],
            contexts[0][1],
            contexts[0][2],
            contexts[0][3],
            contexts[0][5],
            contexts[0][6],
            select_workspace,
            config,
        )
        kv_workspace = kv_int8.DecodeWorkspace(device)
        graphs = [[], []]
        for context_index, (cache, table, _, seq, cu, q, out) in enumerate(contexts):
            scales = kv_int8.half_base(cache)

            def dense_call(
                cache=cache,
                table=table,
                seq=seq,
                cu=cu,
                q=q,
                out=out,
                scales=scales,
                kv_workspace=kv_workspace,
                config=config,
            ):
                kv_int8.kv_int8_decode(
                    q,
                    cache,
                    scales,
                    out,
                    cu,
                    seq,
                    table,
                    128**-0.5,
                    kv_workspace,
                    config=config,
                )

            def sparse_call(
                cache=cache,
                table=table,
                seq=seq,
                cu=cu,
                q=q,
                out=out,
                scales=scales,
                refresher=refreshers[context_index],
                refresh_jobs=jobs[context_index],
                dense_call=dense_call,
                use_sparse_route=actual_sparse_route,
                batch=batch,
                select_workspace=select_workspace,
                kv_workspace=kv_workspace,
                config=config,
            ):
                if not use_sparse_route:
                    dense_call()
                    return
                refresher.refresh(refresh_jobs, batch)
                kmin, kmax = core.metadata_views(cache)
                compact_table, compact_lengths = native_sparse_decode.compact(
                    q,
                    kmin,
                    kmax,
                    table,
                    seq,
                    select_workspace,
                )
                kv_int8.kv_int8_decode(
                    q,
                    cache,
                    scales,
                    out,
                    cu,
                    compact_lengths,
                    compact_table,
                    128**-0.5,
                    kv_workspace,
                    config=config,
                )

            graphs[0].append(_capture_graph(device, dense_call))
            graphs[1].append(_capture_graph(device, sparse_call))
        dense_ms, sparse_ms = graph_pair_medians(device, *graphs, rounds=2)
        results.append(
            {
                "batch": batch,
                "length": length,
                "eligible": length > performance_dense_limit,
                "dense_ms": dense_ms,
                "sparse_full_chain_ms": sparse_ms,
                "positive": sparse_ms < dense_ms,
            }
        )
        dense_call = sparse_call = None
        contexts = graphs = refreshers = jobs = select_workspace = kv_workspace = None
        gc.collect()
        torch.cuda.empty_cache()
    return results


@dataclass
class PrefillBundle:
    plan: object
    prefix_plan: object
    tail_plan: object
    prefill_row: int
    decode_external: bool
    num_decodes: int


def _plan_bundle(
    common,
    *,
    query_lengths,
    choice,
    prefill_wide,
    fixed_anchor,
    lpt,
    split_prefix_mask,
    decode_external,
):
    from vllm_fl.attention import mixed_select, native_sparse_prefill
    from vllm_fl.attention.gqa_hybrid import HybridPlan

    buffers = _STATE.prefill_buffers
    if buffers is None or not _STATE.prefill_enabled:
        return None
    qstarts = common.query_start_loc_cpu
    seq_cpu = getattr(common, "_seq_lens_cpu", None)
    if seq_cpu is None:
        seq_cpu = common.seq_lens_cpu_upper_bound
    if seq_cpu is None or seq_cpu.device.type != "cpu":
        return None
    num_decodes = query_lengths.count(1)
    num_prefills = sum(q > 1 for q in query_lengths)
    decode_tokens = num_decodes
    plan = buffers.prepare(
        req_count=common.num_reqs,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_decode_tokens=decode_tokens,
        query_start_loc_cpu=qstarts,
        seq_lens_cpu=seq_cpu,
        causal=common.causal is True,
        use_cascade=False,
        dcp_world_size=1,
        dense_limit=max(_PREFILL_HISTORY, _STATE.dense_limit),
        block_table_width=common.block_table_tensor.shape[1],
    )
    if plan is None:
        return None
    groups, _ = native_sparse_prefill.global_query_groups(plan.history, plan.query_len)
    qlens = tuple(end - start for start, end in groups)
    group_ends = buffers.group_end_cpu[: plan.num_groups].tolist()
    kv_lens = tuple(
        (_BUDGET_TOKENS - _PAGE) + ((end % _PAGE) or _PAGE) for end in group_ends
    )
    pm, pn, reblock = mixed_select.TILES[choice.tile]
    common_options = dict(
        device=_STATE.workspace.device,
        layers=1,
        split_prefix_mask=split_prefix_mask,
        prefill_order="lpt" if lpt else "natural",
        prefill_reblock=reblock,
        prefill_wide=prefill_wide,
        kv_int8_scratch_tokens=kv_int8_scratch_cap(),
        prefill_fixed_anchor=fixed_anchor,
    )
    prefix = HybridPlan.build(
        qlens,
        kv_lens,
        pm=pm,
        pn=pn,
        workspace=_STATE.prefix_workspace,
        **common_options,
    )
    tail = HybridPlan.build(
        (_DENSE_TAIL,),
        (plan.seq_len,),
        pm=pm,
        pn=pn,
        workspace=_STATE.tail_workspace,
        **common_options,
    )
    prefix.reset()
    tail.reset()
    return PrefillBundle(
        plan=plan,
        prefix_plan=prefix,
        tail_plan=tail,
        prefill_row=num_decodes,
        decode_external=bool(decode_external),
        num_decodes=num_decodes,
    )


def kv_int8_scratch_cap() -> int:
    from vllm_fl.attention import kv_int8

    return kv_int8.prefill_scratch_cap_tokens()


def make_prefill_bundle(
    common,
    query_lengths,
    lengths,
    *,
    choice,
    prefill_wide,
    fixed_anchor,
    lpt,
    split_prefix_mask,
    decode_external,
):
    if not prefill_enabled():
        return None
    prefill_count = sum(q > 1 for q in query_lengths)
    if prefill_count != 1:
        return None
    row = query_lengths.count(1)
    if row >= len(query_lengths):
        return None
    qlen = query_lengths[row]
    history = int(lengths[row]) - qlen
    if not (
        _PREFILL_Q_MIN <= qlen <= _PREFILL_Q_MAX
        and history >= max(_PREFILL_HISTORY, _STATE.dense_limit)
    ):
        return None
    if query_lengths.count(1) and not decode_external:
        return None
    bundle = _plan_bundle(
        common,
        query_lengths=query_lengths,
        choice=choice,
        prefill_wide=prefill_wide,
        fixed_anchor=fixed_anchor,
        lpt=lpt,
        split_prefix_mask=split_prefix_mask,
        decode_external=decode_external,
    )
    if bundle is not None:
        bundle.prefill_row = row
        if not _STATE.logged_prefill:
            _STATE.logged_prefill = True
            logger.info(
                "Sparse INT8 prefill plan: history=%d qlen=%d groups=%d dense_tail=%d",
                history,
                qlen,
                bundle.plan.num_groups,
                _DENSE_TAIL,
            )
    return bundle


def run_prefill(
    bundle: PrefillBundle,
    query,
    kv_cache,
    block_table,
    seq_lens,
    output,
    softmax_scale,
    *,
    layer_id: int,
):
    from vllm_fl.attention import int8_sparse as core, kv_int8, native_sparse_decode

    plan = bundle.plan
    buffers = plan.buffers
    row = bundle.prefill_row
    table_row = block_table[row]
    seq_len = plan.seq_len
    # The CPU progress planner only refreshes pages on decode updates. A dense
    # prefill can have populated every earlier page without metadata, so refresh
    # every complete page before selecting this prefix.
    core.refresh_full_pages(kv_cache, table_row, seq_len // _PAGE)
    groups = plan.num_groups
    torch.index_select(
        query, 0, buffers.representative[:groups], out=buffers.representative_q[:groups]
    )
    group_table = table_row.view(1, -1).expand(groups, -1)
    kmin, kmax = core.metadata_views(kv_cache)
    selected, compact_lengths = native_sparse_decode.compact(
        buffers.representative_q[:groups],
        kmin,
        kmax,
        group_table,
        buffers.group_end[:groups],
        _STATE.workspace,
    )
    k, v = kv_cache.unbind(1)
    scales = kv_int8.half_base(kv_cache)
    prefix_start = plan.prefill_start
    prefix_end = prefix_start + plan.prefix_tokens
    bundle.prefix_plan.run(
        dict(
            q=query[prefix_start:prefix_end],
            k=k,
            v=v,
            out=output[prefix_start:prefix_end],
            cu_seqlens_q=buffers.cu_q[: groups + 1],
            seqused_k=compact_lengths[:groups],
            block_table=selected[:groups],
            softmax_scale=softmax_scale,
            kv_scales=scales,
        ),
        mode="fused",
        layer=layer_id,
    )
    tail_start = prefix_end
    tail_end = prefix_start + plan.query_len
    bundle.tail_plan.run(
        dict(
            q=query[tail_start:tail_end],
            k=k,
            v=v,
            out=output[tail_start:tail_end],
            cu_seqlens_q=buffers.tail_cu_q,
            seqused_k=seq_lens[row : row + 1],
            block_table=table_row.view(1, -1),
            softmax_scale=softmax_scale,
            kv_scales=scales,
        ),
        mode="fused",
        layer=layer_id,
    )
    return selected, compact_lengths


def bind_cache(kv_cache, impl) -> None:
    if not is_enabled():
        return
    from vllm_fl.attention import int8_sparse as core, kv_int8

    if kv_int8.page_stride_bytes(kv_cache) != kv_int8.SPARSE_PAGE_BYTES:
        raise RuntimeError("sparse INT8 state requires 9344-byte physical pages")
    impl._fl_kv_int8_sparse_metadata = core.metadata_views(kv_cache)
    impl._fl_kv_int8_sparse_workspace = _STATE.workspace


def compact_decode(impl, query, table, seq_lens):
    if not is_enabled() or getattr(impl, "_fl_kv_int8_sparse_metadata", None) is None:
        return None
    from vllm_fl.attention import native_sparse_decode

    kmin, kmax = impl._fl_kv_int8_sparse_metadata
    return native_sparse_decode.compact(
        query,
        kmin,
        kmax,
        table,
        seq_lens,
        _STATE.workspace,
    )


def should_compact_decode(impl, attn_metadata) -> bool:
    if not is_enabled():
        return False
    from vllm.forward_context import get_forward_context

    from vllm_fl.ops import native_sparse

    forward_context = get_forward_context()
    should_compact = native_sparse.should_use_sparse_route(
        forward_context.batch_descriptor,
        forward_context.cudagraph_runtime_mode,
        int(attn_metadata.max_seq_len),
        effective_dense_limit(),
    )
    return (
        should_compact
        and getattr(impl, "_fl_kv_int8_sparse_metadata", None) is not None
    )


def reset_metadata_progress(scheduler_output) -> None:
    if _STATE.planner is not None:
        _STATE.planner.reset_for_scheduler(scheduler_output)


def refresh_metadata(
    req_ids, num_computed_tokens_cpu, num_scheduled_tokens_np, block_table_cpu
) -> None:
    planner, refresher = _STATE.planner, _STATE.refresher
    if planner is not None and refresher is not None:
        planner.refresh(
            refresher,
            req_ids,
            num_computed_tokens_cpu,
            num_scheduled_tokens_np,
            block_table_cpu,
        )


def _cpu_selected_prefill_check(bundle, query, cache, table, output) -> None:
    from vllm_fl.attention import int8_sparse as core, kv_int8, native_sparse_prefill
    from vllm_fl.ops import native_sparse

    groups, _ = native_sparse_prefill.global_query_groups(
        bundle.plan.history, bundle.plan.query_len
    )
    kmin, kmax = core.metadata_views(cache)
    table_cpu = table.detach().cpu()[bundle.prefill_row]
    q_cpu = query.detach().cpu()
    out_cpu = output.detach().float().cpu()
    scale = kv_int8.scale_view(cache)
    selected = _STATE.workspace.selected[: len(groups)].detach().cpu()
    lengths = _STATE.workspace.compact_lengths[: len(groups)].detach().cpu()
    chosen_groups = sorted({0, len(groups) - 1})
    for gi in chosen_groups:
        start, end = groups[gi]
        selected_len = int(lengths[gi])
        count = math.ceil(selected_len / _PAGE)
        physical = selected[gi, :count].tolist()
        logical_pages = math.ceil((bundle.plan.history + end) / _PAGE)
        live_physical = table_cpu[:logical_pages].to(torch.int64)
        live_device = live_physical.to(cache.device)
        meta_min = kmin.index_select(0, live_device).detach().cpu()
        meta_max = kmax.index_select(0, live_device).detach().cpu()
        score = native_sparse._cpu_page_scores(
            q_cpu[
                bundle.plan.prefill_start + start : bundle.plan.prefill_start
                + start
                + 1
            ],
            meta_min,
            meta_max,
        )[0]
        expected = native_sparse._cpu_expected_pages(
            score,
            logical_pages,
            bundle.plan.history + end,
            _STATE.threshold,
            _BUDGET_TOKENS,
        )
        expected_physical = table_cpu[expected[:count]].tolist()
        if physical != expected_physical:
            raise RuntimeError(
                "sparse prefill page selection differs from CPU FP32 ranking"
            )
        key, value = _dequant_selected(
            cache, scale, physical, selected_len, round_bf16=True
        )
        for local in sorted({0, end - start - 1}):
            allowed = selected_len - (end - start) + local + 1
            expected_out = native_sparse._cpu_attention(
                q_cpu[
                    bundle.plan.prefill_start
                    + start
                    + local : bundle.plan.prefill_start + start + local + 1
                ],
                key,
                value,
                allowed,
            )[0]
            native_sparse._assert_close(
                "INT8 sparse prefill selected-key CPU FP32 attention",
                out_cpu[bundle.plan.prefill_start + start + local],
                expected_out,
            )


def _prefill_graph_gate(
    device,
    state: State,
    choice,
    *,
    lpt: bool,
    split_prefix_mask: bool,
    prefill_wide: bool,
    fixed_anchor: bool,
) -> dict:
    from vllm.v1.attention.backend import CommonAttentionMetadata

    from vllm_fl.attention import kv_int8, mixed_select
    from vllm_fl.attention.gqa_hybrid import HybridPlan, HybridWorkspace
    from vllm_fl.ops.native_sparse import graph_pair_medians

    seq_len, qlen = 16384, 2048
    contexts = []
    full_cu = torch.tensor([0, qlen], dtype=torch.int32, device=device)
    for context_index in range(4):
        generator = torch.Generator(device=device).manual_seed(20261201 + context_index)
        cache, table, _, seq, _, _, _ = _decode_fixture(
            device, 1, seq_len, state.max_pages, seed=20261201 + context_index
        )
        q = torch.randn(
            (qlen, 16, 128), dtype=torch.bfloat16, device=device, generator=generator
        )
        out = torch.empty_like(q)
        qstarts_cpu = torch.tensor([0, qlen], dtype=torch.int32)
        seq_cpu = torch.tensor([seq_len], dtype=torch.int32)
        common = CommonAttentionMetadata(
            query_start_loc=full_cu,
            query_start_loc_cpu=qstarts_cpu,
            seq_lens=seq,
            _seq_lens_cpu=seq_cpu,
            seq_lens_cpu_upper_bound=seq_cpu,
            num_reqs=1,
            num_actual_tokens=qlen,
            max_query_len=qlen,
            max_seq_len=seq_len,
            block_table_tensor=table,
            slot_mapping=torch.empty((qlen,), dtype=torch.int64, device=device),
            causal=True,
        )
        bundle = _plan_bundle(
            common,
            query_lengths=[qlen],
            choice=choice,
            prefill_wide=prefill_wide,
            fixed_anchor=fixed_anchor,
            lpt=lpt,
            split_prefix_mask=split_prefix_mask,
            decode_external=False,
        )
        if bundle is None:
            raise RuntimeError(
                "startup fixture did not create the grouped sparse prefill plan"
            )
        run_prefill(bundle, q, cache, table, seq, out, 128**-0.5, layer_id=0)
        if context_index == 0:
            _cpu_selected_prefill_check(bundle, q, cache, table, out)
        dense_workspace = HybridWorkspace(device)
        pm, pn, reblock = mixed_select.TILES[choice.tile]
        dense = HybridPlan.build(
            [qlen],
            [seq_len],
            device=device,
            pm=pm,
            pn=pn,
            layers=1,
            split_prefix_mask=split_prefix_mask,
            prefill_order="lpt" if lpt else "natural",
            prefill_reblock=reblock,
            prefill_wide=prefill_wide,
            kv_int8_scratch_tokens=kv_int8_scratch_cap(),
            prefill_fixed_anchor=fixed_anchor,
            workspace=dense_workspace,
        )
        dense_out = torch.empty_like(q)
        k, v = cache.unbind(1)
        dense.run(
            dict(
                q=q,
                k=k,
                v=v,
                out=dense_out,
                cu_seqlens_q=full_cu,
                seqused_k=seq,
                block_table=table,
                softmax_scale=128**-0.5,
                kv_scales=kv_int8.half_base(cache),
            ),
            mode="fused",
        )
        torch.cuda.synchronize(device)
        prefix_start = bundle.plan.prefill_start + bundle.plan.prefix_tokens
        torch.testing.assert_close(
            out[prefix_start:], dense_out[prefix_start:], rtol=0.03, atol=0.003
        )
        contexts.append(
            (cache, table, seq, q, out, dense_out, bundle, dense, dense_workspace)
        )

    graphs = [[], []]
    for cache, table, seq, q, out, dense_out, bundle, dense, _ in contexts:
        k, v = cache.unbind(1)
        scales = kv_int8.half_base(cache)

        def sparse_call(cache=cache, table=table, seq=seq, q=q, out=out, bundle=bundle):
            run_prefill(bundle, q, cache, table, seq, out, 128**-0.5, layer_id=0)

        def dense_call(
            dense=dense,
            q=q,
            k=k,
            v=v,
            dense_out=dense_out,
            cu=full_cu,
            seq=seq,
            table=table,
            scales=scales,
        ):
            dense.run(
                dict(
                    q=q,
                    k=k,
                    v=v,
                    out=dense_out,
                    cu_seqlens_q=cu,
                    seqused_k=seq,
                    block_table=table,
                    softmax_scale=128**-0.5,
                    kv_scales=scales,
                ),
                mode="fused",
            )

        graphs[0].append(_capture_graph(device, dense_call))
        graphs[1].append(_capture_graph(device, sparse_call))
    dense_ms, sparse_ms = graph_pair_medians(device, *graphs, rounds=2)
    result = {
        "batch": 1,
        "history": seq_len - qlen,
        "query_len": qlen,
        "dense_ms": dense_ms,
        "sparse_full_chain_ms": sparse_ms,
        "positive": sparse_ms < dense_ms,
    }
    del contexts, graphs
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _startup_gate(device, state: State) -> None:
    from vllm_fl.attention import int8_decode_config, kv_int8, mixed_select

    state.decode_config = int8_decode_config.resolve_decode(device)
    state.prefill_enabled = envs.switch(
        _PREFILL_SWITCH,
        auto=True,
        reason="combined INT8 sparse candidate passed startup gates",
    )
    lpt = envs.switch("VLLM_FL_GQA_HYBRID_PREFILL_LPT", auto=True)
    reblock = envs.switch("VLLM_FL_GQA_HYBRID_PREFILL_REBLOCK", auto=True)
    scratch = kv_int8.prefill_scratch_cap_tokens()
    decision = mixed_select.resolve(
        device,
        page=_PAGE,
        lpt=lpt,
        reblock_ok=reblock,
        decode=(kv_int8.DecodeWorkspace(device), state.decode_config),
        scratch_tokens=scratch,
    )
    choice = decision.lookup(0, 2048, 16384)
    prefill_wide = bool(decision.prefill_wide and choice.tile == "reblock")
    split_prefix_mask = os.environ.get(
        "VLLM_FL_GQA_HYBRID_PREFIX_MASK", "0"
    ).strip().lower() in ("1", "true", "yes")
    from vllm_fl.attention.gqa_hybrid import HybridWorkspace

    state.prefix_workspace = HybridWorkspace(device)
    state.tail_workspace = HybridWorkspace(device)
    decode_results = _decode_graph_gate(device, state)
    logger.info(
        "Sparse INT8 decode startup: dense_limit=%d results=%s",
        state.dense_limit,
        decode_results,
    )
    eligible_decode = [item for item in decode_results if item["eligible"]]
    if not eligible_decode or not all(item["positive"] for item in eligible_decode):
        raise RuntimeError(
            "sparse INT8 decode did not pass the full-chain Graph cost gate"
        )
    if state.prefill_enabled:
        prefill_result = _prefill_graph_gate(
            device,
            state,
            choice,
            lpt=lpt,
            split_prefix_mask=split_prefix_mask,
            prefill_wide=prefill_wide,
            fixed_anchor=decision.prefill_fixed_anchor,
        )
        logger.info(
            "Sparse INT8 prefill startup: tile=%s results=%s",
            choice.tile,
            prefill_result,
        )
        if not prefill_result["positive"]:
            raise RuntimeError(
                "sparse INT8 prefill did not pass the full-chain Graph cost gate"
            )


def install(vllm_config, device: torch.device) -> bool:
    """Enable only after selected-key CPU checks and complete-chain Graph gates."""
    global _STATE
    disable("install pending")
    index = device.index if device.index is not None else torch.cuda.current_device()
    profile = device_profiles.lookup(index) or {}
    try:
        requested = envs.switch(_SWITCH, auto=profile.get("int8_sparse") is True)
        if not requested:
            _STATE.reason = "switch disabled"
            return False
        threshold, max_pages, max_seqs, num_layers = _validate(
            vllm_config, device, profile
        )
        from vllm_fl.attention import native_sparse_decode, native_sparse_prefill
        from vllm_fl.ops.native_sparse import MetadataRefreshPlanner

        budget = _BUDGET_TOKENS
        workspace = native_sparse_decode.Workspace(
            device,
            max_seqs=max_seqs,
            max_pages=max_pages,
            dense_threshold=threshold,
            budget_tokens=budget,
            pipeline=None,
        )
        planner_capacity = max_pages * max_seqs
        planner = MetadataRefreshPlanner(
            planner_capacity,
            workspace.dense_limit,
            device,
            pin_memory=True,
        )
        buffers = native_sparse_prefill.PlanBuffers(
            device, max_groups=64, max_pages=max_pages
        )
        _STATE = State(
            enabled=True,
            reason="startup gates pending",
            threshold=threshold,
            dense_limit=workspace.dense_limit,
            workspace=workspace,
            planner=planner,
            prefill_buffers=buffers,
            num_layers=num_layers,
            max_pages=max_pages,
            max_seqs=max_seqs,
        )
        from vllm_fl.attention import kv_int8

        with kv_int8.temporary_prefill_scratch(device):
            _startup_gate(device, _STATE)
        _STATE.enabled = True
        _STATE.reason = "selected-key and full-chain Graph gates passed"
        logger.info(
            "Sparse INT8 KV enabled: threshold=%d dense_limit=%d prefill=%s page_bytes=9344 "
            "workspace_bytes=%d layers=%d",
            _STATE.threshold,
            _STATE.dense_limit,
            _STATE.prefill_enabled,
            workspace_nbytes(),
            _STATE.num_layers,
        )
        return True
    except Exception as exc:
        disable(f"startup gate failed: {type(exc).__name__}: {exc}")
        logger.warning(
            "Sparse INT8 KV disabled; retaining dense 8320-byte INT8 pages: %s",
            exc,
            exc_info=True,
        )
        return False


def spec_is_supported(spec, backend, layout_order, impl) -> bool:
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

    from vllm_fl.dispatch.backends.flaggems.impl.gqa_hybrid_attention import (
        HybridAttentionBackend,
    )

    return (
        type(spec) is FullAttentionSpec
        and isinstance(backend, type)
        and issubclass(backend, HybridAttentionBackend)
        and tuple(layout_order) == (0, 1, 2, 3, 4)
        and spec.dtype == torch.bfloat16
        and spec.kv_quant_mode == KVQuantMode.NONE
        and spec.page_size_padded is None
        and spec.sliding_window is None
        and spec.attention_chunk_size is None
        and not spec.non_causal
        and (spec.block_size, spec.num_kv_heads, spec.head_size, spec.head_size_v)
        == (_PAGE, 2, 128, 128)
        and getattr(impl, "sliding_window", None) in (None, (-1, -1))
        and getattr(impl, "alibi_slopes", None) is None
        and getattr(impl, "sinks", None) is None
        and getattr(impl, "logits_soft_cap", None) in (None, 0.0)
    )


def bind_refresher(kv_caches, static_forward_context) -> int:
    if not is_enabled():
        return 0
    from vllm_fl.attention import int8_sparse as core, kv_int8

    unique = {}
    for name, cache in kv_caches.items():
        if (
            isinstance(cache, torch.Tensor)
            and cache.dtype == torch.int8
            and kv_int8.page_stride_bytes(cache) == kv_int8.SPARSE_PAGE_BYTES
        ):
            unique.setdefault(cache.data_ptr(), cache)
            layer = static_forward_context.get(name)
            impl = getattr(layer, "impl", None)
            if impl is None:
                raise RuntimeError(
                    f"sparse INT8 KV layer {name} has no attention implementation"
                )
            bind_cache(cache, impl)
    if not unique:
        raise RuntimeError("sparse INT8 enabled without 9344-byte KV layers")
    refresher = core.MultiLayerMetadataRefresher(list(unique.values()))
    if refresher.num_blocks > _STATE.planner.num_physical_blocks:
        raise RuntimeError(
            "KV block count exceeds preallocated sparse metadata planner"
        )
    _STATE.refresher = refresher
    return len(unique)


def clear_cache_bindings(static_forward_context) -> None:
    """Drop profiling KV owners before the real cache is allocated."""
    _STATE.refresher = None
    for layer in static_forward_context.values():
        impl = getattr(layer, "impl", None)
        if impl is not None:
            if hasattr(impl, "_fl_kv_int8_sparse_metadata"):
                impl._fl_kv_int8_sparse_metadata = None
            if hasattr(impl, "_fl_kv_int8_sparse_workspace"):
                impl._fl_kv_int8_sparse_workspace = None


def disable(reason: str) -> None:
    global _STATE
    _STATE = State(reason=reason)
