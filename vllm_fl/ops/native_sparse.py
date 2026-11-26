"""Startup correctness gate and process state for native sparse decode."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger

from vllm_fl import device_profiles, envs

logger = init_logger("vllm.vllm_fl.native_sparse")

_SWITCH = "VLLM_FL_NATIVE_SPARSE_DECODE"
_DENSE_THRESHOLD = "VLLM_FL_SPARSE_DECODE_DENSE_T"
_BUDGET_TOKENS = "VLLM_FL_SPARSE_DECODE_BUDGET"
_DEFAULT_THRESHOLD = 8192
_FIXED_BUDGET_TOKENS = 2048
_PAGE = 16
_PHYSICAL_PAGE = 17
_MAX_PAGES = 8192
_MAX_TOKENS = _MAX_PAGES * _PAGE
_SELF_CHECK_LENGTHS = (17, 4097)
_SELF_CHECK_SEED = 20261114
_LONG_PREFILL_CACHE_LEN = 16384
_LONG_PREFILL_QUERY_LEN = 2048
_GRAPH_DRIVER_FLOOR_BYTES = 1 << 20


@dataclass
class _MetadataProgress:
    completed_pages: int
    last_computed_tokens: int


class MetadataRefreshPlanner:
    """Plan completed physical pages that need metadata before decode."""

    def __init__(
        self,
        num_physical_blocks: int,
        dense_limit: int,
        device: torch.device | str,
        *,
        pin_memory: bool = True,
    ):
        if num_physical_blocks <= 0:
            raise ValueError("metadata planner requires at least one physical KV block")
        self.num_physical_blocks = int(num_physical_blocks)
        self.dense_limit = int(dense_limit)
        self.jobs_cpu = torch.empty(
            (self.num_physical_blocks,),
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.jobs_gpu = torch.empty(
            (self.num_physical_blocks,), dtype=torch.int32, device=device
        )
        self._jobs_numpy = self.jobs_cpu.numpy()
        self._progress: dict[str, _MetadataProgress] = {}

    def reset_for_scheduler(self, scheduler_output) -> None:
        req_ids = set(scheduler_output.finished_req_ids)
        req_ids.update(req.req_id for req in scheduler_output.scheduled_new_reqs)
        req_ids.update(scheduler_output.scheduled_cached_reqs.resumed_req_ids)
        for req_id in req_ids:
            self._progress.pop(req_id, None)

    def plan(
        self,
        req_ids: list[str],
        num_computed_tokens,
        num_scheduled_tokens,
        block_table_cpu: torch.Tensor,
    ) -> int:
        """Fill the pinned job array; return the number of unique physical pages."""
        computed = np.asarray(num_computed_tokens)
        scheduled = np.asarray(num_scheduled_tokens)
        if computed.ndim != 1 or scheduled.ndim != 1:
            raise ValueError(
                "computed and scheduled token counts must be one-dimensional"
            )
        if len(req_ids) != computed.size or computed.size != scheduled.size:
            raise ValueError(
                "request IDs and token-count arrays must have matching lengths"
            )
        if block_table_cpu.device.type != "cpu" or block_table_cpu.dtype != torch.int32:
            raise ValueError("metadata planning requires the CPU int32 block table")
        block_table = block_table_cpu.numpy()
        if block_table.ndim != 2 or block_table.shape[0] < len(req_ids):
            raise ValueError("CPU block table must cover all active request rows")

        pending = []
        for row, req_id in enumerate(req_ids):
            current = int(computed[row])
            scheduled_count = int(scheduled[row])
            progress = self._progress.get(req_id)
            if progress is not None:
                if current < progress.last_computed_tokens:
                    # Speculative correction or cache rewind invalidates the old page range.
                    self._progress.pop(req_id, None)
                    progress = None
                elif current + scheduled_count <= self.dense_limit:
                    self._progress.pop(req_id, None)
                    progress = None
                else:
                    # Observe prefill rows too, so a later decode cannot miss a rollback.
                    progress.last_computed_tokens = current

            if scheduled_count != 1 or current + 1 <= self.dense_limit:
                continue

            completed_pages = current // _PAGE
            if progress is not None and completed_pages < progress.completed_pages:
                self._progress.pop(req_id, None)
                progress = None
            first_page = progress.completed_pages if progress is not None else 0
            if completed_pages <= first_page:
                if progress is not None:
                    progress.last_computed_tokens = current
                continue
            if completed_pages > block_table.shape[1]:
                raise RuntimeError(
                    f"request {req_id!r} needs {completed_pages} KV pages but the block table has "
                    f"{block_table.shape[1]} columns"
                )
            physical_ids = block_table[row, first_page:completed_pages]
            if physical_ids.size != completed_pages - first_page:
                raise RuntimeError(
                    "CPU block table slice did not contain every completed page"
                )
            if np.any(physical_ids <= 0) or np.any(
                physical_ids >= self.num_physical_blocks
            ):
                raise RuntimeError(
                    f"request {req_id!r} has an invalid physical KV page ID in its completed context"
                )
            pending.append(physical_ids)
            self._progress[req_id] = _MetadataProgress(completed_pages, current)

        if not pending:
            return 0
        unique_ids = np.unique(np.concatenate(pending))
        if unique_ids.size > self.num_physical_blocks:
            raise RuntimeError("unique metadata jobs exceed physical KV cache capacity")
        self._jobs_numpy[: unique_ids.size] = unique_ids
        return int(unique_ids.size)

    def copy_jobs_to_device(self, num_jobs: int) -> None:
        count = int(num_jobs)
        if count == 0:
            return
        if count < 0 or count > self.num_physical_blocks:
            raise ValueError("num_jobs exceeds the persistent metadata job buffer")
        self.jobs_gpu[:count].copy_(self.jobs_cpu[:count], non_blocking=True)

    def refresh(self, refresher, req_ids, computed, scheduled, block_table) -> None:
        count = self.plan(req_ids, computed, scheduled, block_table)
        if count:
            self.copy_jobs_to_device(count)
            refresher.refresh(self.jobs_gpu, count)


@dataclass(frozen=True)
class SparseDecodeBatchDescriptor(BatchDescriptor):
    """FULL uniform-decode graph key with an explicit sparse-route bit."""

    sparse_decode: bool = False


@lru_cache(maxsize=8192)
def sparse_decode_batch_descriptor(
    base: BatchDescriptor, sparse_decode: bool
) -> SparseDecodeBatchDescriptor:
    """Return a cached route key while retaining every framework key field."""
    inherited = {
        item.name: getattr(base, item.name) for item in fields(BatchDescriptor)
    }
    return SparseDecodeBatchDescriptor(**inherited, sparse_decode=bool(sparse_decode))


def should_use_sparse_route(
    batch_descriptor: BatchDescriptor | None,
    cudagraph_runtime_mode: CUDAGraphMode,
    max_seq_len: int,
    dense_limit: int,
) -> bool:
    """Resolve a captured FULL decode key or the CPU mixed/eager route."""
    if isinstance(batch_descriptor, SparseDecodeBatchDescriptor):
        return batch_descriptor.sparse_decode
    if (
        cudagraph_runtime_mode == CUDAGraphMode.FULL
        and batch_descriptor is not None
        and batch_descriptor.uniform
    ):
        raise RuntimeError(
            "native sparse FULL decode is missing its route-specific graph key"
        )
    return max_seq_len > dense_limit


@dataclass
class State:
    workspace: object | None = None
    enabled: bool = False
    threshold: int = _DEFAULT_THRESHOLD
    budget_tokens: int = _FIXED_BUDGET_TOKENS
    reason: str = "not installed"
    logged_runtime_routes: set[str] = field(default_factory=set)


_STATE = State()


def state() -> State:
    return _STATE


def is_enabled() -> bool:
    return _STATE.enabled and _STATE.workspace is not None


def workspace_nbytes() -> int:
    if not is_enabled():
        return 0
    return int(_STATE.workspace.nbytes)


def effective_dense_limit() -> int:
    workspace = _STATE.workspace
    if workspace is not None:
        return int(workspace.dense_limit)
    return max(_STATE.threshold, _STATE.budget_tokens)


def route_batch_descriptor(
    batch_descriptor: BatchDescriptor, max_seq_len: int, dense_limit: int
) -> BatchDescriptor:
    return sparse_decode_batch_descriptor(batch_descriptor, max_seq_len > dense_limit)


def full_decode_capture_modes(
    enabled: bool, runtime_mode: CUDAGraphMode, batch_descriptor: BatchDescriptor
) -> tuple[bool | None, ...]:
    """Return explicit route overrides for each FULL uniform-decode capture."""
    if enabled and runtime_mode == CUDAGraphMode.FULL and batch_descriptor.uniform:
        return (False, True)
    return (None,)


def sparse_graph_gate_passes(results: list[dict], dense_limit: int) -> bool:
    """Require valid timings for every case and wins for every eligible case."""
    if not results:
        return False
    for result in results:
        baseline = result["baseline_median_ms_per_call"]
        candidate = result["candidate_median_ms_per_call"]
        if (
            not math.isfinite(baseline)
            or not math.isfinite(candidate)
            or baseline <= 0
            or candidate <= 0
        ):
            return False
    eligible = [result for result in results if result["max_seq_len"] > dense_limit]
    return bool(eligible) and all(
        result["candidate_median_ms_per_call"] < result["baseline_median_ms_per_call"]
        for result in eligible
    )


def graph_pair_memory_estimate(
    mem_samples: list[int], pair_count: int
) -> tuple[int, int, int]:
    """Estimate graph-pair memory with a 2 MiB floor for every pair."""
    if not mem_samples or pair_count <= 0:
        raise ValueError(
            "graph-pair memory estimate needs samples and a positive count"
        )
    first_pair = max(int(mem_samples[0]), 2 * _GRAPH_DRIVER_FLOOR_BYTES)
    per_pair = max(
        int(mem_samples[1]) if len(mem_samples) > 1 else 0,
        2 * _GRAPH_DRIVER_FLOOR_BYTES,
    )
    return first_pair, per_pair, first_pair + per_pair * (pair_count - 1)


def disable(reason: str = "disabled") -> None:
    global _STATE
    workspace = _STATE.workspace
    released_bytes = (
        int(getattr(workspace, "nbytes", 0)) if workspace is not None else 0
    )
    _STATE = State(reason=reason)
    prefill_module = sys.modules.get("vllm_fl.attention.native_sparse_prefill")
    if prefill_module is not None:
        prefill_module.disable(reason)
    if released_bytes:
        logger.warning(
            "Native sparse decode disabled after workspace allocation: "
            "reason=%s released_workspace_bytes=%d; model-memory accounting may "
            "remain conservatively high if disable occurs after model load",
            reason,
            released_bytes,
        )
    del workspace


def source_hash() -> str:
    """Hash every source file that changes cache layout or sparse dispatch."""
    from vllm_fl.attention import native_sparse_decode

    paths = (
        Path(native_sparse_decode.__file__),
        Path(__file__),
        Path(__file__).parents[1] / "attention/native_sparse_prefill.py",
        Path(__file__).parents[1] / "device_profiles.py",
        Path(__file__).parents[1]
        / "dispatch/backends/vendor/metax/impl/attention/flash_attn.py",
        Path(__file__).parents[1] / "worker/model_runner.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        resolved = path.resolve()
        digest.update(str(resolved).encode())
        digest.update(resolved.read_bytes())
    config = {
        "threshold": _STATE.threshold,
        "budget_tokens": _STATE.budget_tokens,
    }
    digest.update(json.dumps(config, sort_keys=True).encode())
    return digest.hexdigest()


def _threshold_env() -> int:
    value = int(os.environ.get(_DENSE_THRESHOLD, str(_DEFAULT_THRESHOLD)))
    if value < 0 or value > _MAX_TOKENS:
        raise ValueError(f"{_DENSE_THRESHOLD} must be in [0, {_MAX_TOKENS}]")
    return value


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _validate_config(
    vllm_config, device: torch.device, profile: dict
) -> tuple[int, int, int]:
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    cache = vllm_config.cache_config
    scheduler = vllm_config.scheduler_config
    compilation = vllm_config.compilation_config

    if device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("native sparse decode requires CUDA and is unavailable on HIP")
    if profile.get("native_sparse_decode") is not True:
        raise ValueError("native sparse decode requires the native BF16 sparse profile")
    if getattr(model, "dtype", None) != torch.bfloat16:
        raise ValueError("native sparse decode requires a BF16 model")
    if cache.cache_dtype != "auto":
        raise ValueError("native sparse decode requires an unquantized BF16 KV cache")
    if (cache.block_size or _PAGE) != _PAGE:
        raise ValueError(f"native sparse decode requires block_size={_PAGE}")
    if (
        model.get_num_attention_heads(parallel),
        model.get_num_kv_heads(parallel),
        model.get_head_size(),
    ) != (16, 2, 128):
        raise ValueError(
            "native sparse decode requires local attention geometry (16,2,128)"
        )
    if (
        parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
    ):
        raise ValueError("native sparse decode requires TP/PP/DCP equal to one")
    if parallel.use_ubatching:
        raise ValueError("native sparse decode does not support microbatching")
    if vllm_config.speculative_config is not None:
        raise ValueError("native sparse decode does not support speculative decoding")
    if vllm_config.kv_transfer_config is not None:
        raise ValueError("native sparse decode does not support KV transfer or offload")
    if cache.kv_offloading_size is not None:
        raise ValueError("native sparse decode does not support KV cache offloading")
    if getattr(model, "is_encoder_decoder", False) or getattr(
        model, "is_hybrid", False
    ):
        raise ValueError("native sparse decode supports decoder-only full attention")

    hf_config = getattr(model, "hf_text_config", None) or getattr(
        model, "hf_config", None
    )
    sliding_window = getattr(model, "sliding_window", None)
    if sliding_window is None and hf_config is not None:
        sliding_window = getattr(hf_config, "sliding_window", None)
    if sliding_window is not None:
        raise ValueError(
            "native sparse decode does not support sliding-window attention"
        )

    configured_backend = getattr(vllm_config.attention_config, "backend", None)
    if (
        configured_backend is not None
        and getattr(configured_backend, "name", "") != "FLASH_ATTN"
    ):
        raise ValueError(
            "native sparse decode requires the native FlashAttention backend"
        )
    from vllm.v1.attention.backends.utils import get_kv_cache_layout

    if get_kv_cache_layout() != "NHD":
        raise ValueError("native sparse decode requires NHD KV cache layout")

    threshold = _threshold_env()
    budget = _positive_int_env(_BUDGET_TOKENS, _FIXED_BUDGET_TOKENS)
    if budget != _FIXED_BUDGET_TOKENS:
        raise ValueError(
            f"native sparse decode supports only budget={_FIXED_BUDGET_TOKENS}"
        )
    max_pages = math.ceil(model.max_model_len / _PAGE)
    if max_pages < math.ceil(16512 / _PAGE):
        raise ValueError("max_model_len is too short for the startup Graph cost gate")
    if max_pages > _MAX_PAGES:
        raise ValueError(f"max_model_len exceeds the {_MAX_TOKENS}-token sparse limit")
    max_seqs = max(
        int(scheduler.max_num_seqs),
        int(getattr(compilation, "max_cudagraph_capture_size", 0) or 0),
    )
    if max_seqs <= 0:
        raise ValueError("max_seqs must be positive")
    if max_seqs < 64:
        raise ValueError("startup Graph cost gate requires max_seqs of at least 64")
    return threshold, budget, max_seqs


def _cpu_page_scores(
    query: torch.Tensor, kmin: torch.Tensor, kmax: torch.Tensor
) -> torch.Tensor:
    """Independent FP32 implementation of sparse_decode.py's scoring rule."""
    q = query.detach().float().cpu().view(query.shape[0], 2, 8, 128)
    lo, hi = kmin.detach().float().cpu(), kmax.detach().float().cpu()
    scores = torch.empty((query.shape[0], lo.shape[0]), dtype=torch.float32)
    for row in range(query.shape[0]):
        for page in range(lo.shape[0]):
            per_kv_head = []
            for kv_head in range(2):
                qh = q[row, kv_head]
                page_lo = lo[page, kv_head]
                page_hi = hi[page, kv_head]
                bounded = torch.maximum(qh * page_lo, qh * page_hi).sum(dim=-1)
                per_kv_head.append(float(bounded.max()))
            scores[row, page] = max(per_kv_head)
    return scores


def _cpu_expected_pages(
    scores: torch.Tensor,
    num_pages: int,
    seq_len: int,
    threshold: int,
    budget_tokens: int,
) -> list[int]:
    budget_pages = budget_tokens // _PAGE
    if seq_len <= max(threshold, budget_tokens):
        return list(range(num_pages))
    forced = set(range(min(1, num_pages)))
    forced.update(range(max(0, num_pages - 4), num_pages))
    candidates = [page for page in range(num_pages) if page not in forced]
    candidates.sort(key=lambda page: (-float(scores[page]), page))
    chosen = forced | set(candidates[: max(0, budget_pages - len(forced))])
    return sorted(chosen)


def _cpu_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, cache_len: int
) -> torch.Tensor:
    """CPU FP32 grouped-query attention oracle for decode and suffix causal prefill."""
    q = query.detach().float().cpu()
    k = key.detach().float().cpu()
    v = value.detach().float().cpu()
    qlen = q.shape[0]
    rows = []
    for qi in range(qlen):
        end = cache_len - qlen + qi + 1
        heads = []
        for head in range(16):
            kv_head = head // 8
            logits = torch.mv(k[:end, kv_head], q[qi, head]) * (128**-0.5)
            probs = torch.softmax(logits, dim=0)
            heads.append(torch.mv(v[:end, kv_head].T, probs))
        rows.append(torch.stack(heads))
    return torch.stack(rows)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise RuntimeError(f"{name} produced non-finite values")
    nrmse = torch.sqrt(
        torch.mean((actual - expected) ** 2)
        / torch.clamp(torch.mean(expected**2), min=1e-8)
    )
    if float(nrmse) > 0.02:
        raise RuntimeError(f"{name} NRMSE {float(nrmse):.6g} exceeds 0.02")
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.003, msg=name)


def _gather_physical_cpu(
    kv: torch.Tensor, physical: torch.Tensor, length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = physical.to(device=kv.device, dtype=torch.int64)
    key = kv[0].index_select(0, ids).detach().cpu().reshape(-1, 2, 128)[:length]
    value = kv[1].index_select(0, ids).detach().cpu().reshape(-1, 2, 128)[:length]
    return key, value


def _check_compaction(
    core,
    query,
    kmin,
    kmax,
    table,
    seq_lens,
    workspace,
    table_cpu,
    expected_min,
    expected_max,
    length,
    threshold,
    budget_tokens,
):
    compact_table, compact_lengths = core.compact(
        query, kmin, kmax, table, seq_lens, workspace
    )
    live_pages = math.ceil(length / _PAGE)
    for row in range(query.shape[0]):
        ids = table_cpu[row, :live_pages].to(torch.int64)
        expected_scores = _cpu_page_scores(
            query[row : row + 1], expected_min[ids], expected_max[ids]
        )[0]
        expected_logical = _cpu_expected_pages(
            expected_scores,
            live_pages,
            length,
            threshold,
            budget_tokens,
        )
        compact_len = int(compact_lengths[row].item())
        selected_count = math.ceil(compact_len / _PAGE)
        expected_physical = table_cpu[row, expected_logical[:selected_count]]
        actual_physical = compact_table[row, :selected_count].cpu()
        if not torch.equal(actual_physical, expected_physical):
            raise RuntimeError(
                f"compact table differs from CPU page ranking L={length}, "
                f"threshold={threshold}, row={row}, actual={actual_physical.tolist()}, "
                f"expected={expected_physical.tolist()}"
            )
        expected_compact_len = (
            length
            if length <= max(threshold, budget_tokens)
            else (budget_tokens // _PAGE - 1) * _PAGE
            + (length - (live_pages - 1) * _PAGE)
        )
        if compact_len != expected_compact_len:
            raise RuntimeError(
                f"compact length mismatch L={length}, threshold={threshold}: "
                f"{compact_len} != {expected_compact_len}"
            )
    return compact_table.clone(), compact_lengths.clone()


def _call_varlen(fa_utils, query, kcache, vcache, table, cache_len):
    qlen = query.shape[0]
    result = fa_utils.flash_attn_varlen_func(
        q=query,
        k=kcache,
        v=vcache,
        cu_seqlens_q=torch.tensor([0, qlen], dtype=torch.int32, device=query.device),
        cu_seqlens_k=torch.tensor(
            [0, cache_len], dtype=torch.int32, device=query.device
        ),
        max_seqlen_q=qlen,
        max_seqlen_k=cache_len,
        block_table=table,
        softmax_scale=128**-0.5,
        causal=True,
    )
    return result[0] if isinstance(result, tuple) else result


def _startup_long_prefill_abi_check(core, fa_utils, device, max_pages) -> None:
    """Exercise the 17-slot page ABI through the production-length varlen path."""
    cache_len, q_len = _LONG_PREFILL_CACHE_LEN, _LONG_PREFILL_QUERY_LEN
    live_pages = math.ceil(cache_len / _PAGE)
    if live_pages > max_pages:
        raise RuntimeError(
            f"long-prefill pages exceed startup capacity: {live_pages}>{max_pages}"
        )
    generator = torch.Generator(device="cpu").manual_seed(_SELF_CHECK_SEED + cache_len)
    storage_cpu = torch.randn(
        (2, live_pages, _PHYSICAL_PAGE, 2, 128),
        generator=generator,
        dtype=torch.bfloat16,
    ).mul_(0.25)
    storage_cpu[0, :, _PAGE].fill_(-8)
    storage_cpu[1, :, _PAGE].fill_(8)
    storage = storage_cpu.to(device)
    kv = core.padded_view(storage, live_pages)
    kcache, vcache = kv.unbind(0)
    kmin, kmax = core.metadata_views(kv)
    slots = torch.arange(live_pages, dtype=torch.int64, device=device) * _PAGE + (
        _PAGE - 1
    )
    core.update_metadata(kcache, kmin, kmax, slots)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(
        kmin.cpu(), storage_cpu[0, :, :_PAGE].amin(dim=1), rtol=0, atol=0
    )
    torch.testing.assert_close(
        kmax.cpu(), storage_cpu[0, :, :_PAGE].amax(dim=1), rtol=0, atol=0
    )
    dense_kv = kv.contiguous()
    dense_k, dense_v = dense_kv.unbind(0)

    table_cpu = torch.zeros((1, max_pages), dtype=torch.int32)
    table_cpu[0, :live_pages] = torch.roll(
        torch.arange(live_pages, dtype=torch.int32), shifts=1
    )
    table = table_cpu.to(device=device)

    qkv_cpu = torch.randn(
        (q_len, 2560), generator=generator, dtype=torch.bfloat16
    ).mul_(0.1)
    qkv = qkv_cpu.to(device)
    query = qkv[:, :2048].view(q_len, 16, 128)
    tail_before = storage[:, :, _PAGE].clone()
    padded_output = _call_varlen(fa_utils, query, kcache, vcache, table, cache_len)
    contiguous_output = _call_varlen(
        fa_utils, query, dense_k, dense_v, table, cache_len
    )
    if (
        not torch.isfinite(padded_output).all()
        or not torch.isfinite(contiguous_output).all()
    ):
        raise RuntimeError("long-prefill native FA output is non-finite")
    if not torch.equal(padded_output, contiguous_output):
        raise RuntimeError("long-prefill padded17 and contiguous16 outputs differ")
    if not torch.equal(storage[:, :, _PAGE], tail_before):
        raise RuntimeError("long-prefill native FA modified the reserved metadata tail")

    physical = table_cpu[0, :live_pages].to(torch.int64)
    key = (
        storage_cpu[0]
        .index_select(0, physical)[:, :_PAGE]
        .reshape(-1, 2, 128)[:cache_len]
    )
    value = (
        storage_cpu[1]
        .index_select(0, physical)[:, :_PAGE]
        .reshape(-1, 2, 128)[:cache_len]
    )
    for qi in (0, q_len // 2 - 1, q_len - 1):
        end = cache_len - q_len + qi + 1
        reference = _cpu_attention(
            query[qi : qi + 1].cpu(), key[:end], value[:end], end
        )
        _assert_close(
            f"long-prefill suffix-causal CPU FP32 q={qi}",
            padded_output[qi : qi + 1],
            reference,
        )


def _startup_case(core, fa_utils, device, max_pages, length, workspace) -> None:
    """Check padded pages, page selection and both native decode routes."""
    batch = 2
    live_pages = math.ceil(length / _PAGE)
    blocks = batch * live_pages
    generator = torch.Generator(device="cpu").manual_seed(_SELF_CHECK_SEED + length)
    storage_cpu = torch.randn(
        (2, blocks, _PHYSICAL_PAGE, 2, 128),
        generator=generator,
        dtype=torch.bfloat16,
    ).mul_(0.25)
    storage_cpu[0, :, _PAGE].fill_(-8)
    storage_cpu[1, :, _PAGE].fill_(8)
    storage = storage_cpu.to(device)
    kv = core.padded_view(storage, blocks)
    kcache, vcache = kv.unbind(0)
    table_cpu = torch.zeros((batch, max_pages), dtype=torch.int32)
    for row in range(batch):
        ids = torch.arange(row * live_pages, (row + 1) * live_pages, dtype=torch.int32)
        table_cpu[row, :live_pages] = torch.roll(ids, shifts=1)
    table = table_cpu.to(device)
    lengths = torch.full((batch,), length, dtype=torch.int32, device=device)

    kmin, kmax = core.metadata_views(kv)
    slots = torch.arange(blocks, dtype=torch.int64, device=device) * _PAGE + (_PAGE - 1)
    core.update_metadata(kcache, kmin, kmax, slots)
    torch.cuda.synchronize(device)
    expected_min = storage_cpu[0, :, :_PAGE].amin(dim=1)
    expected_max = storage_cpu[0, :, :_PAGE].amax(dim=1)
    torch.testing.assert_close(kmin.cpu(), expected_min, rtol=0, atol=0)
    torch.testing.assert_close(kmax.cpu(), expected_max, rtol=0, atol=0)

    qkv = torch.zeros((batch, 2560), dtype=torch.bfloat16, device=device)
    base_query = torch.linspace(0.03, 0.2, 16 * 128).to(device, torch.bfloat16)
    qkv[0, :2048] = base_query
    qkv[1, :2048] = -base_query
    query = qkv[:, :2048].view(batch, 1, 16, 128)
    selected, selected_lengths = _check_compaction(
        core,
        query[:, 0],
        kmin,
        kmax,
        table,
        lengths,
        workspace,
        table_cpu,
        expected_min,
        expected_max,
        length,
        workspace.dense_threshold,
        _FIXED_BUDGET_TOKENS,
    )
    tail_before = storage[:, :, _PAGE].clone()
    contiguous = kv.contiguous()
    padded_output = None
    for route, keys, values, page_table, seq_lens in (
        ("padded", kcache, vcache, table, lengths),
        ("contiguous", contiguous[0], contiguous[1], table, lengths),
        ("sparse", kcache, vcache, selected, selected_lengths),
    ):
        output = fa_utils.flash_attn_with_kvcache(
            q=query,
            k_cache=keys,
            v_cache=values,
            block_table=page_table,
            cache_seqlens=seq_lens,
            softmax_scale=128**-0.5,
            causal=True,
        )
        if route == "padded":
            padded_output = output
        elif route == "contiguous" and not torch.equal(output, padded_output):
            raise RuntimeError(
                f"padded and contiguous decode outputs differ L={length}"
            )
        for row in range(batch):
            selected_len = int(seq_lens[row].item())
            count = math.ceil(selected_len / _PAGE)
            key, value = _gather_physical_cpu(kv, page_table[row, :count], selected_len)
            expected = _cpu_attention(
                query[row, 0].cpu().unsqueeze(0), key, value, selected_len
            )[0]
            _assert_close(
                f"{route} decode L={length} row={row}", output[row, 0], expected
            )
    if not torch.equal(storage[:, :, _PAGE], tail_before):
        raise RuntimeError(f"native FlashAttention modified page metadata L={length}")


def _startup_adversarial_page_ranking_check(core, device, max_pages, workspace) -> None:
    """Check negative-score and cutoff-tie ordering without running attention."""
    threshold = 1024
    budget_tokens = _FIXED_BUDGET_TOKENS
    if (
        workspace.dense_threshold != threshold
        or workspace.budget_tokens != budget_tokens
        or workspace.max_pages != max_pages
    ):
        raise RuntimeError("adversarial ranking workspace has an unexpected capacity")

    for length, live_pages, negative in ((3000, 188, True), (2049, 129, False)):
        if live_pages > max_pages:
            raise RuntimeError("invalid adversarial startup ranking case")
        logical_to_physical = torch.roll(
            torch.arange(live_pages, dtype=torch.int32), shifts=1
        )
        table_cpu = torch.zeros((1, max_pages), dtype=torch.int32)
        table_cpu[0, :live_pages] = logical_to_physical
        logical_values = torch.arange(1, live_pages + 1, dtype=torch.int32)
        logical_values = logical_values.neg_() if negative else logical_values.zero_()
        physical_values = torch.empty(live_pages, dtype=torch.bfloat16)
        physical_values[logical_to_physical.long()] = logical_values.to(torch.bfloat16)
        kmin_cpu = physical_values[:, None, None].expand(-1, 2, 128).contiguous()
        query = torch.ones((1, 16, 128), dtype=torch.bfloat16, device=device)
        kmin = kmin_cpu.to(device=device)
        table = table_cpu.to(device=device)
        seq_lens = torch.tensor([length], dtype=torch.int32, device=device)
        _check_compaction(
            core,
            query,
            kmin,
            kmin,
            table,
            seq_lens,
            workspace,
            table_cpu,
            kmin_cpu,
            kmin_cpu,
            length,
            threshold,
            budget_tokens,
        )


def _startup_metadata_refresh_check(core, device: torch.device) -> None:
    """Validate the cross-layer refresh against independent CPU reductions."""
    blocks = 5
    jobs_cpu = torch.tensor([3, 1, 4], dtype=torch.int32)
    generator = torch.Generator(device=device).manual_seed(_SELF_CHECK_SEED + 71)
    kv_layers = []
    references = []
    for _ in range(2):
        raw = torch.randn(
            (2, blocks, _PHYSICAL_PAGE, 2, 128),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        kv = core.padded_view(raw, blocks)
        kmin, kmax = core.metadata_views(kv)
        kmin.fill_(37)
        kmax.fill_(-37)
        kv_layers.append(kv)
        references.append((kv[0].detach().float().cpu(), kmin, kmax))

    refresher = core.MultiLayerMetadataRefresher(kv_layers)
    page_ids_device = jobs_cpu.to(device=device)
    with torch.cuda.device(device):
        refresher.refresh(page_ids_device, jobs_cpu.numel())
        torch.cuda.synchronize(device)
    for k_cpu, kmin, kmax in references:
        page_ids = jobs_cpu.long()
        expected_min = k_cpu[page_ids].amin(dim=1).to(torch.bfloat16)
        expected_max = k_cpu[page_ids].amax(dim=1).to(torch.bfloat16)
        torch.testing.assert_close(kmin[page_ids_device].cpu(), expected_min)
        torch.testing.assert_close(kmax[page_ids_device].cpu(), expected_max)

    changed = torch.randn(
        (16, 2, 128),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    kv_layers[1][0, 1].copy_(changed)
    with torch.cuda.device(device):
        refresher.refresh(torch.tensor([1], dtype=torch.int32, device=device), 1)
        torch.cuda.synchronize(device)
    expected_min = changed.float().amin(dim=0).to(torch.bfloat16)
    expected_max = changed.float().amax(dim=0).to(torch.bfloat16)
    _, kmin, kmax = references[1]
    torch.testing.assert_close(kmin[1].cpu(), expected_min.cpu())
    torch.testing.assert_close(kmax[1].cpu(), expected_max.cpu())


def _startup_self_check(
    core, device: torch.device, max_pages: int, threshold: int
) -> None:
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils import fa_utils

    if not fa_utils.is_flash_attn_varlen_func_available():
        raise RuntimeError("native FlashAttention varlen entry point is unavailable")
    _startup_metadata_refresh_check(core, device)
    thresholds = sorted({threshold, 1024})
    workspaces = {
        dense_threshold: core.Workspace(
            device=device,
            max_seqs=2,
            max_pages=max_pages,
            dense_threshold=dense_threshold,
            budget_tokens=_FIXED_BUDGET_TOKENS,
        )
        for dense_threshold in thresholds
    }
    effective_dense_limit = max(threshold, _FIXED_BUDGET_TOKENS)
    startup_lengths = tuple(
        length
        for length in dict.fromkeys(
            (*_SELF_CHECK_LENGTHS, effective_dense_limit, effective_dense_limit + 1)
        )
        if 0 < length <= max_pages * _PAGE
    )
    if not startup_lengths:
        raise RuntimeError(
            "no startup self-check length fits the configured KV page capacity"
        )
    try:
        with torch.cuda.device(device):
            for length in startup_lengths:
                _startup_case(
                    core,
                    fa_utils,
                    device,
                    max_pages,
                    length,
                    workspaces[threshold],
                )
            _startup_long_prefill_abi_check(core, fa_utils, device, max_pages)
            _startup_adversarial_page_ranking_check(
                core, device, max_pages, workspaces[1024]
            )
        torch.cuda.synchronize(device)
    finally:
        del workspaces
        gc.collect()
        torch.cuda.empty_cache()


def _make_graph_pools(core, device, batch, length, workspace, seed):
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils import fa_utils

    page_count = math.ceil(length / _PAGE)
    blocks = batch * page_count
    generator = torch.Generator(device=device).manual_seed(seed)
    padded_storage = torch.randn(
        (2, blocks, _PHYSICAL_PAGE, 2, 128),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    ).mul_(0.25)
    padded_kv = core.padded_view(padded_storage, blocks)
    contiguous_kv = padded_kv.contiguous()
    padded_k, padded_v = padded_kv.unbind(0)
    contiguous_k, contiguous_v = contiguous_kv.unbind(0)

    block_table = torch.zeros(
        (batch, workspace.max_pages), dtype=torch.int32, device=device
    )
    for row in range(batch):
        ids = torch.arange(
            row * page_count, (row + 1) * page_count, dtype=torch.int32, device=device
        )
        block_table[row, :page_count] = torch.roll(ids, shifts=1)
    seq_lens = torch.full((batch,), length, dtype=torch.int32, device=device)
    seq_lens[1::2] -= 1
    last_page = torch.div(seq_lens - 1, _PAGE, rounding_mode="floor")
    last_block = block_table.gather(1, last_page[:, None]).squeeze(1)
    slot_mapping = (
        last_block.to(torch.int64) * _PAGE + (seq_lens.to(torch.int64) - 1) % _PAGE
    )

    qkv = torch.zeros((batch, 2560), dtype=torch.bfloat16, device=device)
    qbase = torch.linspace(-0.2, 0.2, 16 * 128, device=device).to(torch.bfloat16)
    qkv[:, : 16 * 128] = qbase
    query = qkv[:, : 16 * 128].view(batch, 1, 16, 128)
    key = torch.randn(
        (batch, 2, 128), generator=generator, device=device, dtype=torch.bfloat16
    ).mul_(0.1)
    value = torch.randn(
        (batch, 2, 128), generator=generator, device=device, dtype=torch.bfloat16
    ).mul_(0.1)
    scales = torch.ones((1,), dtype=torch.float32, device=device)
    kmin, kmax = core.metadata_views(padded_kv)
    kmin.copy_(padded_k.amin(dim=1))
    kmax.copy_(padded_k.amax(dim=1))
    return {
        "padded_k": padded_k,
        "padded_v": padded_v,
        "contiguous_k": contiguous_k,
        "contiguous_v": contiguous_v,
        "table": block_table,
        "seq_lens": seq_lens,
        "max_seq_len": int(length),
        "slot_mapping": slot_mapping,
        "query": query,
        "key": key,
        "value": value,
        "scales": scales,
        "kmin": kmin,
        "kmax": kmax,
        "logical_kv_bytes": contiguous_kv.numel() * contiguous_kv.element_size(),
        "padded_kv_bytes": padded_storage.numel() * padded_storage.element_size(),
        "writer": fa_utils.reshape_and_cache_flash,
    }


def _run_graph_call(core, fa_utils, pool, sparse, workspace):
    use_sparse_selector = sparse and pool["max_seq_len"] > workspace.dense_limit
    pool["writer"](
        pool["key"],
        pool["value"],
        pool["padded_k"] if sparse else pool["contiguous_k"],
        pool["padded_v"] if sparse else pool["contiguous_v"],
        pool["slot_mapping"],
        "auto",
        pool["scales"],
        pool["scales"],
    )
    if sparse:
        core.update_metadata(
            pool["padded_k"],
            pool["kmin"],
            pool["kmax"],
            pool["slot_mapping"],
        )
        if use_sparse_selector:
            block_table, seq_lens = core.compact(
                pool["query"].view(pool["query"].shape[0], 16, 128),
                pool["kmin"],
                pool["kmax"],
                pool["table"],
                pool["seq_lens"],
                workspace,
            )
        else:
            block_table, seq_lens = pool["table"], pool["seq_lens"]
        kcache, vcache = pool["padded_k"], pool["padded_v"]
    else:
        block_table, seq_lens = pool["table"], pool["seq_lens"]
        kcache, vcache = pool["contiguous_k"], pool["contiguous_v"]
    return fa_utils.flash_attn_with_kvcache(
        q=pool["query"],
        k_cache=kcache,
        v_cache=vcache,
        block_table=block_table,
        cache_seqlens=seq_lens,
        softmax_scale=128**-0.5,
        causal=True,
    )


def _prepare_graph_arm(core, fa_utils, pools, sparse, workspace, device):
    eager_outputs = [
        _run_graph_call(core, fa_utils, pool, sparse, workspace) for pool in pools
    ]
    torch.cuda.synchronize(device)
    if any(not torch.isfinite(output).all() for output in eager_outputs):
        raise RuntimeError(
            f"native sparse graph eager output is non-finite sparse={sparse}"
        )

    outputs = []
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for pool in pools * 4:
            outputs.append(_run_graph_call(core, fa_utils, pool, sparse, workspace))
    return {"graph": graph, "outputs": outputs, "eager_outputs": eager_outputs}


def _measure_graph(graph, device) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.device(device):
        start.record()
        graph.replay()
        end.record()
    end.synchronize()
    elapsed = float(start.elapsed_time(end))
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError(f"invalid sparse attention Graph event time {elapsed}")
    return elapsed


def graph_pair_medians(device, baseline, candidate, *, rounds=1):
    """Time paired graphs in ABBA order, rotating KV pools when available."""
    samples = ([], [])
    for _ in range(rounds):
        for index, arm in enumerate((0, 1, 1, 0)):
            graphs = (baseline, candidate)[arm]
            samples[arm].append(_measure_graph(graphs[index % len(graphs)], device))
    return statistics.median(samples[0]), statistics.median(samples[1])


def _benchmark_sparse_case(core, fa_utils, device, workspace, batch, length, l2_bytes):
    pools = [
        _make_graph_pools(
            core, device, batch, length, workspace, _SELF_CHECK_SEED + batch + index
        )
        for index in range(3)
    ]
    if any(
        pool["logical_kv_bytes"] <= l2_bytes or pool["padded_kv_bytes"] <= l2_bytes
        for pool in pools
    ):
        raise RuntimeError(
            f"B{batch} L{length} rotating KV pools must each exceed device L2={l2_bytes}"
        )
    baseline = candidate = None
    try:
        baseline = _prepare_graph_arm(core, fa_utils, pools, False, workspace, device)
        candidate = _prepare_graph_arm(core, fa_utils, pools, True, workspace, device)
        baseline["graph"].replay()
        candidate["graph"].replay()
        torch.cuda.synchronize(device)
        for label, arm in (("baseline", baseline), ("candidate", candidate)):
            for index, output in enumerate(arm["outputs"]):
                eager = arm["eager_outputs"][index % len(pools)]
                if not torch.isfinite(output).all():
                    raise RuntimeError(
                        f"B{batch} L{length} {label} Graph output non-finite"
                    )
                torch.testing.assert_close(
                    output.float(), eager.float(), rtol=0.01, atol=0.01
                )

        graphs = (baseline["graph"],), (candidate["graph"],)
        graph_pair_medians(device, *graphs)  # Warmup.
        baseline_ms, candidate_ms = graph_pair_medians(device, *graphs)
        graph_calls = len(baseline["outputs"])
        result = {
            "batch": batch,
            "max_seq_len": length,
            "baseline_median_ms_per_call": baseline_ms / graph_calls,
            "candidate_median_ms_per_call": candidate_ms / graph_calls,
        }
        logger.info("Native sparse full-chain Graph ABBA: %s", result)
        return result
    finally:
        baseline = candidate = graphs = pools = None
        gc.collect()
        torch.cuda.empty_cache()


def _benchmark_sparse_graphs(core, device, workspace):
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils import fa_utils

    properties = torch.cuda.get_device_properties(device)
    l2_bytes = int(
        getattr(properties, "L2_cache_size", 0)
        or getattr(properties, "l2_cache_size", 0)
    )
    if l2_bytes <= 0:
        raise RuntimeError("device L2 cache size is unavailable")
    if not fa_utils.is_flash_attn_varlen_func_available():
        raise RuntimeError("native FlashAttention entry points are unavailable")
    results = []
    with torch.cuda.device(device):
        for batch, length in ((64, 4224), (32, 16512)):
            results.append(
                _benchmark_sparse_case(
                    core, fa_utils, device, workspace, batch, length, l2_bytes
                )
            )
    passes = len(results) == 2 and sparse_graph_gate_passes(
        results, int(workspace.dense_limit)
    )
    logger.info(
        "Native sparse startup Graph cost gate: enabled=%s results=%s", passes, results
    )
    return passes


def install(vllm_config, device: torch.device) -> bool:
    """Run correctness and full-chain Graph gates before changing KV layout."""
    global _STATE
    disable("install pending")
    if device.type != "cuda" or torch.version.hip is not None:
        _STATE.reason = "unsupported device"
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    profile = device_profiles.lookup(index) or {}
    requested = envs.switch(_SWITCH, auto=profile.get("native_sparse_decode") is True)
    if not requested:
        _STATE.reason = "switch disabled"
        return False

    workspace = None
    try:
        threshold, budget, max_seqs = _validate_config(vllm_config, device, profile)
        from vllm_fl.attention import native_sparse_decode as core

        max_pages = math.ceil(vllm_config.model_config.max_model_len / _PAGE)
        _startup_self_check(core, device, max_pages, threshold)
        workspace = core.Workspace(
            device=device,
            max_seqs=max_seqs,
            max_pages=max_pages,
            dense_threshold=threshold,
            budget_tokens=budget,
        )
        if not _benchmark_sparse_graphs(core, device, workspace):
            _STATE.reason = "full-chain startup Graph cost gate failed"
            logger.info(
                "Native sparse decode remains disabled after startup Graph gate"
            )
            return False
        _STATE = State(
            workspace=workspace,
            enabled=True,
            threshold=threshold,
            budget_tokens=budget,
            reason="startup correctness and full-chain Graph cost gates passed",
        )
        workspace = None
        logger.info(
            "Native sparse decode enabled after correctness and full-chain Graph gates: "
            "threshold=%d budget_tokens=%d max_seqs=%d max_pages=%d workspace_bytes=%d",
            threshold,
            budget,
            max_seqs,
            max_pages,
            _STATE.workspace.nbytes,
        )
        try:
            from vllm_fl.attention import native_sparse_prefill

            native_sparse_prefill.install(
                vllm_config,
                device,
                profile,
                _STATE.workspace,
            )
        except Exception:
            logger.warning(
                "Native sparse prefill initialization failed; sparse decode remains enabled",
                exc_info=True,
            )
        return True
    except Exception as exc:
        disable(f"startup gate failed: {type(exc).__name__}: {exc}")
        logger.warning(
            "Native sparse decode startup gate failed; retaining contiguous KV layout",
            exc_info=True,
        )
        return False
    finally:
        if workspace is not None:
            del workspace
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                logger.debug(
                    "Could not release native sparse startup workspace", exc_info=True
                )


def spec_is_supported(spec, backend, layout_order) -> bool:
    """Strict layout/spec predicate used before changing the KV cache page size."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

    return (
        type(spec) is FullAttentionSpec
        and is_native_backend(backend)
        and tuple(layout_order) == (0, 1, 2, 3, 4)
        and spec.dtype == torch.bfloat16
        and spec.kv_quant_mode == KVQuantMode.NONE
        and spec.page_size_padded is None
        and spec.sliding_window is None
        and spec.attention_chunk_size is None
        and not spec.non_causal
        and (spec.block_size, spec.num_kv_heads, spec.head_size, spec.head_size_v)
        == (_PAGE, 2, 128, 128)
    )


def is_native_backend(backend) -> bool:
    return getattr(
        backend, "__name__", None
    ) == "MacaFlashAttentionBackend" and getattr(backend, "__module__", "").endswith(
        "vendor.metax.impl.attention.flash_attn"
    )


def padded_page_size() -> int:
    return 2 * _PHYSICAL_PAGE * 2 * 128 * 2


def padded_view(raw: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """Expose the native core's zero-copy 17-slot cache view."""
    from vllm_fl.attention import native_sparse_decode as core

    return core.padded_view(raw, num_blocks)


def impl_is_supported(impl) -> bool:
    return (
        getattr(impl, "sliding_window", None) in (None, (-1, -1))
        and getattr(impl, "alibi_slopes", None) is None
        and getattr(impl, "sinks", None) is None
        and getattr(impl, "logits_soft_cap", None) in (None, 0.0)
    )


def bind_kv_metadata(kv: torch.Tensor, impl) -> None:
    """Bind aliases once after cache construction, before graph capture."""
    if not is_enabled():
        return
    from vllm_fl.attention import native_sparse_decode as core

    if kv.dtype != torch.bfloat16 or kv.ndim != 5:
        raise RuntimeError(
            f"unexpected native sparse cache: {kv.dtype} {tuple(kv.shape)}"
        )
    impl._fl_native_sparse_metadata = core.metadata_views(kv)
    impl._fl_native_sparse_workspace = _STATE.workspace
    impl._fl_native_sparse_threshold = _STATE.threshold


def is_padded_cache(kv: torch.Tensor) -> bool:
    return (
        kv.dtype == torch.bfloat16
        and kv.ndim == 5
        and tuple(kv.shape[0:1]) == (2,)
        and tuple(kv.shape[2:]) == (_PAGE, 2, 128)
        and kv.stride(1) == _PHYSICAL_PAGE * 2 * 128
    )


def should_compact_decode(impl, attn_metadata) -> bool:
    """Choose the decode route without inspecting device sequence lengths."""
    if not is_enabled() or getattr(impl, "_fl_native_sparse_metadata", None) is None:
        return False
    from vllm.forward_context import get_forward_context

    forward_context = get_forward_context()
    max_seq_len = int(attn_metadata.decode_max_seq_len)
    dense_limit = effective_dense_limit()
    sparse = should_use_sparse_route(
        forward_context.batch_descriptor,
        forward_context.cudagraph_runtime_mode,
        max_seq_len,
        dense_limit,
    )
    return sparse


def log_runtime_route_once(
    sparse: bool, max_seq_len: int, dense_limit: int, runtime_mode: CUDAGraphMode
) -> None:
    route = "sparse" if sparse else "dense"
    if route in _STATE.logged_runtime_routes:
        return
    _STATE.logged_runtime_routes.add(route)
    logger.info(
        "Native sparse decode runtime route: path=%s max_seq_len=%d dense_limit=%d mode=%s",
        route,
        max_seq_len,
        dense_limit,
        runtime_mode,
    )


def compact_decode(impl, query, table, seq_lens):
    if not is_enabled() or not hasattr(impl, "_fl_native_sparse_workspace"):
        return None
    from vllm_fl.attention import native_sparse_decode as core

    kmin, kmax = impl._fl_native_sparse_metadata
    return core.compact(
        query,
        kmin,
        kmax,
        table,
        seq_lens,
        impl._fl_native_sparse_workspace,
    )
