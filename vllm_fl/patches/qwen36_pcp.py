# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6 hybrid (GDN) + Prefill Context Parallel (PCP) patches for vLLM.

Enables tensor-parallel + prefill-context-parallel serving of Qwen3.6-style
hybrid models (interleaved full-attention + GatedDeltaNet/linear-attention
layers) on an unmodified vLLM 0.20.2, via monkey-patches collected here and
installed from ``apply_platform_patches()``.

Every patch self-gates on ``prefill_context_parallel_size > 1`` (equivalently
``get_pcp_group().world_size > 1``): with PCP disabled the wrappers fall
through to the original vLLM code, so installing them is a no-op for
non-PCP runs.

SCOPE: prefill correctness only (verified: argmax token[0] matches the tp=4
baseline).  Decode is out of scope — full-attention KV is sequence-split and
the decode KV-cache path is not handled, so multi-token generation drifts /
produces NaN.  See the module README / PR description for known limitations
(tp*pcp>4 crash, generate() teardown stall).

Approach (step-1, correctness-first / all-gather):
  - Each PCP rank owns a contiguous slice of the prefill tokens.
  - All-gather raw K/V across PCP ranks (rank order == sequence order).
  - Each rank attends to tokens up to its own segment end (causal prefix).
  - flash_attn_varlen_func with causal=True + end-alignment handles masking.

Sequence splitting (Step A+B):
  - Scheduler output is unchanged (num_scheduled_tokens = full prefill length).
  - GPUModelRunner.prepare_inputs is patched to present a LOCAL view:
      * local_len = ceil(full_len / P) for ranks 0..P-2
      * local_len = full_len - (P-1)*ceil(full_len/P) for rank P-1
    so that ranks always see seq_lens <= prefill_len (avoids false "decode" checks).
  - num_computed_tokens is temporarily shifted by pcp_rank * ceil(full_len/P) so
    that the existing Triton kernels (_prepare_prefill_inputs / _prepare_pos_seq_lens)
    naturally read the correct token slice and write the correct absolute positions.
  - The shift is restored after the kernels are enqueued (CUDA-stream-ordered).

TODO step-2: replace all-gather with ring P2P send/recv to save memory.
TODO step-3: striped/zigzag assignment for load balance.
TODO: multi-request batch with unequal lengths (need per-request K reorder).
"""

import logging
import math
import os
import sys

import torch

logger = logging.getLogger(__name__)

_PCP_DEBUG = os.environ.get("VLLM_PCP_DEBUG", "") not in ("", "0", "false", "False")

# When set to a path prefix, dump the FIRST compute_logits output (the prefill
# last-position logits over the sampled token) to "<prefix>.pt" on global rank
# 0, then keep running.  Config-agnostic: fires for both pcp==1 (baseline) and
# pcp>1, at the same logical point (post-broadcast for PCP), so the two dumps
# are element-wise comparable.
_PCP_DUMP = os.environ.get("VLLM_PCP_DUMP", "")


def _dbg(msg: str) -> None:
    if _PCP_DEBUG:
        print(f"[PCP-DBG] {msg}", file=sys.stderr, flush=True)


def _install_pcp_logits_dump(runner) -> None:
    """Wrap runner.model.compute_logits to torch.save the first prefill logits.

    Installed OUTSIDE the PCP broadcast gather (called first in prepare_inputs)
    so that, for pcp>1, it observes the broadcast-corrected logits.  Only the
    global-rank-0 process writes a file; the save happens inside the forward so
    it survives even if teardown later hangs.
    """
    if not _PCP_DUMP:
        return
    if getattr(runner, "_pcp_logits_dump_installed", False):
        return
    model = getattr(runner, "model", None)
    if model is None or not hasattr(model, "compute_logits"):
        return

    _orig = model.compute_logits

    def _dumping_compute_logits(hidden_states, *args, **kwargs):
        logits = _orig(hidden_states, *args, **kwargs)
        try:
            if logits is not None and not getattr(runner, "_pcp_logits_dumped", False):
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
                if rank == 0:
                    path = f"{_PCP_DUMP}.pt"
                    torch.save(logits.detach().float().cpu(), path)
                    runner._pcp_logits_dumped = True
                    _dbg(f"DUMP wrote first-forward logits shape={tuple(logits.shape)} "
                         f"argmax={logits.float().argmax(dim=-1).tolist()} -> {path}")
        except Exception as e:
            _dbg(f"DUMP failed: {e!r}")
        return logits

    model.compute_logits = _dumping_compute_logits
    runner._pcp_logits_dump_installed = True
    logger.info("vllm_fl: PCP logits dump installed")


# ---------------------------------------------------------------------------
# FlashAttentionImpl patch  (Step 1: PCP attention with all-gather K/V)
# ---------------------------------------------------------------------------

def _build_attn_kv_gather(all_qsl_local, rank: int, max_total: int):
    """
    Build the index that maps the rank-major, per-rank-PADDED all-gather buffer
    into this rank's per-request CAUSAL-PREFIX K/V layout.

    all_qsl_local : np.ndarray [P, B]  — every rank's LOCAL per-request seqlens.
    Padded gather layout: rank r' occupies rows [r'*max_total, r'*max_total+max_total);
    within that block, request i starts at cumsum of that rank's earlier requests.

    For this rank, request i needs K/V from ranks 0..rank (its causal prefix),
    laid out per-request-contiguous so flash_attn_varlen_func with causal=True
    (end-aligned) attends Q[rank's chunk] against exactly its causal prefix.

    Returns (kidx int64 [total_k], cu_seqlens_k int32 [B+1], max_k_len int).
    """
    import numpy as np
    a = all_qsl_local.astype(np.int64)          # [P, B]
    P, B = a.shape
    local_cumoff = np.zeros((P, B), dtype=np.int64)
    local_cumoff[:, 1:] = np.cumsum(a[:, :-1], axis=1)   # offset of req i within rank r'

    k_len = a[: rank + 1].sum(axis=0)           # [B] causal-prefix length per request
    cu_k = np.zeros(B + 1, dtype=np.int32)
    cu_k[1:] = np.cumsum(k_len)
    total_k = int(cu_k[-1])

    kidx = np.empty(total_k, dtype=np.int64)
    dst = 0
    for i in range(B):
        for rp in range(rank + 1):              # ranks 0..rank = causal prefix
            L = int(a[rp, i])
            if L == 0:
                continue
            src = rp * max_total + int(local_cumoff[rp, i])
            kidx[dst:dst + L] = np.arange(src, src + L)
            dst += L
    max_k_len = int(k_len.max()) if B > 0 else 0
    return kidx, cu_k, max_k_len


def _forward_with_pcp(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    attn_metadata,
) -> None:
    from vllm.distributed.parallel_state import get_pcp_group

    try:
        from vllm.v1.attention.backends.flash_attn import flash_attn_varlen_func
    except ImportError:
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func  # fallback

    pcp_group = get_pcp_group()
    P = pcp_group.world_size
    rank = pcp_group.rank_in_group

    assert self.vllm_flash_attn_version is not None, "FlashAttention version not detected"

    # 1. Per-request local seqlens for this rank, then all-gather → [P, B].
    #    local_q_seqlens[i] = tokens this rank holds for request i.
    cu_q = attn_metadata.query_start_loc          # [B+1], int32
    local_q_seqlens = (cu_q[1:] - cu_q[:-1]).to(torch.int32)  # [B]
    all_q_seqlens = pcp_group.all_gather(
        local_q_seqlens.contiguous(), dim=0
    ).view(P, -1)                                 # [P, B]
    all_qsl_cpu = all_q_seqlens.cpu().numpy()

    # 2. All-gather K/V.  CRITICAL: every rank must contribute the SAME number
    #    of rows (all_gather_into_tensor requires equal sizes), but contiguous
    #    PCP split gives unequal local token counts.  So pad each rank's K/V to
    #    max_total rows before gathering, then reshuffle out the real tokens.
    n_local = key.shape[0]
    cuq_total = int(cu_q[-1].item())
    n_local_t = torch.tensor([n_local], device=query.device, dtype=torch.int32)
    all_n_local = pcp_group.all_gather(n_local_t, dim=0).cpu().tolist()  # [P]
    max_total = int(max(all_n_local))
    _dbg(f"attn r{rank} PRE: n_local={n_local} cuq_total={cuq_total} "
         f"num_actual={attn_metadata.num_actual_tokens} max_q={attn_metadata.max_query_len} "
         f"all_n_local={all_n_local} local_qsl_sum={int(local_q_seqlens.sum())} "
         f"all_qsl={all_qsl_cpu.tolist()}")
    Hk = key.shape[1:]
    key_pad = key.new_zeros((max_total,) + Hk)
    val_pad = value.new_zeros((max_total,) + Hk)
    key_pad[:n_local] = key
    val_pad[:n_local] = value
    key_gathered = pcp_group.all_gather(key_pad.contiguous(), dim=0)  # [P*max_total, ...]
    val_gathered = pcp_group.all_gather(val_pad.contiguous(), dim=0)

    # 3. Reshuffle into this rank's per-request causal-prefix (ranks 0..rank).
    kidx_np, cu_k_np, max_k_len = _build_attn_kv_gather(all_qsl_cpu, rank, max_total)
    kidx = torch.as_tensor(kidx_np, device=query.device, dtype=torch.long)
    key_causal = key_gathered.index_select(0, kidx)
    val_causal = val_gathered.index_select(0, kidx)
    cu_seqlens_k = torch.as_tensor(cu_k_np, device=query.device, dtype=torch.int32)

    _dbg(f"attn r{rank}: n_local={n_local} max_total={max_total} "
         f"k_causal={tuple(key_causal.shape)} max_k_len={max_k_len} "
         f"k_nan={int(torch.isnan(key).sum())} kc_nan={int(torch.isnan(key_causal).sum())}")

    sliding_window_size = (
        list(self.sliding_window) if self.sliding_window is not None else None
    )

    # 4. Flash attention: Q_local vs K_causal with causal=True.
    #    flash_attn end-alignment: Q[i] attends to K[0 .. K_len - Q_len + i],
    #    which is exactly the causal prefix for rank r's segment.
    flash_attn_varlen_func(
        q=query,
        k=key_causal,
        v=val_causal,
        out=output,
        cu_seqlens_q=cu_q,
        max_seqlen_q=attn_metadata.max_query_len,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=max_k_len,
        softmax_scale=self.scale,
        causal=True,
        alibi_slopes=self.alibi_slopes,
        window_size=sliding_window_size,
        softcap=self.logits_soft_cap,
        fa_version=self.vllm_flash_attn_version,
    )


def _patched_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, **kwargs):
    """Wraps the original forward to intercept the PCP prefill path."""
    from vllm.distributed.parallel_state import get_pcp_group
    try:
        _pcp_P = get_pcp_group().world_size
    except AssertionError:
        _pcp_P = 1

    if (
        _pcp_P > 1
        and attn_metadata is not None
        and not attn_metadata.use_cascade
        and attn_metadata.max_query_len > 1  # skip decode (each request has 1 query token)
    ):
        # The real per-rank local token count is query_start_loc[-1], NOT
        # num_actual_tokens.  Under PCP the runner allocates a buffer sized to
        # the (padded) full length but only fills the local slice; the tail
        # rows are garbage/NaN.  Use the cu_q length so attention matches the
        # GDN island path (which also keys off query_start_loc[-1]).
        n_real = int(attn_metadata.query_start_loc[-1].item())
        n_buf = attn_metadata.num_actual_tokens
        self._forward_with_pcp(
            query[:n_real],
            key[:n_real],
            value[:n_real],
            output[:n_real],
            attn_metadata,
        )
        # Kill any pre-existing NaN in the padded tail so it can't poison the
        # residual stream at phantom positions.
        if n_buf > n_real:
            output[n_real:n_buf].zero_()
        _dbg(f"attn PCP path taken: max_q={attn_metadata.max_query_len} "
             f"n_real={n_real} n_buf={n_buf} out_norm={output[:n_real].float().norm().item():.3f}")
        return output

    if _pcp_P > 1:
        _dbg(f"attn FELL THROUGH to original (P={_pcp_P}): "
             f"md={attn_metadata is not None} "
             f"cascade={getattr(attn_metadata,'use_cascade',None)} "
             f"max_q={getattr(attn_metadata,'max_query_len',None)}")
    return self._original_forward(layer, query, key, value, kv_cache, attn_metadata, output, **kwargs)


def apply_pcp_attn_patch() -> None:
    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    except ImportError:
        logger.warning("vllm_fl: FlashAttentionImpl not found, skipping attn patch")
        return

    if getattr(FlashAttentionImpl, "_pcp_patched", False):
        return  # idempotent

    FlashAttentionImpl._forward_with_pcp = _forward_with_pcp
    FlashAttentionImpl._original_forward = FlashAttentionImpl.forward
    FlashAttentionImpl.forward = _patched_forward
    FlashAttentionImpl.supports_pcp = True
    FlashAttentionImpl._pcp_patched = True
    logger.info("vllm_fl: PCP patch applied to FlashAttentionImpl")


# ---------------------------------------------------------------------------
# GPUModelRunner patch  (Step A+B: sequence splitting in _prepare_inputs)
#
# With PCP enabled, vLLM automatically falls back to the V1 model runner
# (vllm.v1.worker.gpu_model_runner.GPUModelRunner).  Qwen3-like hybrid
# models (has_inner_state=True) also assert V1.  The V2 model runner
# (vllm.v1.worker.gpu.model_runner.GPUModelRunner) is NOT used.
# ---------------------------------------------------------------------------

def _compute_pcp_local_len(full_len: int, P: int, r: int) -> tuple[int, int]:
    """
    Returns (local_len, pcp_offset) for rank r.

    Uses ceiling division so that local slices partition [0, full_len) exactly:
      rank 0 .. P-2 : local_len = ceil(full_len / P)
      rank P-1      : local_len = full_len - (P-1) * ceil(full_len / P)

    pcp_offset = r * ceil(full_len / P)  — start index into all_token_ids.
    """
    local_len_base = math.ceil(full_len / P)
    pcp_offset = r * local_len_base
    local_len = max(0, min(local_len_base, full_len - pcp_offset))
    return local_len, pcp_offset


class _V1PcpSchedWrapper:
    """
    Thin wrapper that overrides total_num_scheduled_tokens to the local sum
    so _prepare_inputs allocates correctly-sized GPU buffers.

    All other attributes are forwarded to the original SchedulerOutput.
    The original object is left untouched — _update_states_after_model_execute
    will still see the full token counts and advance num_computed_tokens by the
    full prefill length.
    """

    def __init__(self, original, local_total: int):
        self._original = original
        self.total_num_scheduled_tokens = local_total

    def __getattr__(self, name: str):
        return getattr(self._original, name)


def _patched_v1_prepare_inputs(self, scheduler_output, num_scheduled_tokens_np):
    """
    Intercepts V1 GPUModelRunner._prepare_inputs to split prefill sequences.

    V1's _prepare_inputs receives a numpy array num_scheduled_tokens_np
    (one entry per active request slot, in input_batch.req_ids order).

    For each prefill request:
      1. Replace num_scheduled_tokens_np[slot] with local_len.
      2. Add pcp_offset to input_batch.num_computed_tokens_cpu[slot].
         (cpu and cpu_tensor share memory, so both are updated automatically.)
      3. Wrap scheduler_output to present the local total token count.
      4. Call original _prepare_inputs.
      5. Restore num_computed_tokens_cpu (stream-ordered, always safe).

    Effect inside _prepare_inputs:
      positions_np = num_computed_tokens_cpu[req_indices] + query_pos_np
        → absolute positions for rank-r slice  ✓
      token_indices = positions_np + req_indices * max_model_len
        → reads token_ids[pcp_offset : pcp_offset + local_len]  ✓
      self.positions (GPU) = num_computed_tokens_gpu[req_indices] + query_pos_gpu
        (num_computed_tokens_gpu is copied from cpu_tensor; shares memory)  ✓
    """
    # Activate only under real prefill context parallelism.  vLLM does not
    # expose a `use_pcp` flag on the runner, so derive it from the config.
    try:
        pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
    except AttributeError:
        pcp_size = 1
    # Config-agnostic logits dump (fires for pcp==1 baseline too).  Install
    # before the gather so it sits innermost and observes corrected logits.
    _install_pcp_logits_dump(self)
    if pcp_size <= 1:
        return self._original_prepare_inputs(scheduler_output, num_scheduled_tokens_np)
    # Ensure the PCP sample_hidden_states gather is installed on the LM head.
    _install_pcp_logits_gather(self)
    from vllm.distributed.parallel_state import get_pcp_group
    pcp_group = get_pcp_group()
    P = pcp_group.world_size
    r = pcp_group.rank_in_group

    if _PCP_DEBUG:
        _states = []
        for _sidx, _rid in enumerate(list(self.input_batch.req_ids[:self.input_batch.num_reqs])):
            _rs = self.requests.get(_rid)
            if _rs is not None:
                _states.append((int(num_scheduled_tokens_np[_sidx]),
                                _rs.num_computed_tokens, _rs.num_prompt_tokens))
        _dbg(f"prepare_inputs r{r}: num_reqs={self.input_batch.num_reqs} "
             f"(sched,computed,prompt)={_states}")

    num_reqs = self.input_batch.num_reqs
    req_ids = list(self.input_batch.req_ids[:num_reqs])

    # Identify prefill requests.
    # req_state.num_computed_tokens < num_prompt_tokens → still in prefill.
    # pcp_info: slot_idx -> (local_len, pcp_offset)
    pcp_info: dict[int, tuple[int, int]] = {}
    for slot_idx, req_id in enumerate(req_ids):
        req_state = self.requests.get(req_id)
        if req_state is None:
            continue
        full_len = int(num_scheduled_tokens_np[slot_idx])
        if req_state.num_computed_tokens < req_state.num_prompt_tokens:
            local_len, pcp_offset = _compute_pcp_local_len(full_len, P, r)
            pcp_info[slot_idx] = (local_len, pcp_offset)

    if not pcp_info:
        return self._original_prepare_inputs(scheduler_output, num_scheduled_tokens_np)
    # Build modified num_scheduled_tokens array (local_len for prefills).
    import numpy as np
    modified_np = num_scheduled_tokens_np.copy()
    for slot_idx, (local_len, _) in pcp_info.items():
        modified_np[slot_idx] = local_len

    local_total = int(modified_np.sum())
    modified_sched = _V1PcpSchedWrapper(scheduler_output, local_total)

    # Temporarily shift num_computed_tokens_cpu by pcp_offset.
    # num_computed_tokens_cpu and num_computed_tokens_cpu_tensor share memory.
    idx_offsets = [(slot_idx, pcp_offset) for slot_idx, (_, pcp_offset) in pcp_info.items()]
    for slot_idx, offset in idx_offsets:
        self.input_batch.num_computed_tokens_cpu[slot_idx] += offset

    try:
        result = self._original_prepare_inputs(modified_sched, modified_np)
    finally:
        for slot_idx, offset in idx_offsets:
            self.input_batch.num_computed_tokens_cpu[slot_idx] -= offset

    return result


def _install_pcp_logits_gather(runner) -> None:
    """
    Lazily wrap ``runner.model.compute_logits`` so that, under PCP, the
    sample_hidden_states fed to the LM head come from the LAST PCP rank.

    Why: with contiguous sequence splitting, request i's final prompt token
    lives on the last PCP rank.  ``execute_model`` computes
    ``sample_hidden_states = hidden_states[logits_indices]`` locally on every
    rank, but the sampled token is taken from the *driver* (global rank 0),
    whose local slice ends at the wrong token → garbage logits.

    Fix: in-place broadcast sample_hidden_states from the last PCP rank
    (src = world_size - 1, a group-local rank) to all PCP ranks before the LM
    head runs.  compute_logits is called on every rank, so the collective
    stays in sync.  ``hidden_states[logits_indices]`` is a fresh contiguous
    tensor (advanced indexing copies), so the in-place write is safe and does
    not corrupt the underlying activations.
    """
    if getattr(runner, "_pcp_logits_gather_installed", False):
        return
    model = getattr(runner, "model", None)
    if model is None or not hasattr(model, "compute_logits"):
        return

    _orig_compute_logits = model.compute_logits

    def _wrapped_compute_logits(hidden_states, *args, **kwargs):
        if hidden_states is not None:
            from vllm.distributed.parallel_state import get_pcp_group
            try:
                pcp_group = get_pcp_group()
                P = pcp_group.world_size
            except AssertionError:
                P = 1
            if P > 1:
                # src is the group-local rank of the last PCP rank, which
                # holds every request's true final token.  Broadcast in place.
                _dbg(f"compute_logits rank={pcp_group.rank_in_group} P={P} "
                     f"shape={tuple(hidden_states.shape)} "
                     f"norm_before={hidden_states.float().norm().item():.3f}")
                pcp_group.broadcast(hidden_states, src=P - 1)
                _dbg(f"compute_logits rank={pcp_group.rank_in_group} "
                     f"norm_after={hidden_states.float().norm().item():.3f}")
        logits = _orig_compute_logits(hidden_states, *args, **kwargs)
        if _PCP_DEBUG and logits is not None:
            try:
                _top = logits.float().topk(5, dim=-1)
                _dbg(f"compute_logits OUT shape={tuple(logits.shape)} "
                     f"argmax={logits.float().argmax(dim=-1).tolist()} "
                     f"top5_ids={_top.indices.tolist()} "
                     f"lnan={int(torch.isnan(logits).sum())}")
            except Exception:
                pass
        return logits

    model.compute_logits = _wrapped_compute_logits
    runner._pcp_logits_gather_installed = True
    logger.info("vllm_fl: PCP compute_logits gather installed")


def apply_pcp_model_runner_patch() -> None:
    # PCP causes vLLM to use V1 GPUModelRunner (not V2).
    # Hybrid models (Qwen3.6) also require V1 due to has_inner_state.
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        logger.warning("vllm_fl: GPUModelRunner not found, skipping model runner patch")
        return

    if getattr(GPUModelRunner, "_pcp_patched", False):
        return

    GPUModelRunner._original_prepare_inputs = GPUModelRunner._prepare_inputs
    GPUModelRunner._prepare_inputs = _patched_v1_prepare_inputs
    GPUModelRunner._pcp_patched = True
    logger.info("vllm_fl: PCP patch applied to GPUModelRunner")


# ---------------------------------------------------------------------------
# Hybrid + PCP hard-guard patches  (Step A: make tp*pcp actually boot)
#
# vLLM ships three guards that reject context parallelism for hybrid /
# mamba KV-cache layouts.  Qwen3.6 is exactly such a model (full-attention
# groups + linear-attention/GDN mamba groups → multiple block sizes).  The
# guards fire during EngineCore init, before the model runner ever runs, so
# our attention/model-runner patches can't intercept them.
#
# Guard #1 (always on the boot path):
#   kv_cache_utils.resolve_kv_cache_block_sizes  → raises ValueError for
#   multiple block sizes when dcp/pcp > 1.
# Guard #2 (only when enable_prefix_caching=True):
#   HybridKVCacheCoordinator.__init__            → assert pcp_world_size == 1
# Guard #3 (only when enable_prefix_caching=True):
#   MambaManager.find_longest_cache_hit          → assert pcp_world_size == 1
#
# Strategy: reuse the original logic but present pcp/dcp == 1 to the block-size
# accounting.  Sequence splitting is done later at the model-runner level, so
# the KV-cache bookkeeping stays as if there were no context parallelism.
# This is Step A (boot + generate); GDN/decode correctness is handled in Step B.
# ---------------------------------------------------------------------------

def apply_pcp_guard_patches() -> None:
    # ---- Guard #1: resolve_kv_cache_block_sizes -----------------------------
    try:
        from vllm.v1.core import kv_cache_utils
    except ImportError:
        logger.warning("vllm_fl: kv_cache_utils not found, skipping guard #1")
    else:
        if not getattr(kv_cache_utils, "_pcp_guard_patched", False):
            _orig_resolve = kv_cache_utils.resolve_kv_cache_block_sizes

            def _patched_resolve(kv_cache_config, vllm_config, *args, **kwargs):
                pc = vllm_config.parallel_config
                groups = kv_cache_config.kv_cache_groups
                needs_bypass = len(groups) > 1 and (
                    pc.prefill_context_parallel_size != 1
                    or pc.decode_context_parallel_size != 1
                )
                if not needs_bypass:
                    return _orig_resolve(kv_cache_config, vllm_config, *args, **kwargs)
                saved_pcp = pc.prefill_context_parallel_size
                saved_dcp = pc.decode_context_parallel_size
                pc.prefill_context_parallel_size = 1
                pc.decode_context_parallel_size = 1
                try:
                    result = _orig_resolve(
                        kv_cache_config, vllm_config, *args, **kwargs
                    )
                finally:
                    pc.prefill_context_parallel_size = saved_pcp
                    pc.decode_context_parallel_size = saved_dcp
                logger.info(
                    "vllm_fl: bypassed hybrid+CP block-size guard "
                    "(pcp=%d, dcp=%d, %d kv-cache groups)",
                    saved_pcp, saved_dcp, len(groups),
                )
                return result

            kv_cache_utils.resolve_kv_cache_block_sizes = _patched_resolve
            kv_cache_utils._pcp_guard_patched = True
            # core.py imported the name directly — patch that binding too.
            try:
                from vllm.v1.engine import core as _core_mod
                if hasattr(_core_mod, "resolve_kv_cache_block_sizes"):
                    _core_mod.resolve_kv_cache_block_sizes = _patched_resolve
            except ImportError:
                pass
            logger.info("vllm_fl: guard #1 (block-size) patched")

    # ---- Guard #2: HybridKVCacheCoordinator.__init__ ------------------------
    try:
        from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
    except ImportError:
        logger.warning("vllm_fl: HybridKVCacheCoordinator not found, skip guard #2")
    else:
        if not getattr(HybridKVCacheCoordinator, "_pcp_guard_patched", False):
            _orig_hybrid_init = HybridKVCacheCoordinator.__init__

            def _patched_hybrid_init(self, *args, **kwargs):
                # Force the CP world sizes seen by the coordinator to 1 so the
                # (dcp/pcp == 1) asserts pass; sequence splitting is external.
                if "pcp_world_size" in kwargs:
                    kwargs["pcp_world_size"] = 1
                if "dcp_world_size" in kwargs:
                    kwargs["dcp_world_size"] = 1
                _orig_hybrid_init(self, *args, **kwargs)

            HybridKVCacheCoordinator.__init__ = _patched_hybrid_init
            HybridKVCacheCoordinator._pcp_guard_patched = True
            logger.info("vllm_fl: guard #2 (hybrid coordinator) patched")

    # ---- Guard #3: MambaManager.find_longest_cache_hit ----------------------
    try:
        from vllm.v1.core.single_type_kv_cache_manager import MambaManager
    except ImportError:
        logger.warning("vllm_fl: MambaManager not found, skipping guard #3")
    else:
        if not getattr(MambaManager, "_pcp_guard_patched", False):
            _orig_find = MambaManager.find_longest_cache_hit.__func__

            def _patched_find(cls, *args, **kwargs):
                kwargs["dcp_world_size"] = 1
                kwargs["pcp_world_size"] = 1
                return _orig_find(cls, *args, **kwargs)

            MambaManager.find_longest_cache_hit = classmethod(_patched_find)
            MambaManager._pcp_guard_patched = True
            logger.info("vllm_fl: guard #3 (mamba cache-hit) patched")


# ---------------------------------------------------------------------------
# GatedDeltaNet (GDN / linear-attention) patch  (Step B, Huawei-style)
#
# Full-attention layers are sequence-split (real PCP savings).  GDN layers,
# however, carry a recurrent state that depends on the WHOLE sequence, so a
# sequence split breaks them (produces NaN).  Following the vllm-ascend /
# RFC #37995 design, GDN must see the full sequence.  We implement that as a
# "full-sequence island": at each GDN layer, all-gather the split hidden
# states back into per-request contiguous full sequences, run the ORIGINAL
# GDN forward with a freshly-built full-sequence GDNAttentionMetadata, then
# slice this rank's local tokens back out.  The scan is run redundantly on
# every rank (GDN is cheap/linear); as a bonus every rank's ssm/conv cache
# ends up holding the correct full-sequence state.
# ---------------------------------------------------------------------------

def _build_gdn_reshuffle(all_qsl_cpu, this_rank: int):
    """
    From every PCP rank's LOCAL cu_seqlens (all_qsl_cpu, shape [P, num_reqs+1]),
    build the index tensors that map the rank-major all-gathered (padded) buffer
    into a per-request contiguous FULL layout, and back to this rank's slice.

    Contiguous split (rank r owns the r-th chunk of each request) means the full
    sequence order for request i is rank0's chunk, then rank1's, ...

    Returns dict with numpy arrays / ints:
      perm       [full_total]     : full_pos -> gathered_pos (into padded gather)
      inv_local  [local_total_r]  : this rank's local_pos -> full_pos
      full_qsl   [num_reqs+1]     : full cu_seqlens
      full_total, max_total, num_reqs
    """
    import numpy as np
    a = all_qsl_cpu.numpy().astype(np.int64)
    P, R1 = a.shape
    num_reqs = R1 - 1
    local_totals = a[:, -1]                      # [P]
    max_total = int(local_totals.max())
    local_len = a[:, 1:] - a[:, :-1]             # [P, num_reqs]
    full_len = local_len.sum(axis=0)             # [num_reqs]
    full_qsl = np.zeros(num_reqs + 1, dtype=np.int64)
    full_qsl[1:] = np.cumsum(full_len)
    full_total = int(full_qsl[-1])

    perm = np.empty(full_total, dtype=np.int64)
    for i in range(num_reqs):
        off = int(full_qsl[i])
        for r in range(P):
            L = int(local_len[r, i])
            if L == 0:
                continue
            g0 = r * max_total + int(a[r, i])    # base of rank r in padded gather
            perm[off:off + L] = np.arange(g0, g0 + L)
            off += L

    lt = int(local_totals[this_rank])
    inv_local = np.empty(lt, dtype=np.int64)
    for i in range(num_reqs):
        L = int(local_len[this_rank, i])
        if L == 0:
            continue
        lo = int(a[this_rank, i])                       # local buffer offset
        prefix = int(local_len[:this_rank, i].sum())    # tokens before this rank
        fpos = int(full_qsl[i]) + prefix
        inv_local[lo:lo + L] = np.arange(fpos, fpos + L)

    return {
        "perm": perm,
        "inv_local": inv_local,
        "full_qsl": full_qsl,
        "full_total": full_total,
        "max_total": max_total,
        "num_reqs": num_reqs,
    }


def _get_gdn_pcp_plan(pcp_group, P, rank, gmd, device):
    """
    Build the reshuffle plan + a full-sequence GDNAttentionMetadata template.

    Recomputed per GDN layer (the cu_seqlens all-gather is tiny).  NOT cached on
    the forward context: that object is reused across forward passes, so caching
    there would reuse a stale plan for a differently-shaped batch.
    """
    import torch as _torch
    from vllm.model_executor.layers.fla.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    # All-gather each rank's local cu_seqlens (small): [num_reqs+1] -> [P, num_reqs+1]
    local_qsl = gmd.non_spec_query_start_loc.to(_torch.int64)
    gathered_qsl = pcp_group.all_gather(local_qsl.contiguous(), dim=0)
    all_qsl_cpu = gathered_qsl.view(P, -1).cpu()

    rs = _build_gdn_reshuffle(all_qsl_cpu, rank)

    perm = _torch.as_tensor(rs["perm"], device=device, dtype=_torch.long)
    inv_local = _torch.as_tensor(rs["inv_local"], device=device, dtype=_torch.long)
    full_qsl_cpu = _torch.as_tensor(rs["full_qsl"], dtype=_torch.int32)
    full_qsl_gpu = full_qsl_cpu.to(device)
    num_reqs = rs["num_reqs"]
    full_total = rs["full_total"]

    chunk_indices = prepare_chunk_indices(full_qsl_cpu, FLA_CHUNK_SIZE).to(device)
    chunk_offsets = prepare_chunk_offsets(full_qsl_cpu, FLA_CHUNK_SIZE).to(device)
    nums_dict, batch_ptr, tok_off = compute_causal_conv1d_metadata(
        full_qsl_cpu, device=device
    )
    has_initial_state = _torch.zeros(num_reqs, dtype=_torch.bool, device=device)

    def _make_full_md(state_indices):
        return GDNAttentionMetadata(
            num_prefills=num_reqs,
            num_prefill_tokens=full_total,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=full_total,
            has_initial_state=has_initial_state,
            non_spec_query_start_loc=full_qsl_gpu,
            non_spec_state_indices_tensor=state_indices,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=tok_off,
        )

    plan = {
        "perm": perm,
        "inv_local": inv_local,
        "full_total": full_total,
        "max_total": rs["max_total"],
        "make_full_md": _make_full_md,
    }
    return plan


def _patched_gdn_forward(self, hidden_states, output):
    from vllm.distributed.parallel_state import get_pcp_group
    from vllm.forward_context import get_forward_context

    try:
        pcp_group = get_pcp_group()
        P = pcp_group.world_size
        rank = pcp_group.rank_in_group
    except AssertionError:
        P = 1

    if P <= 1:
        return self._pcp_orig_gdn_forward(hidden_states, output)

    ctx = get_forward_context()
    md = getattr(ctx, "attn_metadata", None)
    gmd = md.get(self.prefix) if isinstance(md, dict) else None

    # Only intercept pure-prefill GDN.  Fall through for profile runs, pure
    # decode, or mixed batches (handled by the original path for now).
    if (
        gmd is None
        or getattr(gmd, "num_prefills", 0) == 0
        or getattr(gmd, "num_decodes", 0) > 0
        or getattr(gmd, "num_spec_decodes", 0) > 0
    ):
        return self._pcp_orig_gdn_forward(hidden_states, output)

    import torch as _torch
    device = hidden_states.device
    H = hidden_states.shape[-1]
    # Actual local prefill tokens (the hidden_states buffer may be padded larger).
    local_actual = int(gmd.non_spec_query_start_loc[-1].item())

    plan = _get_gdn_pcp_plan(pcp_group, P, rank, gmd, device)

    # All-gather hidden states (padded to max_total so sizes are equal).
    max_total = plan["max_total"]
    padded = hidden_states.new_zeros((max_total, H))
    padded[:local_actual] = hidden_states[:local_actual]
    gathered = pcp_group.all_gather(padded, dim=0)          # [P*max_total, H]
    full_hidden = gathered.index_select(0, plan["perm"]).contiguous()

    full_md = plan["make_full_md"](gmd.non_spec_state_indices_tensor)
    full_output = full_hidden.new_zeros((plan["full_total"], H))

    saved = md[self.prefix]
    md[self.prefix] = full_md
    try:
        self._pcp_orig_gdn_forward(full_hidden, full_output)
    finally:
        md[self.prefix] = saved

    # Slice this rank's local tokens back out of the full output.
    output[:local_actual] = full_output.index_select(0, plan["inv_local"])
    _dbg(f"gdn rank={rank} local_actual={local_actual} full_total={plan['full_total']} "
         f"in_norm={hidden_states[:local_actual].float().norm().item():.3f} "
         f"full_out_norm={full_output.float().norm().item():.3f} "
         f"out_norm={output[:local_actual].float().norm().item():.3f}")


def apply_pcp_gdn_patch() -> None:
    try:
        from vllm.model_executor.layers.mamba.gdn_linear_attn import (
            GatedDeltaNetAttention,
        )
    except ImportError:
        logger.warning("vllm_fl: GatedDeltaNetAttention not found, skip GDN patch")
        return
    if getattr(GatedDeltaNetAttention, "_pcp_patched", False):
        return
    GatedDeltaNetAttention._pcp_orig_gdn_forward = GatedDeltaNetAttention.forward
    GatedDeltaNetAttention.forward = _patched_gdn_forward
    GatedDeltaNetAttention._pcp_patched = True
    logger.info("vllm_fl: PCP full-sequence island patch applied to GatedDeltaNetAttention")


# ---------------------------------------------------------------------------
# Combined registration
# ---------------------------------------------------------------------------

def apply_pcp_patch() -> None:
    apply_pcp_guard_patches()
    apply_pcp_attn_patch()
    apply_pcp_model_runner_patch()
    apply_pcp_gdn_patch()


def apply_platform_patches() -> None:
    """Public entry point (called from ``vllm_fl.register()``).

    Installs all Qwen3.6 hybrid + PCP patches.  Idempotent and safe to call
    unconditionally: each patched function self-gates on
    ``prefill_context_parallel_size > 1`` and falls through to the original
    vLLM implementation when PCP is disabled.
    """
    apply_pcp_patch()
