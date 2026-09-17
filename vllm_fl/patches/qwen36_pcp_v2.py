# SPDX-License-Identifier: Apache-2.0
# ==========================================================================
# Qwen3.6 hybrid (GDN) + PCP -- v5.2 (SELF-CONTAINED, no sibling imports).
# Functionally identical to v5.1. Assembled from v3 (guards/attn/model-runner
# + Path B) + v5's parallel two-pass GDN op + v5.1's install hooks + conv halo,
# all inlined verbatim below. v5's _dbg is renamed _dbg_v5s2 to coexist with
# v3's _dbg. Enable with VLLM_PCP_V5_2=1. Built by /local/build_v5_2.py.
# ==========================================================================

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
DONE step-3: striped/zigzag 2P-block assignment for load balance (VLLM_PCP_ZIGZAG=1).
TODO: multi-request batch with unequal lengths (need per-request K reorder;
      the single-broadcast last-token src is only correct for equal-length batches).
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

# VLLM_PCP_ZIGZAG=1 switches the token->rank assignment from CONTIGUOUS slicing
# to 2P-block ZIGZAG interleaving, balancing the causal attention load across
# PCP ranks (each rank owns one early + one late block).  Off -> v2 behaviour.
_PCP_ZIGZAG = os.environ.get("VLLM_PCP_ZIGZAG", "") == "1"
# Path B: shrink the model-forward row count to this rank's local token count
# so MoE/MLP/norm stop computing the phantom padding tail.  Gated because it
# rewrites the runner's forward-sizing; off => current full-buffer behaviour.
_LOCAL_FWD = os.environ.get("VLLM_PCP_LOCAL_FWD", "") == "1"
# Fix the cross-request crash: recompute discard_request_mask from FULL seq len
# (default ON; set VLLM_PCP_FIX_DISCARD=0 to A/B the old buggy behavior).
_PCP_FIX_DISCARD = os.environ.get("VLLM_PCP_FIX_DISCARD", "1") != "0"
# EXPERIMENT (opt-in): make the routed MoE PCP-local (pcp_size=1) so experts are
# NOT sharded over the PCP group -> each rank holds all experts -> MoE runs on its
# local L/P tokens with NO cross-rank all_gather. Costs ~P x expert memory; only
# sensible for tp1+pcp4 (no EP/EPLB). VLLM_PCP_MOE_LOCAL=1 to try; revert by unset.
_MOE_LOCAL = os.environ.get("VLLM_PCP_MOE_LOCAL", "") == "1"
# EXPERIMENT (opt-in): force flash attention to num_splits=1 (non-splitkv kernel)
# by setting attn_metadata.max_num_splits=1. Tests whether tp4's slow splitkv is
# a kernel-choice artifact. VLLM_PCP_NOSPLIT=1.
_NOSPLIT = os.environ.get("VLLM_PCP_NOSPLIT", "") == "1"

# FP32 localization knobs for the GDN two-pass (localization experiments only).
#   VLLM_PCP_GDN_FP32=1  -> run the ENTIRE two-pass in fp32 (upcast q/k/v/g/beta
#                           at entry, downcast o at exit). State/Phi/recombine are
#                           already fp32; this additionally removes the bf16
#                           rounding of the WY (w,u), solve_tril, h-scan and
#                           chunk_fwd_o token-level intermediates.
_GDN_FP32 = os.environ.get("VLLM_PCP_GDN_FP32", "") == "1"

# Channel from _patched_v1_prepare_inputs (which decides the zigzag layout) to
# _forward_with_pcp (which must build the matching block-causal K/V gather).
# Single worker process, one forward at a time, so a module global is safe.
# Reset every prepare_inputs call so stale prefill state never leaks into a
# later decode step.  full_lens is per-prefill-request in input_batch order.
_ZZ_STATE: dict = {"active": False}

# Channel from the attention patch to the MoE phantom-truncation patch
# (qwen36_pcp_v5_1._maybe_install_moe_trunc).  Under PCP the runner allocates a
# forward buffer padded to the full sequence length but only fills this rank's
# local slice [0:n_real]; the tail [n_real:n_buf] is phantom (garbage/NaN, zeroed
# by attention).  Token-wise layers (MoE/MLP) redundantly process the phantom
# tail.  The attention patch records the current prefill's (n_real, n_buf) here
# so the MoE patch can skip the tail.  Single worker, one forward at a time.
_PCP_REAL: dict = {"n_real": None, "n_buf": None}


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


# ---------------------------------------------------------------------------
# Zigzag (2P-block) helpers
# ---------------------------------------------------------------------------

def _zigzag_block_sizes(full_len: int, P: int):
    """Sizes of the 2P equal blocks of a prefill of length full_len.

    block_size = ceil(full_len / 2P); the tail blocks absorb the remainder and
    may be short or zero.  Returns a python list of 2P ints summing to full_len.
    """
    import numpy as np
    nb = 2 * P
    bs = math.ceil(full_len / nb)
    sizes = []
    start = 0
    for _ in range(nb):
        end = min(start + bs, full_len)
        sizes.append(end - start)
        start = end
    return sizes  # len == 2P, sum == full_len


def _zigzag_local_positions(full_len: int, P: int, r: int):
    """Absolute token positions rank r owns, in local-buffer order.

    Rank r owns block r (lo) and block 2P-1-r (hi).  Layout: [lo tokens asc,
    hi tokens asc].  Returns np.int64 array of length s[r] + s[2P-1-r].
    """
    import numpy as np
    sizes = _zigzag_block_sizes(full_len, P)
    nb = 2 * P
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)  # [nb+1]
    lo, hi = r, nb - 1 - r
    pos = []
    for b in (lo, hi):
        if sizes[b] > 0:
            pos.append(np.arange(starts[b], starts[b] + sizes[b], dtype=np.int64))
    if pos:
        return np.concatenate(pos)
    return np.empty(0, dtype=np.int64)


def _zigzag_last_token_rank(full_len: int, P: int) -> int:
    """Group-local rank owning the LAST non-empty block (holds the final prompt
    token, whose hidden state feeds the LM head)."""
    sizes = _zigzag_block_sizes(full_len, P)
    nb = 2 * P
    for b in range(nb - 1, -1, -1):
        if sizes[b] > 0:
            return min(b, nb - 1 - b)
    return P - 1


def _build_zigzag_kv_gather(full_lens, P: int, rank: int, max_total: int):
    """Zigzag analogue of _build_attn_kv_gather.

    Every rank contributes its padded local K/V (max_total rows) to the
    all-gather buffer; rank r' occupies rows [r'*max_total, ...).  Within that
    block, request i is laid out [block r' tokens, block 2P-1-r' tokens], and
    requests are concatenated in batch order.

    For THIS rank we emit 2B query sub-sequences — for each request its lo block
    (=block rank) then its hi block (=block 2P-1-rank).  Sub-sequence for query
    block b attends to the causal prefix = blocks 0..b (blocks 0..b-1 in full,
    block b causally via flash end-alignment).

    Returns (kidx int64[total_k], cu_q int32[2B+1], cu_k int32[2B+1],
             max_q int, max_k int).  The q sub-sequence order matches the local
    buffer layout, so flash output lands in `output` directly.
    """
    import numpy as np
    nb = 2 * P
    B = len(full_lens)
    # Per-request block sizes for every request (global, from full_len).
    sizes = [np.asarray(_zigzag_block_sizes(int(fl), P), dtype=np.int64) for fl in full_lens]

    # Local per-request length on each rank r' = s[r'] + s[nb-1-r'].
    def local_len(rp, i):
        return int(sizes[i][rp] + sizes[i][nb - 1 - rp])

    # cumoff[rp][i] = start row of request i within rank rp's local buffer.
    cumoff = np.zeros((P, B), dtype=np.int64)
    for rp in range(P):
        acc = 0
        for i in range(B):
            cumoff[rp, i] = acc
            acc += local_len(rp, i)

    def block_gather_start(c, i):
        """Row of block c (request i) within the all-gather buffer."""
        owner = min(c, nb - 1 - c)
        base = owner * max_total + int(cumoff[owner, i])
        # lo block of owner sits first; hi block sits after it.
        if c == owner:
            return base                       # lo block
        return base + int(sizes[i][owner])    # hi block (offset past lo)

    cu_q = np.zeros(2 * B + 1, dtype=np.int32)
    cu_k = np.zeros(2 * B + 1, dtype=np.int32)
    kidx_parts = []
    max_q = 0
    max_k = 0
    sub = 0
    for i in range(B):
        for b in (rank, nb - 1 - rank):       # lo then hi query block
            qs = int(sizes[i][b])
            cu_q[sub + 1] = cu_q[sub] + qs
            k_prefix = 0
            for c in range(b + 1):            # causal prefix blocks 0..b
                sc = int(sizes[i][c])
                if sc == 0:
                    continue
                start = block_gather_start(c, i)
                kidx_parts.append(np.arange(start, start + sc, dtype=np.int64))
                k_prefix += sc
            cu_k[sub + 1] = cu_k[sub] + k_prefix
            max_q = max(max_q, qs)
            max_k = max(max_k, k_prefix)
            sub += 1

    kidx = (np.concatenate(kidx_parts) if kidx_parts
            else np.empty(0, dtype=np.int64))
    return kidx, cu_q, cu_k, max_q, max_k


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

    # 3. Reshuffle into this rank's per-request causal K/V.
    zz = _ZZ_STATE if _PCP_ZIGZAG else {"active": False}
    if zz.get("active"):
        # Zigzag: 2B query sub-sequences (lo=block rank, hi=block 2P-1-rank per
        # request), each attending its block-causal prefix 0..b.
        full_lens = zz["full_lens"]
        B = cu_q.numel() - 1
        assert len(full_lens) == B, (
            f"zigzag full_lens {len(full_lens)} != attn batch {B}")
        kidx_np, cu_q_np, cu_k_np, max_q_len, max_k_len = _build_zigzag_kv_gather(
            full_lens, P, rank, max_total)
        kidx = torch.as_tensor(kidx_np, device=query.device, dtype=torch.long)
        key_causal = key_gathered.index_select(0, kidx)
        val_causal = val_gathered.index_select(0, kidx)
        cu_seqlens_q = torch.as_tensor(cu_q_np, device=query.device, dtype=torch.int32)
        cu_seqlens_k = torch.as_tensor(cu_k_np, device=query.device, dtype=torch.int32)
        _dbg(f"attn r{rank} ZIGZAG: n_local={n_local} max_total={max_total} "
             f"full_lens={full_lens} 2B={cu_q_np.shape[0]-1} "
             f"cu_q={cu_q_np.tolist()} cu_k={cu_k_np.tolist()} "
             f"max_q={max_q_len} max_k={max_k_len} "
             f"kc_nan={int(torch.isnan(key_causal).sum())}")
    else:
        # Contiguous: causal prefix = ranks 0..rank (v2 behaviour).
        kidx_np, cu_k_np, max_k_len = _build_attn_kv_gather(all_qsl_cpu, rank, max_total)
        kidx = torch.as_tensor(kidx_np, device=query.device, dtype=torch.long)
        key_causal = key_gathered.index_select(0, kidx)
        val_causal = val_gathered.index_select(0, kidx)
        cu_seqlens_q = cu_q
        max_q_len = attn_metadata.max_query_len
        cu_seqlens_k = torch.as_tensor(cu_k_np, device=query.device, dtype=torch.int32)
        _dbg(f"attn r{rank}: n_local={n_local} max_total={max_total} "
             f"k_causal={tuple(key_causal.shape)} max_k_len={max_k_len} "
             f"k_nan={int(torch.isnan(key).sum())} kc_nan={int(torch.isnan(key_causal).sum())}")

    sliding_window_size = (
        list(self.sliding_window) if self.sliding_window is not None else None
    )

    # 4. Flash attention: Q_local vs K_causal with causal=True.
    #    flash_attn end-alignment: Q[i] attends to K[0 .. K_len - Q_len + i],
    #    which is exactly the causal prefix for each query (segment or block).
    flash_attn_varlen_func(
        q=query,
        k=key_causal,
        v=val_causal,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_q_len,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_k=max_k_len,
        softmax_scale=self.scale,
        causal=True,
        alibi_slopes=self.alibi_slopes,
        window_size=sliding_window_size,
        softcap=self.logits_soft_cap,
        fa_version=self.vllm_flash_attn_version,
    )

    if (os.environ.get("VLLM_PCP_ATTN_SELFCHECK") == "1"
            and not zz.get("active") and cu_q.numel() == 2 and n_local >= 1024):
        # TP4-independent attention self-consistency: gather full Q/K/V (contiguous
        # = global order), run ONE monolithic causal flash over the whole sequence
        # (== what tp1 computes), compare this rank's Q-segment output vs PCP output.
        try:
            def _agq(x):  # [n_local, H, D] -> [P*n_local, H, D] rank order
                g = pcp_group.all_gather(x.contiguous(), dim=0)  # [P*n_local, H, D]
                return g.contiguous()
            q_full = _agq(query)
            k_full = _agq(key[:n_local])
            v_full = _agq(value[:n_local])
            Nf = P * n_local
            cu_full = torch.tensor([0, Nf], device=query.device, dtype=torch.int32)
            out_full = torch.empty_like(q_full)
            flash_attn_varlen_func(
                q=q_full, k=k_full, v=v_full, out=out_full,
                cu_seqlens_q=cu_full, max_seqlen_q=Nf,
                cu_seqlens_k=cu_full, max_seqlen_k=Nf,
                softmax_scale=self.scale, causal=True,
                alibi_slopes=self.alibi_slopes, window_size=sliding_window_size,
                softcap=self.logits_soft_cap, fa_version=self.vllm_flash_attn_version,
            )
            o_seg = out_full[rank * n_local:(rank + 1) * n_local]
            if o_seg.shape == output.shape:
                d = (output.float() - o_seg.float()).norm() / (o_seg.float().norm() + 1e-9)
                dmax = (output.float() - o_seg.float()).abs().max().item()
                _dbg_v5s2(f"ATTN-SELFCHECK rank={rank}/{P} n_local={n_local} "
                          f"d_attn(pcp_vs_mono)={d.item():.3e} max|d|={dmax:.3e}")
            else:
                _dbg_v5s2(f"ATTN-SELFCHECK rank={rank} SHAPE out={tuple(output.shape)} "
                          f"o_seg={tuple(o_seg.shape)}")
        except Exception as e:  # noqa: BLE001
            _dbg_v5s2(f"ATTN-SELFCHECK rank={rank} FAILED {type(e).__name__}: {e}")


def _patched_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, **kwargs):
    """Wraps the original forward to intercept the PCP prefill path."""
    # Lazy model-runner patch: by the time attention runs (warmup/profiling pass,
    # before any real request), vllm.v1.worker.gpu_model_runner is fully imported
    # and sits in sys.modules -- no circular import, canonical class identity.
    # Patching _prepare_inputs here means the NEXT _prepare_inputs (the real
    # request's) uses the zigzag/contiguous split wrapper. This is the reliable
    # install point; the plugin-apply-time import always hits a circular import.
    if not getattr(_patched_forward, "_runner_patch_tried", False):
        import sys as _sys
        _patched_forward._runner_patch_tried = True
        # FL runs its OWN runner ModelRunnerFL (vllm_fl.worker.model_runner), not
        # vLLM's GPUModelRunner. By attention time that module is fully imported.
        _mod = _sys.modules.get("vllm_fl.worker.model_runner")
        if _mod is not None and hasattr(_mod, "ModelRunnerFL"):
            _install_pcp_prepare_inputs_on(_mod.ModelRunnerFL)
    if _NOSPLIT and attn_metadata is not None:
        # Force non-splitkv flash kernel (num_splits=1) on the ORIGINAL path.
        try:
            attn_metadata.max_num_splits = 1
        except Exception:
            pass
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
        # Record for the MoE phantom-truncation patch (constant across layers
        # within this forward).  Only prefill reaches here (max_query_len>1).
        _PCP_REAL["n_real"] = n_real
        _PCP_REAL["n_buf"] = n_buf
        # Diagnostic upper bound: VLLM_PCP_ATTN=local -> each rank attends ONLY
        # its own local segment (no cross-rank K/V gather, no causal prefix).
        # NUMERICALLY WRONG, but a timing floor for the attention layers: if
        # this doesn't beat the real gather+prefix path, the attention
        # gather/imbalance isn't the bottleneck and zigzag (B) isn't worth it.
        if os.environ.get("VLLM_PCP_ATTN") == "local":
            try:
                from vllm.v1.attention.backends.flash_attn import flash_attn_varlen_func
            except ImportError:
                from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
            flash_attn_varlen_func(
                q=query[:n_real], k=key[:n_real], v=value[:n_real], out=output[:n_real],
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_k=attn_metadata.max_query_len,
                softmax_scale=self.scale, causal=True,
                fa_version=self.vllm_flash_attn_version,
            )
            if n_buf > n_real:
                output[n_real:n_buf].zero_()
            return output
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
    if not getattr(_patched_v1_prepare_inputs, "_probed", False):
        import sys
        _patched_v1_prepare_inputs._probed = True
        print("vllm_fl: [PCP-PATCH] _patched_v1_prepare_inputs IS RUNNING "
              f"(zigzag={_PCP_ZIGZAG})", file=sys.stderr, flush=True)
    try:
        pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
    except AttributeError:
        pcp_size = 1
    # Config-agnostic logits dump (fires for pcp==1 baseline too).  Install
    # before the gather so it sits innermost and observes corrected logits.
    _install_pcp_logits_dump(self)
    # Reset the Path-B local row count every step; set below only when this step
    # actually reduces the prefill (so decode / non-PCP fall through to full).
    _PCP_REAL["local_total"] = None
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

    global _ZZ_STATE
    _ZZ_STATE = {"active": False}  # reset every step; only prefill steps re-arm

    # Identify prefill requests: num_computed_tokens < num_prompt_tokens.
    # prefills: (slot_idx, full_len, num_computed, num_prompt)
    prefills = []
    for slot_idx, req_id in enumerate(req_ids):
        req_state = self.requests.get(req_id)
        if req_state is None:
            continue
        full_len = int(num_scheduled_tokens_np[slot_idx])
        if req_state.num_computed_tokens < req_state.num_prompt_tokens:
            prefills.append((slot_idx, full_len,
                             req_state.num_computed_tokens,
                             req_state.num_prompt_tokens))

    if not prefills:
        return self._original_prepare_inputs(scheduler_output, num_scheduled_tokens_np)

    import numpy as np

    # Zigzag requires each prefill to be scheduled whole in one step (fresh,
    # num_computed==0, full prompt), because the 2P-block layout is defined
    # relative to prompt position 0.  Chunked/continued prefill -> fall back to
    # the contiguous split (keeps this experimental path bounded to the tested
    # single-shot regime).
    use_zigzag = _PCP_ZIGZAG and all(
        (nc == 0 and fl == npt) for (_, fl, nc, npt) in prefills)
    if _PCP_ZIGZAG and not use_zigzag:
        _dbg("zigzag: chunked/continued prefill present -> contiguous fallback")

    if use_zigzag:
        # local_len per prefill = s[r] + s[2P-1-r] (its two owned blocks).
        modified_np = num_scheduled_tokens_np.copy()
        full_lens = []
        for slot_idx, full_len, _, _ in prefills:
            sizes = _zigzag_block_sizes(full_len, P)
            modified_np[slot_idx] = sizes[r] + sizes[2 * P - 1 - r]
            full_lens.append(full_len)
        local_total = int(modified_np.sum())
        _PCP_REAL["local_total"] = local_total
        modified_sched = _V1PcpSchedWrapper(scheduler_output, local_total)
        # NO num_computed shift: positions/input_ids are overwritten post-hoc to
        # the non-contiguous zigzag layout.
        result = self._original_prepare_inputs(modified_sched, modified_np)

        # Overwrite self.positions (RoPE) and self.input_ids.gpu (tokens) for
        # each prefill's local slice with its zigzag layout [lo block, hi block].
        local_starts = np.concatenate([[0], np.cumsum(modified_np)]).astype(np.int64)
        tok_buf = self.input_batch.token_ids_cpu_tensor  # [max_reqs, max_model_len]
        pos_dev, pos_dt = self.positions.device, self.positions.dtype
        ids_gpu = self.input_ids.gpu
        for slot_idx, full_len, _, _ in prefills:
            zpos = _zigzag_local_positions(full_len, P, r)  # np.int64, len local_len
            qs = int(local_starts[slot_idx])
            L = int(zpos.shape[0])
            zpos_cpu = torch.as_tensor(zpos, dtype=torch.long)
            self.positions[qs:qs + L] = zpos_cpu.to(pos_dev, dtype=pos_dt)
            toks = tok_buf[slot_idx].index_select(0, zpos_cpu)
            ids_gpu[qs:qs + L] = toks.to(ids_gpu.device, dtype=ids_gpu.dtype)

        _ZZ_STATE = {"active": True, "P": P, "full_lens": full_lens}
        _dbg(f"prepare_inputs r{r} ZIGZAG: full_lens={full_lens} "
             f"local_total={local_total} modified={modified_np.tolist()}")
        return result

    # ---- Contiguous split (v2 behaviour) ----
    pcp_info: dict[int, tuple[int, int]] = {}
    for slot_idx, full_len, _, _ in prefills:
        local_len, pcp_offset = _compute_pcp_local_len(full_len, P, r)
        pcp_info[slot_idx] = (local_len, pcp_offset)
    modified_np = num_scheduled_tokens_np.copy()
    for slot_idx, (local_len, _) in pcp_info.items():
        modified_np[slot_idx] = local_len

    local_total = int(modified_np.sum())
    _PCP_REAL["local_total"] = local_total
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

    # PCP FIX (cross-request crash): _original_prepare_inputs set discard_request_
    # mask from the LOCAL (pcp-shifted) optimistic_seq_lens, so ranks 0..P-2 see
    # optimistic < num_tokens and WRONGLY discard a COMPLETING prefill's sample.
    # That desyncs their per-rank request state (num_tokens / token_ids_cpu /
    # prev_req_id_to_index) from rank P-1 and the scheduler -> the next decode step
    # reads an unwritten (-1) token at a REUSED batch slot -> device-side assert on
    # the embedding. Recompute discard from the FULL sequence (rank-independent) so
    # every rank agrees "prefill done -> sample".
    if _PCP_FIX_DISCARD:
        try:
            _nreq = self.input_batch.num_reqs
            for _sidx, _rid in enumerate(list(self.input_batch.req_ids[:_nreq])):
                _rs = self.requests.get(_rid)
                if _rs is None:
                    continue
                _full_opt = _rs.num_computed_tokens + int(num_scheduled_tokens_np[_sidx])
                self.discard_request_mask.np[_sidx] = _full_opt < _rs.num_tokens
            self.discard_request_mask.copy_to_gpu(_nreq)
        except Exception as _e:
            if _PCP_DEBUG:
                import sys as _sys
                print(f"vllm_fl:[PCP-FIX-DISCARD] failed {type(_e).__name__}: {_e}",
                      file=_sys.stderr, flush=True)

    if os.environ.get("VLLM_PCP_PROBE") == "1":
        import sys as _sys
        try:
            _lt = int(modified_np.sum())
            _ii = self.input_ids.gpu[:_lt]
            _pf = [(s, fl, nc, npt) for (s, fl, nc, npt) in prefills]
            print(f"vllm_fl:[PCP-PROBE] r{r} prefills(slot,full,comp,prompt)={_pf} "
                  f"local_total={_lt} input_ids[min={int(_ii.min())},max={int(_ii.max())}] "
                  f"pos[min={int(self.positions[:_lt].min())},max={int(self.positions[:_lt].max())}]",
                  file=_sys.stderr, flush=True)
        except Exception as _e:
            print(f"vllm_fl:[PCP-PROBE] err {type(_e).__name__}: {_e}", file=_sys.stderr, flush=True)

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
                # src = group-local rank holding every request's true final
                # token.  Contiguous split -> always the last rank (P-1).
                # Zigzag -> owner of the last non-empty block (block 2P-1 ->
                # rank 0 for block-aligned lengths).
                if _PCP_ZIGZAG and _ZZ_STATE.get("active"):
                    fls = _ZZ_STATE.get("full_lens") or []
                    ranks = {_zigzag_last_token_rank(int(fl), P) for fl in fls}
                    src = ranks.pop() if len(ranks) == 1 else _zigzag_last_token_rank(
                        int(max(fls)), P)
                    if len(fls) and len({_zigzag_last_token_rank(int(fl), P) for fl in fls}) > 1:
                        _dbg(f"compute_logits ZIGZAG: mixed last-token ranks for "
                             f"full_lens={fls}; single-broadcast src={src} may be "
                             f"wrong for some requests (bounded regime)")
                else:
                    src = P - 1
                # Broadcast sample_hidden_states in place.
                _dbg(f"compute_logits rank={pcp_group.rank_in_group} P={P} src={src} "
                     f"shape={tuple(hidden_states.shape)} "
                     f"norm_before={hidden_states.float().norm().item():.3f}")
                pcp_group.broadcast(hidden_states, src=src)
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


def _install_local_forward_on(GPUModelRunner) -> None:
    """Path B (VLLM_PCP_LOCAL_FWD=1): shrink the model-forward row count to this
    rank's local token count so the token-wise layers (MoE/MLP/norm) stop
    computing the phantom padding tail.

    execute_model reads num_tokens_unpadded = scheduler_output.total_num_scheduled
    _tokens (FULL) *before* _prepare_inputs runs, then sizes the whole forward
    from it: num_actual_tokens (attn metadata), slot_mapping, and — via
    batch_desc.num_tokens = num_tokens_padded — the input_ids/positions slice fed
    to the model and set_forward_context's num_tokens (which FusedMoE reads).

    _prepare_inputs already fills only [0:local_total] of every buffer (query_
    start_loc[-1] == local_total) and records local_total in _PCP_REAL.  Here we
    clamp the three forward-sizing consumers to local_total so the ENTIRE forward
    is local-consistent — the same value everywhere, which is exactly what keeps
    FusedMoE from producing NaN (an earlier attempt that only truncated the MoE
    block input while the forward context stayed FULL produced garbage).

    State advancement / bookkeeping read the original scheduler_output (FULL), so
    num_computed_tokens still advances by the full prefill length — untouched.
    """
    import sys
    import dataclasses
    if getattr(GPUModelRunner, "_pcp_local_fwd_patched", False):
        return

    def _lt(x):
        # local_total if this step reduces (int arg strictly larger); else x.
        v = _PCP_REAL.get("local_total")
        if _LOCAL_FWD and isinstance(v, int) and isinstance(x, int) and 0 < v < x:
            return v
        return x

    _orig_determine = GPUModelRunner._determine_batch_execution_and_padding
    _orig_slot = GPUModelRunner._get_slot_mappings
    _orig_attn = GPUModelRunner._build_attention_metadata

    def _determine(self, *args, **kwargs):
        res = _orig_determine(self, *args, **kwargs)
        nt = kwargs.get("num_tokens")
        v = _PCP_REAL.get("local_total")
        if (_LOCAL_FWD and isinstance(v, int) and isinstance(nt, int)
                and 0 < v < nt and isinstance(res, tuple) and len(res) == 5):
            cm, bd, su, ntad, st = res
            try:
                bd = dataclasses.replace(bd, num_tokens=v)
            except Exception as e:
                if _PCP_DEBUG:
                    print(f"vllm_fl:[PCP-LF] replace FAILED {type(e).__name__}: {e} "
                          f"bd_type={type(bd).__name__}", file=sys.stderr, flush=True)
                return res
            if _PCP_DEBUG:
                print(f"vllm_fl:[PCP-LF] forward rows {nt} -> {v}",
                      file=sys.stderr, flush=True)
            res = (cm, bd, su, ntad, st)
        return res

    def _slot(self, *args, **kwargs):
        if "num_tokens_padded" in kwargs:
            kwargs["num_tokens_padded"] = _lt(kwargs["num_tokens_padded"])
        if "num_tokens_unpadded" in kwargs:
            kwargs["num_tokens_unpadded"] = _lt(kwargs["num_tokens_unpadded"])
        return _orig_slot(self, *args, **kwargs)

    def _attn(self, *args, **kwargs):
        if "num_tokens" in kwargs:
            kwargs["num_tokens"] = _lt(kwargs["num_tokens"])
        return _orig_attn(self, *args, **kwargs)

    GPUModelRunner._determine_batch_execution_and_padding = _determine
    # NOTE: clamping _get_slot_mappings / _build_attention_metadata to local
    # corrupts attention (seq_lens/context for the cross-rank causal KV gather
    # must stay FULL) -> degenerate output.  Only the forward row count and the
    # forward-context num_tokens (both via batch_desc.num_tokens from _determine)
    # are reduced; slot_mapping keeps its full size (its phantom tail is -1 and
    # never touched by the local queries) and num_actual_tokens stays full.
    _CLAMP_SLOT_ATTN = os.environ.get("VLLM_PCP_LOCAL_FWD_SLOTATTN", "") == "1"
    if _CLAMP_SLOT_ATTN:
        GPUModelRunner._get_slot_mappings = _slot
        GPUModelRunner._build_attention_metadata = _attn
    GPUModelRunner._pcp_local_fwd_patched = True
    print("vllm_fl: [PCP-PATCH] local-forward (Path B) patched on "
          f"{GPUModelRunner.__module__}.{GPUModelRunner.__name__}",
          file=sys.stderr, flush=True)


def _install_pcp_prepare_inputs_on(GPUModelRunner) -> None:
    import sys
    if getattr(GPUModelRunner, "_pcp_prepare_patched", False):
        return
    GPUModelRunner._original_prepare_inputs = GPUModelRunner._prepare_inputs
    GPUModelRunner._prepare_inputs = _patched_v1_prepare_inputs
    GPUModelRunner._pcp_prepare_patched = True
    if _LOCAL_FWD:
        _install_local_forward_on(GPUModelRunner)
    print(f"vllm_fl: [PCP-PATCH] _prepare_inputs patched on "
          f"{GPUModelRunner.__module__}.{GPUModelRunner.__name__}",
          file=sys.stderr, flush=True)
    logger.info("vllm_fl: PCP patch applied to GPUModelRunner._prepare_inputs")


def _install_moe_local_pcp() -> None:
    """EXPERIMENT (VLLM_PCP_MOE_LOCAL=1): wrap FusedMoE.__init__ so that, under
    tp1 + pcp4 (no EP/EPLB) and when the caller did not pass pcp_size, we force
    pcp_size=1. That drops the PCP dim from flatten_tp -> experts are NOT sharded
    over the PCP group -> each rank holds ALL experts -> the MoE runs on its local
    L/P tokens with NO cross-rank all_gather (Path B's split reaches the MoE).
    Cost: ~pcp_world x expert memory per rank. Revert by unsetting the env."""
    import sys
    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_world_size, get_pcp_group)
        from vllm.config import get_current_vllm_config
    except Exception as e:  # noqa: BLE001
        print(f"vllm_fl: [PCP-MOE-LOCAL] import failed ({type(e).__name__}: {e}); "
              f"skip", file=sys.stderr, flush=True)
        return
    if getattr(FusedMoE, "_pcp_moe_local_patched", False):
        return
    _orig_init = FusedMoE.__init__

    def _init(self, *args, **kwargs):
        if kwargs.get("pcp_size") is None:
            local_pcp_moe = False
            try:
                pc = get_current_vllm_config().parallel_config
                local_pcp_moe = (
                    get_tensor_model_parallel_world_size() == 1
                    and get_pcp_group().world_size == 4
                    and not pc.enable_expert_parallel
                    and not getattr(pc, "enable_eplb", False)
                )
            except Exception:  # noqa: BLE001
                local_pcp_moe = False
            if local_pcp_moe:
                kwargs["pcp_size"] = 1
                print("vllm_fl: [PCP-MOE-LOCAL] forcing pcp_size=1 (experts "
                      "replicated over PCP; MoE local, no all_gather)",
                      file=sys.stderr, flush=True)
        return _orig_init(self, *args, **kwargs)

    FusedMoE.__init__ = _init
    FusedMoE._pcp_moe_local_patched = True
    print("vllm_fl: [PCP-MOE-LOCAL] FusedMoE.__init__ patched",
          file=sys.stderr, flush=True)


def apply_pcp_model_runner_patch() -> None:
    # The model-runner class cannot be imported at plugin-apply time: importing
    # vllm.v1.worker.gpu_model_runner pulls in vllm.config while it is still
    # mid-initialization (circular import), so every direct import here fails --
    # even the late register_model() re-apply. By the time vLLM itself imports
    # the module cleanly (config finished) and builds the runner, our apply hook
    # has long returned. Solution: arm a one-shot meta_path import hook now
    # (register() runs in EVERY process, workers included) that patches the
    # canonical GPUModelRunner class the instant vLLM finishes importing it.
    import sys
    import importlib.util
    import importlib.abc

    if _MOE_LOCAL:
        _install_moe_local_pcp()

    # PRIMARY TARGET: FL instantiates its OWN standalone runner ModelRunnerFL
    # (vllm_fl/worker/worker.py builds ModelRunnerFL, NOT vLLM's GPUModelRunner).
    # ModelRunnerFL does not subclass GPUModelRunner, so patching GPUModelRunner
    # below is a no-op for FL -- the zigzag _prepare_inputs never engages. Patch
    # ModelRunnerFL directly; its module is the plugin's own package and imports
    # cleanly at apply time (no circular import through vllm.config).
    try:
        from vllm_fl.worker.model_runner import ModelRunnerFL
    except Exception as e:  # noqa: BLE001
        print(f"vllm_fl: [PCP-PATCH] ModelRunnerFL import FAILED at apply time: "
              f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
    else:
        _install_pcp_prepare_inputs_on(ModelRunnerFL)

    _MODNAME = "vllm.v1.worker.gpu_model_runner"

    # Fast path: module already imported cleanly -> patch immediately.
    mod = sys.modules.get(_MODNAME)
    if mod is not None and hasattr(mod, "GPUModelRunner"):
        _install_pcp_prepare_inputs_on(mod.GPUModelRunner)
        return

    if getattr(apply_pcp_model_runner_patch, "_hook_armed", False):
        return

    class _PcpRunnerImportHook(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != _MODNAME:
                return None
            # Disarm self so the real finders resolve the spec (no recursion).
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            apply_pcp_model_runner_patch._hook_armed = False
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            _orig_exec = spec.loader.exec_module

            def _exec(module):
                _orig_exec(module)
                try:
                    _install_pcp_prepare_inputs_on(module.GPUModelRunner)
                except Exception as e:  # noqa: BLE001
                    print(f"vllm_fl: [PCP-PATCH] post-import patch FAILED: "
                          f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)

            spec.loader.exec_module = _exec
            return spec

    sys.meta_path.insert(0, _PcpRunnerImportHook())
    apply_pcp_model_runner_patch._hook_armed = True
    print("vllm_fl: [PCP-PATCH] armed import hook for gpu_model_runner",
          file=sys.stderr, flush=True)
    logger.info("vllm_fl: PCP model-runner import hook armed")


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


def _gdn_forward_batchsplit(self, hidden_states, output, pcp_group, P, rank, gmd):
    """v2 (Huawei RFC #37995 style): batch-split GDN.

    Instead of every PCP rank redundantly scanning the WHOLE sequence of ALL
    requests (the island path), each rank runs the GDN scan on only its
    assigned subset of requests (num_reqs/P of them, full-length).  Requests
    are independent for linear attention, so this is exact and removes the
    per-rank scan redundancy (compute ~/P).

    Correctness-first version: we still all-gather the full hidden (so every
    rank can build per-request full sequences), run GDN on this rank's request
    range only, then all-reduce the (disjoint) per-request outputs back so
    every rank holds the full output, and slice out the local segment.
    (Comm not yet optimal — a later step can drop the input gather via an
    all-to-all.  Requires num_reqs >= P; caller falls back to island otherwise.)
    """
    import torch as _torch
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.fla.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    device = hidden_states.device
    H = hidden_states.shape[-1]
    local_actual = int(gmd.non_spec_query_start_loc[-1].item())

    # Per-request full layout across ranks (reuse the island reshuffle plan).
    local_qsl = gmd.non_spec_query_start_loc.to(_torch.int64)
    gathered_qsl = pcp_group.all_gather(local_qsl.contiguous(), dim=0)
    all_qsl_cpu = gathered_qsl.view(P, -1).cpu()
    rs = _build_gdn_reshuffle(all_qsl_cpu, rank)
    perm = _torch.as_tensor(rs["perm"], device=device, dtype=_torch.long)
    inv_local = _torch.as_tensor(rs["inv_local"], device=device, dtype=_torch.long)
    full_qsl = _torch.as_tensor(rs["full_qsl"], dtype=_torch.int64)  # cpu [R+1]
    num_reqs = rs["num_reqs"]
    full_total = rs["full_total"]
    max_total = rs["max_total"]

    # All-gather full hidden of ALL requests onto every rank.
    padded = hidden_states.new_zeros((max_total, H))
    padded[:local_actual] = hidden_states[:local_actual]
    gathered = pcp_group.all_gather(padded, dim=0)
    full_hidden = gathered.index_select(0, perm).contiguous()  # [full_total, H]

    # Assign a contiguous request range to this rank.
    per = (num_reqs + P - 1) // P
    lo = min(rank * per, num_reqs)
    hi = min(lo + per, num_reqs)
    tok_lo = int(full_qsl[lo].item())
    tok_hi = int(full_qsl[hi].item())
    my_ntok = tok_hi - tok_lo

    full_output = full_hidden.new_zeros((full_total, H))
    if hi > lo and my_ntok > 0:
        n_my = hi - lo
        sub_qsl = (full_qsl[lo:hi + 1] - full_qsl[lo]).to(_torch.int32)  # rebased
        sub_qsl_cpu = sub_qsl.cpu()
        chunk_indices = prepare_chunk_indices(sub_qsl_cpu, FLA_CHUNK_SIZE).to(device)
        chunk_offsets = prepare_chunk_offsets(sub_qsl_cpu, FLA_CHUNK_SIZE).to(device)
        nums_dict, batch_ptr, tok_off = compute_causal_conv1d_metadata(
            sub_qsl_cpu, device=device
        )
        has_init = _torch.zeros(n_my, dtype=_torch.bool, device=device)
        sub_md = GDNAttentionMetadata(
            num_prefills=n_my,
            num_prefill_tokens=my_ntok,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=my_ntok,
            has_initial_state=has_init,
            non_spec_query_start_loc=sub_qsl.to(device),
            non_spec_state_indices_tensor=gmd.non_spec_state_indices_tensor[lo:hi],
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=tok_off,
        )
        sub_in = full_hidden[tok_lo:tok_hi].contiguous()
        sub_out = full_hidden.new_zeros((my_ntok, H))
        ctx = get_forward_context()
        md = ctx.attn_metadata
        saved = md[self.prefix]
        md[self.prefix] = sub_md
        try:
            self._pcp_orig_gdn_forward(sub_in, sub_out)
        finally:
            md[self.prefix] = saved
        full_output[tok_lo:tok_hi] = sub_out

    # Ranks filled DISJOINT token spans; sum-reduce assembles the whole output.
    full_output = pcp_group.all_reduce(full_output)
    output[:local_actual] = full_output.index_select(0, inv_local)
    _dbg(f"gdn-batch rank={rank} reqs[{lo}:{hi}] my_ntok={my_ntok} "
         f"full_total={full_total} out_norm={output[:local_actual].float().norm().item():.3f}")


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

    # v2: batch-split GDN when enabled and there are enough requests to split
    # one whole request per rank. Falls back to the island path otherwise.
    if (os.environ.get("VLLM_PCP_GDN", "island") == "batch"
            and getattr(gmd, "num_prefills", 0) >= P):
        return _gdn_forward_batchsplit(self, hidden_states, output,
                                       pcp_group, P, rank, gmd)

    # v2 diagnostic: "local" = run GDN only on this rank's local token segment
    # (no gather, no state-passing) — NUMERICALLY WRONG but a timing UPPER BOUND
    # for "GDN compute halved + parallel + no gather" (best case any head-split
    # / ideal-ring could reach). If this doesn't beat island, GDN compute isn't
    # the bottleneck and no GDN-parallelization is worth pursuing.
    if os.environ.get("VLLM_PCP_GDN", "island") == "local":
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


# ============== inlined from qwen36_pcp_v5 (parallel two-pass GDN) ==============
def _dbg_v5s2(msg):
    if _PCP_DEBUG:
        import sys as _sys
        print(f'[PCP-DBG-v4s2] {msg}', file=_sys.stderr, flush=True)

def _pcp_group_or_none():
    try:
        from vllm.distributed.parallel_state import get_pcp_group

        g = get_pcp_group()
        return g if g.world_size > 1 else None
    except (AssertionError, ImportError):
        return None


def _twopass_impl(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    use_qk_l2norm_in_kernel,
    pcp_group,
):
    """Two-pass GDN forward.  Returns (o, final_state).

    Mirrors chunk_gated_delta_rule_fwd (fla/ops/chunk.py) but splits the
    h-scan into Pass 1 / probe / recombine / Pass 2 across PCP ranks.
    """
    from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
    )
    from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o
    from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
        chunk_scaled_dot_kkt_fwd,
    )
    from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
    from vllm.model_executor.layers.fla.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
    from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril
    from vllm.model_executor.layers.fla.ops.utils import FLA_CHUNK_SIZE
    from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd

    P = pcp_group.world_size
    rank = pcp_group.rank_in_group

    _out_dtype = q.dtype
    if _GDN_FP32:
        q = q.float()
        k = k.float()
        v = v.float()
        g = g.float()
        beta = beta.float()
        if initial_state is not None:
            initial_state = initial_state.float()

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    BT = FLA_CHUNK_SIZE
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    if chunk_offsets is None:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    # --- shared precompute (identical to chunk_gated_delta_rule_fwd) ---
    g_cs = chunk_local_cumsum(
        g, chunk_size=BT, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g_cs,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g_cs,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    # --- Pass 1: local segment scan from s0 (this rank's initial_state) ---
    h1, v_new1, final_state_p1 = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_cs,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=BT,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )

    # final_state shape [N, H, V, K] (fp32).
    N, H, V, K = final_state_p1.shape
    assert V >= K, (
        f"pcp-v4s2: identity-probe needs V>=K, got V={V} K={K}; "
        "cannot recover Phi -> use ring (Step 1) instead."
    )

    # --- Probe: Phi_i = segment transition, via u=0 + identity initial state ---
    ident = torch.zeros(N, H, V, K, dtype=torch.float32, device=k.device)
    eye = torch.eye(K, dtype=torch.float32, device=k.device)
    ident[:, :, :K, :] = eye  # per (n,h): I_K embedded in the [V,K] state
    u_zero = torch.zeros_like(u)
    _, _, Phi = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u_zero,
        g=g_cs,
        initial_state=ident,
        output_final_state=True,
        chunk_size=BT,
        save_new_value=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    # Phi: [N, H, V, K]; rows [:K] hold the [K,K] transition (I @ Phi = Phi).
    Phi_kk = Phi[:, :, :K, :].contiguous()  # [N, H, K, K]

    # --- Recombine: all_gather + local affine prefix scan over P segments ---
    all_final = pcp_group.all_gather(
        final_state_p1.unsqueeze(0).contiguous(), 0
    )  # [P, N, H, V, K]
    all_Phi = pcp_group.all_gather(
        Phi_kk.unsqueeze(0).contiguous(), 0
    )  # [P, N, H, K, K]

    s0 = (
        initial_state.to(torch.float32)
        if initial_state is not None
        else torch.zeros_like(final_state_p1)
    )
    updated = final_state_p1.new_empty(P, N, H, V, K)
    updated[0] = all_final[0]
    for i in range(1, P):
        # correct_out_i = P_i + (correct_out_{i-1} - s0) @ Phi_i   (right-mult on K)
        delta = updated[i - 1] - s0  # [N, H, V, K]
        updated[i] = all_final[i] + torch.matmul(delta, all_Phi[i])
    final_state = updated[P - 1]  # true full-request final state (same on all ranks)

    # --- outputs ---
    if rank == 0:
        # Pass 1 already ran from the true s0 for the first segment -> correct.
        o = chunk_fwd_o(
            q=q,
            k=k,
            v=v_new1,
            h=h1,
            g=g_cs,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
    else:
        # Pass 2: rerun this segment from the corrected incoming boundary state.
        this_in = updated[rank - 1].contiguous()
        h2, v_new2, _ = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g_cs,
            initial_state=this_in,
            output_final_state=False,
            chunk_size=BT,
            save_new_value=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        )
        o = chunk_fwd_o(
            q=q,
            k=k,
            v=v_new2,
            h=h2,
            g=g_cs,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    if os.environ.get("VLLM_PCP_GDN_ISLAND") == "1" and cu_seqlens.numel() == 2:
        # Island override: recompute o as this rank's slice of a MONOLITHIC scan of
        # the all-gathered full sequence (zero recombine rounding; bit-exact to a
        # single-GPU GDN). Decisive test of whether the two-pass recombine ~7e-4 is
        # what drives the count drift.
        n_loc = int(cu_seqlens[-1].item())
        if n_loc >= 1:
            def _agi(x):  # [1, n_loc, ...] -> [1, P*n_loc, ...] rank order
                g = pcp_group.all_gather(x.contiguous(), 0)
                return g.reshape(1, P * n_loc, *x.shape[2:]).contiguous()
            q_i, k_i, w_i, u_i, g_i = _agi(q), _agi(k), _agi(w), _agi(u), _agi(g_cs)
            cu_i = torch.tensor([0, P * n_loc], device=k.device, dtype=torch.int32)
            ci_i = prepare_chunk_indices(cu_i, BT)
            co_i = prepare_chunk_offsets(cu_i, BT)
            s0_i = None if initial_state is None else s0
            h_i, vn_i, _ = chunk_gated_delta_rule_fwd_h(
                k=k_i, w=w_i, u=u_i, g=g_i, initial_state=s0_i,
                output_final_state=True, chunk_size=BT, save_new_value=True,
                cu_seqlens=cu_i, chunk_indices=ci_i, chunk_offsets=co_i,
            )
            o_i = chunk_fwd_o(q=q_i, k=k_i, v=vn_i, h=h_i, g=g_i, scale=scale,
                              cu_seqlens=cu_i, chunk_indices=ci_i)
            o = o_i[:, rank * n_loc:(rank + 1) * n_loc].contiguous()

    if _PCP_DEBUG:
        _dbg_v5s2(
            f"twopass rank={rank}/{P} N={N} H={H} V={V} K={K} "
            f"fp32={_GDN_FP32} qdt={q.dtype} odt_out={_out_dtype} "
            f"o_norm={o.float().norm().item():.3f} "
            f"fs_norm={final_state.float().norm().item():.3f}"
        )

    if os.environ.get("VLLM_PCP_TWOPASS_SELFCHECK") == "1" and cu_seqlens.numel() == 2:
        # Internal self-consistency test (TP4-independent):
        #   h_serial = monolithic scan of the FULL gathered sequence from s0
        #   h_pass2  = this rank's two-pass output `o`
        # WY (w,u) and g_cs are chunk-local and chunk boundaries align with rank
        # boundaries (n_local % BT == 0), so concatenating per-rank w,u,g == the
        # true full-sequence w,u,g -> the gathered monolithic scan IS h_serial.
        n_loc = int(cu_seqlens[-1].item())
        if n_loc >= 1024:  # skip decode/warmup forwards
            try:
                def _ag(x):  # x: [B=1, n_loc, ...] -> [1, P*n_loc, ...] (rank order)
                    g = pcp_group.all_gather(x.contiguous(), 0)  # [P, n_loc, ...]
                    return g.reshape(1, P * n_loc, *x.shape[2:]).contiguous()
                q_f, k_f, w_f, u_f, g_f = _ag(q), _ag(k), _ag(w), _ag(u), _ag(g_cs)
                cu_f = torch.tensor([0, P * n_loc], device=k.device, dtype=torch.int32)
                ci_f = prepare_chunk_indices(cu_f, BT)
                co_f = prepare_chunk_offsets(cu_f, BT)
                s0_f = None if initial_state is None else s0
                h_f, vnew_f, fs_f = chunk_gated_delta_rule_fwd_h(
                    k=k_f, w=w_f, u=u_f, g=g_f, initial_state=s0_f,
                    output_final_state=True, chunk_size=BT, save_new_value=True,
                    cu_seqlens=cu_f, chunk_indices=ci_f, chunk_offsets=co_f,
                )
                o_f = chunk_fwd_o(q=q_f, k=k_f, v=vnew_f, h=h_f, g=g_f, scale=scale,
                                  cu_seqlens=cu_f, chunk_indices=ci_f)
                dstate = (final_state.float() - fs_f.float()).norm() / (fs_f.float().norm() + 1e-9)
                o_seg = o_f[:, rank * n_loc:(rank + 1) * n_loc]
                if o.shape == o_seg.shape and o_seg.numel() > 0:
                    do = (o.float() - o_seg.float()).norm() / (o_seg.float().norm() + 1e-9)
                    dmax = (o.float() - o_seg.float()).abs().max().item()
                    dostr = f"{do.item():.3e} max|do|={dmax:.3e}"
                else:
                    dostr = f"SHAPE_MISMATCH o={tuple(o.shape)} o_f={tuple(o_f.shape)} o_seg={tuple(o_seg.shape)}"
                _dbg_v5s2(
                    f"SELFCHECK rank={rank}/{P} n_loc={n_loc} "
                    f"d_o(pass2_vs_serial)={dostr} "
                    f"d_finalstate(recombine_vs_mono)={dstate.item():.3e}"
                )
            except Exception as e:  # noqa: BLE001
                _dbg_v5s2(f"SELFCHECK rank={rank} FAILED {type(e).__name__}: {e}")

    return o.to(_out_dtype), (final_state if output_final_state else None)


def _make_twopass_wrapper(orig):
    """Wrap the original chunk_gated_delta_rule with the PCP two-pass path.

    Binds call args against the ORIGINAL signature (robust to minor version
    differences) and falls through to orig on any unsupported case.
    """
    try:
        sig = inspect.signature(orig)
    except (TypeError, ValueError):
        sig = None

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        pcp_group = _pcp_group_or_none()
        if pcp_group is None or sig is None:
            return orig(*args, **kwargs)
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            cu_seqlens = a.get("cu_seqlens", None)
            # Only handle variable-length prefill (the PCP-split prefill path).
            if cu_seqlens is None:
                return orig(*args, **kwargs)
            return _twopass_impl(
                q=a["q"],
                k=a["k"],
                v=a["v"],
                g=a["g"],
                beta=a["beta"],
                scale=a.get("scale") or (a["k"].shape[-1] ** -0.5),
                initial_state=a.get("initial_state", None),
                output_final_state=a.get("output_final_state", False),
                cu_seqlens=cu_seqlens,
                chunk_indices=a.get("chunk_indices", None),
                chunk_offsets=a.get("chunk_offsets", None),
                use_qk_l2norm_in_kernel=a.get("use_qk_l2norm_in_kernel", False),
                pcp_group=pcp_group,
            )
        except Exception as e:  # pragma: no cover - safety net
            logger.warning(
                "vllm_fl: PCP v4s2 two-pass failed (%s: %s); falling back to "
                "original chunk_gated_delta_rule",
                type(e).__name__,
                e,
            )
            return orig(*args, **kwargs)

    return wrapper



# ============== inlined from qwen36_pcp_v5_1 (install hooks + conv halo) ==============
"""Qwen3.6 hybrid (GDN) + PCP — v5.1.

Two fixes over v5 (qwen36_pcp_v5.py):

1. RELIABLE two-pass install.  v5's apply_pcp_gdn_twopass_patch() runs at
   plugin-apply time, when ``import vllm.model_executor.layers.fla.ops.chunk``
   hits a circular import (vllm.config still initializing).  v5 swallows the
   ImportError and *permanently* skips (log: "fla.ops.chunk not found, skip
   v4s2 GDN patch"), so the parallel two-pass GDN never engages and the model
   silently falls back to a degenerate full-sequence path.  v5.1 defers the
   install with a one-shot meta_path import hook on the GDN *layer* module
   (which loads late, after config init, when every dependency imports
   cleanly), mirroring apply_pcp_model_runner_patch (qwen36_pcp_v3.py).

2. Conv halo.  The GDN causal_conv1d (kernel width ``linear_conv_kernel_dim``,
   =4 for Qwen3.6) needs the previous PCP rank's last (width-1) conv inputs at
   the shard boundary.  v5 runs the conv purely local (zero left-pad) → the
   first (width-1) outputs of every rank>0 are wrong.  v5.1 wraps
   causal_conv1d_fn: on a single-request contiguous PCP prefill it all_gathers
   the per-rank conv-input tail ([P, D, width-1], tiny) and feeds the left
   neighbour's tail as this rank's conv initial state (has_initial_state=True);
   rank 0 keeps the zero left-pad (true sequence start).

Reuses v5's numerical two-pass (_twopass_impl via _make_twopass_wrapper) and
v3's guard/attention/model-runner patches unchanged.

Enable with:  VLLM_PCP_V5_1=1
Scope (same as v5): fresh single-request, equal-length, CONTIGUOUS split.
"""

import functools
import importlib
import importlib.abc
import importlib.util
import inspect
import logging
import os
import sys

import torch

# Reuse v3's guard / attention / model-runner patches unchanged.
# Reuse v5's two-pass op + pcp-group helper.

logger = logging.getLogger(__name__)

# GDN layer module candidates (Qwen3.6 uses gdn_linear_attn).  Whichever loads
# first triggers the install; loading it also imports fla.ops.chunk and
# causal_conv1d cleanly, so we patch every reference from one safe callback.
_GDN_LAYER_MODULES = (
    "vllm.model_executor.layers.mamba.gdn_linear_attn",
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn",
    "vllm.model_executor.layers.mamba.gdn.base",
)
_CHUNK_MODULE = "vllm.model_executor.layers.fla.ops.chunk"
_CHUNK_PKG = "vllm.model_executor.layers.fla.ops"
_CONV_MODULE = "vllm.model_executor.layers.mamba.ops.causal_conv1d"

_TWOPASS_WRAPPER = None  # created once from the real orig
_CONV_WRAPPER = None
_INSTALLED = {"twopass": False, "conv": False}


# ---------------------------------------------------------------------------
# Conv halo wrapper
# ---------------------------------------------------------------------------
def _make_conv_halo_wrapper(orig):
    """Wrap causal_conv1d_fn to supply the left-neighbour halo as conv init
    state on PCP contiguous single-request prefill.  Falls through to orig on
    any unsupported case (decode, multi-request, non-PCP)."""
    try:
        sig = inspect.signature(orig)
    except (TypeError, ValueError):
        sig = None

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        pcp = _pcp_group_or_none()
        if (
            pcp is None
            or getattr(pcp, "world_size", 1) <= 1
            or sig is None
            or os.environ.get("VLLM_PCP_V51_HALO", "1") == "0"
        ):
            return orig(*args, **kwargs)
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            x = a.get("x")
            weight = a.get("weight")
            qsl = a.get("query_start_loc")
            conv_states = a.get("conv_states")
            cache_indices = a.get("cache_indices")
            has_init = a.get("has_initial_state")
            # Scope: single-request varlen prefill, x = [D, L] (2D packed).
            if (
                x is None
                or weight is None
                or qsl is None
                or conv_states is None
                or x.dim() != 2
                or qsl.numel() != 2
            ):
                return orig(*args, **kwargs)
            width = weight.shape[-1]
            h = width - 1
            # Real sequence end = query_start_loc[-1], NOT x.shape[-1]: under PCP
            # the conv input is padded to num_actual_tokens (n_buf) while only the
            # front [0:n_real] holds real tokens, so x[:, L-h:L] with L=x.shape[-1]
            # would grab the PHANTOM padding tail as the next rank's conv init
            # state. Mirrors Huawei extract_last_width(mixed_qkv, query_start_loc,
            # state_len).
            seq_end = int(qsl[-1].item())
            if h <= 0 or seq_end <= 0 or seq_end > x.shape[-1] or seq_end < h:
                return orig(*args, **kwargs)

            P = pcp.world_size
            r = pcp.rank_in_group
            tail = x[:, seq_end - h : seq_end].contiguous()  # [D, h] = last h REAL inputs
            all_tails = pcp.all_gather(
                tail.unsqueeze(0).contiguous(), 0
            )  # [P, D, h]

            if r > 0:
                # This rank's true left context = previous rank's tail.
                slot = (
                    int(cache_indices[0].item())
                    if cache_indices is not None
                    else 0
                )
                conv_states[slot, :, :].copy_(all_tails[r - 1])
                if has_init is None:
                    has_init = torch.ones(1, dtype=torch.bool, device=x.device)
                else:
                    has_init = has_init.clone()
                    has_init[0] = True
                a["has_initial_state"] = has_init
            # rank 0 keeps its original (zero) left pad = true sequence start.
            out = orig(*bound.args, **bound.kwargs)
            if os.environ.get("VLLM_PCP_CONV_SELFCHECK") == "1" and seq_end >= 1024:
                try:
                    xr = x[:, :seq_end].contiguous()                  # [D, seq_end]
                    xg = pcp.all_gather(xr.unsqueeze(0).contiguous(), 0)  # [P, D, seq_end]
                    D = xr.shape[0]
                    x_full = xg.permute(1, 0, 2).reshape(D, P * seq_end).contiguous()
                    qsl_full = torch.tensor([0, P * seq_end], device=x.device, dtype=qsl.dtype)
                    cs_full = torch.zeros_like(conv_states)
                    a2 = dict(a)
                    a2["x"] = x_full
                    a2["query_start_loc"] = qsl_full
                    a2["conv_states"] = cs_full
                    a2["has_initial_state"] = None
                    b2 = sig.bind(**{k: v for k, v in a2.items()})
                    b2.apply_defaults()
                    out_full = orig(*b2.args, **b2.kwargs)
                    if isinstance(out_full, tuple):
                        out_full = out_full[0]
                    outr = out[0] if isinstance(out, tuple) else out
                    o_seg = out_full[:, r * seq_end:(r + 1) * seq_end]
                    ol = outr[:, :seq_end]
                    if o_seg.shape == ol.shape:
                        d = (ol.float() - o_seg.float()).norm() / (o_seg.float().norm() + 1e-9)
                        # boundary-only: first h outputs of this rank's segment
                        db = (ol[:, :h].float() - o_seg[:, :h].float()).norm() / (o_seg[:, :h].float().norm() + 1e-9)
                        _dbg_v5s2(f"CONV-SELFCHECK rank={r}/{P} seq_end={seq_end} "
                                  f"d_conv(halo_vs_mono)={d.item():.3e} d_boundary[:{h}]={db.item():.3e}")
                except Exception as e:  # noqa: BLE001
                    _dbg_v5s2(f"CONV-SELFCHECK rank={r} FAILED {type(e).__name__}: {e}")
            return out
        except Exception as e:  # pragma: no cover - safety net
            logger.warning(
                "vllm_fl: v5.1 conv halo failed (%s: %s); falling back to "
                "original causal_conv1d_fn",
                type(e).__name__,
                e,
            )
            return orig(*args, **kwargs)

    wrapper._pcp_v51_conv_halo = True  # type: ignore[attr-defined]
    return wrapper


# ---------------------------------------------------------------------------
# Install (runs from the deferred hook, when all imports are clean)
# ---------------------------------------------------------------------------
def _install_twopass(gdn_mod) -> None:
    global _TWOPASS_WRAPPER
    if _INSTALLED["twopass"]:
        # still (re)bind this GDN module's alias in case it loaded later.
        if _TWOPASS_WRAPPER is not None:
            _rebind_twopass_on_layer(gdn_mod)
        return
    try:
        chunk_mod = importlib.import_module(_CHUNK_MODULE)
    except Exception as e:  # noqa: BLE001
        logger.warning("vllm_fl: v5.1 import %s failed (%s); two-pass not installed",
                       _CHUNK_MODULE, e)
        return
    orig = getattr(chunk_mod, "chunk_gated_delta_rule", None)
    if orig is None:
        logger.warning("vllm_fl: v5.1 chunk_gated_delta_rule missing")
        return
    if getattr(orig, "_pcp_v4s2_wrapped", False):
        _TWOPASS_WRAPPER = orig
    else:
        _TWOPASS_WRAPPER = _make_twopass_wrapper(orig)
        _TWOPASS_WRAPPER._pcp_v4s2_wrapped = True  # type: ignore[attr-defined]

    patched = []
    setattr(chunk_mod, "chunk_gated_delta_rule", _TWOPASS_WRAPPER)
    patched.append("fla.ops.chunk")
    for modname in (_CHUNK_PKG, _CHUNK_PKG + ".__init__"):
        m = sys.modules.get(modname)
        if m is not None and getattr(m, "chunk_gated_delta_rule", None) is orig:
            setattr(m, "chunk_gated_delta_rule", _TWOPASS_WRAPPER)
            patched.append(modname)
    _INSTALLED["twopass"] = True
    _rebind_twopass_on_layer(gdn_mod, patched)
    logger.info("vllm_fl: v5.1 two-pass GDN patch applied to: %s", ", ".join(patched))


def _rebind_twopass_on_layer(gdn_mod, patched=None) -> None:
    for attr in ("fla_chunk_gated_delta_rule", "chunk_gated_delta_rule"):
        cur = getattr(gdn_mod, attr, None)
        if cur is not None and cur is not _TWOPASS_WRAPPER:
            setattr(gdn_mod, attr, _TWOPASS_WRAPPER)
            if patched is not None:
                patched.append(f"{gdn_mod.__name__}.{attr}")


def _install_conv_halo(gdn_mod) -> None:
    global _CONV_WRAPPER
    if not _INSTALLED["conv"]:
        try:
            conv_mod = importlib.import_module(_CONV_MODULE)
        except Exception as e:  # noqa: BLE001
            logger.warning("vllm_fl: v5.1 import %s failed (%s); conv halo not installed",
                           _CONV_MODULE, e)
            return
        orig = getattr(conv_mod, "causal_conv1d_fn", None)
        if orig is None:
            logger.warning("vllm_fl: v5.1 causal_conv1d_fn missing")
            return
        if getattr(orig, "_pcp_v51_conv_halo", False):
            _CONV_WRAPPER = orig
        else:
            _CONV_WRAPPER = _make_conv_halo_wrapper(orig)
        setattr(conv_mod, "causal_conv1d_fn", _CONV_WRAPPER)
        _INSTALLED["conv"] = True
        logger.info("vllm_fl: v5.1 conv halo wrapper installed on %s", _CONV_MODULE)
    # (re)bind the GDN layer module's imported reference.
    cur = getattr(gdn_mod, "causal_conv1d_fn", None)
    if cur is not None and cur is not _CONV_WRAPPER:
        setattr(gdn_mod, "causal_conv1d_fn", _CONV_WRAPPER)
        logger.info("vllm_fl: v5.1 conv halo bound on %s.causal_conv1d_fn",
                    gdn_mod.__name__)


def _install_all(gdn_mod) -> None:
    _install_twopass(gdn_mod)
    _install_conv_halo(gdn_mod)


# ---------------------------------------------------------------------------
# Deferred install: hook the GDN layer module import (loads after config init).
# ---------------------------------------------------------------------------
def apply_pcp_gdn_v51_patch() -> None:
    # Fast path: any GDN layer module already imported -> install now.
    installed_any = False
    for modname in _GDN_LAYER_MODULES:
        mod = sys.modules.get(modname)
        if mod is not None:
            _install_all(mod)
            installed_any = True
    if installed_any and _INSTALLED["twopass"] and _INSTALLED["conv"]:
        return

    if getattr(apply_pcp_gdn_v51_patch, "_hook_armed", False):
        return

    targets = set(_GDN_LAYER_MODULES)

    class _PcpV51ImportHook(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in targets:
                return None
            # Disarm self BEFORE calling find_spec below, otherwise find_spec
            # re-enters this finder -> infinite recursion (RecursionError).
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            apply_pcp_gdn_v51_patch._hook_armed = False
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return None
            _orig_exec = spec.loader.exec_module

            def _exec(module):
                _orig_exec(module)
                try:
                    _install_all(module)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"vllm_fl: [PCP-v5.1] post-import install FAILED: "
                        f"{type(e).__name__}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )

            spec.loader.exec_module = _exec
            return spec

    sys.meta_path.insert(0, _PcpV51ImportHook())
    apply_pcp_gdn_v51_patch._hook_armed = True
    print("vllm_fl: [PCP-v5.1] armed GDN-layer import hook (two-pass + conv halo)",
          file=sys.stderr, flush=True)
    logger.info("vllm_fl: v5.1 GDN import hook armed")


# ---------------------------------------------------------------------------
# Combined registration
# ---------------------------------------------------------------------------
def apply_pcp_patch() -> None:
    apply_pcp_guard_patches()
    apply_pcp_attn_patch()
    apply_pcp_model_runner_patch()
    apply_pcp_gdn_v51_patch()


def apply_platform_patches() -> None:
    """Public entry point (called from vllm_fl.register() when VLLM_PCP_V5_1=1).
    Installs Qwen3.6 hybrid + PCP with the v5 parallel two-pass GDN (reliably
    installed) plus the conv halo.  Idempotent; self-gates on
    prefill_context_parallel_size > 1."""
    apply_pcp_patch()
    _maybe_install_shape_probe()
    _maybe_install_moe_sp()


def _maybe_install_shape_probe() -> None:
    """Gated (VLLM_PCP_SHAPE_PROBE=1) monkeypatch that logs decoder-layer and
    MoE input shapes in EVERY worker — for locating where the sequence expands
    to full-chunk before the MoE under PCP. Runs in serve workers (online)."""
    if os.environ.get("VLLM_PCP_SHAPE_PROBE") != "1":
        return
    try:
        import vllm.model_executor.models.qwen3_next as q
    except Exception:
        return
    _c = {"lyr": 0, "moe": 0}
    if not getattr(q.Qwen3NextDecoderLayer.forward, "_shape_probed", False):
        _ol = q.Qwen3NextDecoderLayer.forward
        def _lyr(self, hidden_states, *a, **k):
            if _c["lyr"] < 8:
                print(f"vllm_fl:[SHAPE] layer[{getattr(self,'layer_type','?')}] "
                      f"in={tuple(hidden_states.shape)}", file=sys.stderr, flush=True)
                _c["lyr"] += 1
            return _ol(self, hidden_states, *a, **k)
        _lyr._shape_probed = True
        q.Qwen3NextDecoderLayer.forward = _lyr
    if not getattr(q.Qwen3NextSparseMoeBlock.forward, "_shape_probed", False):
        _om = q.Qwen3NextSparseMoeBlock.forward
        def _moe(self, hidden_states):
            # Log the first REAL (non-zero-checksum) MoE calls per worker; the
            # warmup/profiling forward feeds all-zero dummies (checksum 0.0) so
            # skipping those lets us see the actual per-rank sequence length.
            if _c["moe"] < 12:
                try:
                    _cs = float(hidden_states.float().sum().item())
                except Exception:
                    _cs = -1.0
                if _cs != 0.0:
                    print(f"vllm_fl:[SHAPE] REAL MoE in={tuple(hidden_states.shape)} "
                          f"is_sp={getattr(self,'is_sequence_parallel',None)} "
                          f"checksum={_cs:.3f}",
                          file=sys.stderr, flush=True)
                    _c["moe"] += 1
            return _om(self, hidden_states)
        _moe._shape_probed = True
        q.Qwen3NextSparseMoeBlock.forward = _moe
    print("vllm_fl:[SHAPE] probe installed", file=sys.stderr, flush=True)


def _maybe_install_moe_sp() -> None:
    """Cut the redundant phantom-tail MoE compute under PCP (VLLM_PCP_MOE_TRUNC=1).

    Empirical layout (verified 2026-08-05, contiguous PCP, online serve, real
    8192-token prefill): the runner allocates a forward buffer padded to the
    FULL sequence length (n_buf) on every PCP rank, but only fills this rank's
    local slice [0:n_real] with real tokens; the tail [n_real:n_buf] is phantom
    (garbage, zeroed by the attention patch).  Measured per rank:
      pcp4  n_real=2048  n_buf=8192   (MoE wastes 3/4 of its work on phantom)
      pcp2  n_real=4096  n_buf=8192   (MoE wastes 1/2)
    So the "pcp4 MoE ~4x tp4" cost is NOT cross-rank redundancy and NOT a
    re-gather to a replicated full sequence -- it is the token-wise MoE
    redundantly processing the phantom padding.  (An earlier combined-group
    resharding attempt was wrong: it assumed a replicated full sequence; the
    per-rank MoE inputs actually differ, real data sits at the FRONT [0:n_real],
    and the checksums per rank differ.)

    Fix: run the MoE only on the real prefix [0:n_real] and zero the phantom
    tail.  MoE is token-independent so MoE(hidden)[0:n_real] == MoE(hidden[0:
    n_real]); the tail output is discarded downstream (attention reads only
    [0:n_real], logits come from real positions).  No cross-rank comm.
      pcp4: 8192->2048 (4x)   pcp2: 8192->4096 (2x)   tp4: n_real==n_buf, no-op.

    n_real/n_buf for the current prefill come from the attention patch via
    qwen36_pcp_v3._PCP_REAL.  Guarded by an exact n_buf match so decode steps
    and non-PCP forwards fall through to the original block unchanged.
    """
    if os.environ.get("VLLM_PCP_MOE_TRUNC") != "1":
        return
    try:
        import vllm.model_executor.models.qwen3_next as q
    except Exception:
        return
    if getattr(q.Qwen3NextSparseMoeBlock.forward, "_pcp_moe_trunc", False):
        return


    _orig = q.Qwen3NextSparseMoeBlock.forward
    _dbg = os.environ.get("VLLM_PCP_DEBUG") == "1"
    _check = os.environ.get("VLLM_PCP_MOE_TRUNC_CHECK") == "1"
    _cnt = {"n": 0, "chk": 0}

    def _forward(self, hidden_states):
        n_real = _PCP_REAL.get("n_real")
        n_buf = _PCP_REAL.get("n_buf")
        num_tokens = hidden_states.shape[0]
        # Only truncate a padded prefill buffer of exactly the recorded width
        # that has a real phantom tail.  Otherwise run the block unchanged.
        if (n_real is None or n_buf is None or num_tokens != n_buf
                or n_real <= 0 or n_real >= num_tokens):
            return _orig(self, hidden_states)

        real = hidden_states[:n_real]
        out_real = _orig(self, real)
        out = hidden_states.new_zeros((num_tokens, out_real.shape[-1]))
        out[:n_real] = out_real

        if _dbg and _cnt["n"] < 6:
            def _nan(t):
                tf = t.float()
                return (int(torch.isnan(tf).sum().item()),
                        int(torch.isinf(tf).sum().item()))
            hr_n, hr_i = _nan(hidden_states[:n_real])
            ht_n, ht_i = _nan(hidden_states[n_real:])
            or_n, or_i = _nan(out_real)
            print(f"vllm_fl:[PCP-MOE-TRUNC] n_real={n_real} n_buf={num_tokens} "
                  f"real_in(nan={hr_n},inf={hr_i}) tail_in(nan={ht_n},inf={ht_i}) "
                  f"out_real(nan={or_n},inf={or_i})",
                  file=sys.stderr, flush=True)
            _cnt["n"] += 1
        return out

    _forward._pcp_moe_trunc = True
    q.Qwen3NextSparseMoeBlock.forward = _forward
    print("vllm_fl:[PCP-MOE-TRUNC] installed (MoE skips PCP phantom tail)",
          file=sys.stderr, flush=True)