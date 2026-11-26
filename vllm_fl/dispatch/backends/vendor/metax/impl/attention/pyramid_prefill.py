"""3-stage smart observation-window pyramid page pruning for MetaX C500.

Schedule (OBS_WINDOW=256, 3-layer voting, neighborhood smoothing, min 28 pages):
  Stage 0: layers  0..10 (100% pages)
  Stage 1: layers 11..19 ( 35% pages)
  Stage 2: layers 20..27 ( 18% pages)
  Stage 3: layers 28..41 (  9% pages)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

PAGE = 16
SINK_PAGES = 4       # 64 tokens
TAIL_PAGES = 16      # 256 tokens (covers observation window + prompt tail)
OBS_WINDOW = 256     # last 256 tokens cover full abstract + question
MIN_KEPT_PAGES = SINK_PAGES + TAIL_PAGES + 8  # 28 pages = 448 tokens
MIN_PROMPT_TOKENS = 1024

STAGES = (
    (10, 0.35),
    (19, 0.18),
    (27, 0.09),
)
PRUNE_LAYERS = {layer: (idx + 1, frac) for idx, (layer, frac) in enumerate(STAGES)}
NUM_PRUNE_STAGES = len(STAGES)  # 3 pruned stages (1..3)

_LOGGED: set[str] = set()


def _once(key: str, msg: str, *args) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.warning(msg, *args)


def stage_of_layer(li: Optional[int]) -> int:
    if li is None or li <= 10:
        return 0
    if li <= 19:
        return 1
    if li <= 27:
        return 2
    return 3


@triton.jit
def _build_stage_tables_kernel(
    SEQ_IN,
    TBL_IN,
    DROP_PAGES,
    KEPT_FULL,
    SEQ_OUT,
    TBL_OUT,
    n_rows,
    n_cols,
    s_in,
    s_out,
    s_meta,
    BLOCK_C: tl.constexpr,
):
    stage = tl.program_id(0)   # 0, 1, 2 -> stages 1, 2, 3
    row = tl.program_id(1)
    col_blk = tl.program_id(2)
    if row >= n_rows:
        return
    meta_off = stage * s_meta + row
    drop = tl.load(DROP_PAGES + meta_off)
    kf = tl.load(KEPT_FULL + meta_off)
    if col_blk == 0:
        orig_seq = tl.load(SEQ_IN + row)
        new_seq = tl.maximum(orig_seq - drop * 16, 0)
        tl.store(SEQ_OUT + stage * s_meta + row, new_seq)
    cols = col_blk * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = cols < n_cols
    src_cols = tl.where((drop == 0) | (cols < kf), cols, cols + drop)
    in_mask = mask & (src_cols < s_in)
    vals = tl.load(TBL_IN + row.to(tl.int64) * s_in + src_cols, mask=in_mask, other=0)
    tl.store(
        TBL_OUT + (stage.to(tl.int64) * s_meta + row.to(tl.int64)) * s_out + cols,
        vals,
        mask=mask,
    )


class PyramidDecodeState:
    """Static GPU buffers for stage 1..3 decode seq_lens and block_tables (CUDA Graph safe)."""

    def __init__(self, device: torch.device, max_reqs: int, max_blocks: int):
        self.device = device
        self.max_reqs = int(max_reqs)
        self.max_blocks = int(max_blocks)
        self.seq_lens = torch.zeros(
            (NUM_PRUNE_STAGES, self.max_reqs), dtype=torch.int32, device=device
        )
        self.block_tables = torch.zeros(
            (NUM_PRUNE_STAGES, self.max_reqs, self.max_blocks), dtype=torch.int32, device=device
        )
        self.drop_gpu = torch.zeros(
            (NUM_PRUNE_STAGES, self.max_reqs), dtype=torch.int32, device=device
        )
        self.kept_gpu = torch.zeros(
            (NUM_PRUNE_STAGES, self.max_reqs), dtype=torch.int32, device=device
        )
        self.drop_cpu = torch.zeros(
            (NUM_PRUNE_STAGES, self.max_reqs), dtype=torch.int32, pin_memory=True
        )
        self.kept_cpu = torch.zeros(
            (NUM_PRUNE_STAGES, self.max_reqs), dtype=torch.int32, pin_memory=True
        )
        self._drop_np = self.drop_cpu.numpy()
        self._kept_np = self.kept_cpu.numpy()
        self._last_sig = None
        self._last_has_pruned = False
        self.active = False

    def sync(
        self,
        seq_in: torch.Tensor,
        tbl_in: torch.Tensor,
        drop_rows: list[tuple[int, int, int]],
        kept_rows: list[tuple[int, int, int]],
        max_copy_cols: int,
        sig=None,
    ) -> None:
        n_rows = int(seq_in.shape[0])
        if n_rows == 0 or n_rows > self.max_reqs:
            return
        n_cols = min(int(tbl_in.shape[1]), self.max_blocks, max(1, int(max_copy_cols)))
        if sig is None or sig != self._last_sig:
            has_pruned = False
            d_np = self._drop_np
            k_np = self._kept_np
            d_np[:, :n_rows] = 0
            k_np[:, :n_rows] = 0
            n_fill = min(n_rows, len(drop_rows))
            for r in range(n_fill):
                dr = drop_rows[r]
                if dr[0] or dr[1] or dr[2]:
                    has_pruned = True
                    kr = kept_rows[r]
                    d_np[0, r] = dr[0]
                    d_np[1, r] = dr[1]
                    d_np[2, r] = dr[2]
                    k_np[0, r] = kr[0]
                    k_np[1, r] = kr[1]
                    k_np[2, r] = kr[2]
            self.drop_gpu[:, :n_rows].copy_(self.drop_cpu[:, :n_rows], non_blocking=True)
            self.kept_gpu[:, :n_rows].copy_(self.kept_cpu[:, :n_rows], non_blocking=True)
            self._last_sig = sig
            self._last_has_pruned = has_pruned
        else:
            has_pruned = self._last_has_pruned
        grid = (NUM_PRUNE_STAGES, n_rows, triton.cdiv(n_cols, 128))
        _build_stage_tables_kernel[grid](
            seq_in,
            tbl_in,
            self.drop_gpu,
            self.kept_gpu,
            self.seq_lens,
            self.block_tables,
            n_rows,
            n_cols,
            tbl_in.stride(0),
            self.max_blocks,
            self.max_reqs,
            BLOCK_C=128,
            num_warps=4,
        )
        self.active = True
        if has_pruned:
            _once(
                "decode_tables",
                "E469 pyramid3 decode tables active stages=%d rows=%d cols=%d",
                NUM_PRUNE_STAGES,
                n_rows,
                n_cols,
            )

    def get_stage_views(
        self, stage: int, n_rows: int, n_cols: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = stage - 1
        return self.seq_lens[s, :n_rows], self.block_tables[s, :n_rows, :n_cols]


_DECODE_STATE: Optional[PyramidDecodeState] = None


def get_decode_state() -> Optional[PyramidDecodeState]:
    return _DECODE_STATE


def ensure_decode_state(
    device: torch.device, max_reqs: int, max_blocks: int
) -> PyramidDecodeState:
    global _DECODE_STATE
    if (
        _DECODE_STATE is None
        or _DECODE_STATE.device != device
        or _DECODE_STATE.max_reqs < max_reqs
        or _DECODE_STATE.max_blocks < max_blocks
    ):
        _DECODE_STATE = PyramidDecodeState(device, max_reqs, max_blocks)
    return _DECODE_STATE


@dataclass
class PrefillCapture:
    target_layer: int
    cu_q_cpu: list[int]
    orig_full_pages: list[int]
    orig_rem: list[int]
    target_frac: float
    vote_layers: tuple[int, ...] = ()
    accum_scores: Optional[list[Optional[torch.Tensor]]] = None
    kept_token_idx: Optional[torch.Tensor] = None
    new_qlen: Optional[list[int]] = None
    new_full_pages: Optional[list[int]] = None
    drop_pages: Optional[list[int]] = None


_ACTIVE_CAPTURE: Optional[PrefillCapture] = None


def set_active_capture(cap: Optional[PrefillCapture]) -> None:
    global _ACTIVE_CAPTURE
    _ACTIVE_CAPTURE = cap


def get_active_capture() -> Optional[PrefillCapture]:
    return _ACTIVE_CAPTURE


@triton.jit
def _obs_fused_score_kernel(
    Q, K, OUT,
    N,
    sq_h, sq_m, sq_d,
    sk_h, sk_n, sk_d,
    so_h, so_n,
    scale,
    QPK: tl.constexpr,
    OBS_WIN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    D: tl.constexpr,
):
    hq = tl.program_id(0)
    pm = tl.program_id(1)
    hkv = hq // QPK

    rm = pm * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    q = tl.load(Q + hq * sq_h + rm[:, None] * sq_m + rd[None, :] * sq_d).to(tl.bfloat16)
    q_pos = (N - OBS_WIN) + rm

    m_i = tl.full((BM,), -1e9, dtype=tl.float32)
    l_i = tl.zeros((BM,), dtype=tl.float32)
    for n0 in range(0, N, BN):
        rn = n0 + tl.arange(0, BN)
        n_mask = rn < N
        k = tl.load(
            K + hkv * sk_h + rn[:, None] * sk_n + rd[None, :] * sk_d,
            mask=n_mask[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        valid = n_mask[None, :] & (rn[None, :] <= q_pos[:, None])
        s = tl.where(valid, s, -1e9)
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.where(valid, tl.exp(s - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    inv_l = 1.0 / tl.maximum(l_i, 1e-12)
    for n0 in range(0, N, BN):
        rn = n0 + tl.arange(0, BN)
        n_mask = rn < N
        k = tl.load(
            K + hkv * sk_h + rn[:, None] * sk_n + rd[None, :] * sk_d,
            mask=n_mask[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        valid = n_mask[None, :] & (rn[None, :] <= q_pos[:, None])
        p = tl.where(valid, tl.exp(s - m_i[:, None]) * inv_l[:, None], 0.0)
        col_sum = tl.sum(p, axis=0)
        tl.atomic_add(OUT + hq * so_h + rn * so_n, col_sum, mask=n_mask)


@torch.inference_mode()
def capture_and_select(
    cap: PrefillCapture,
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float,
    commit: bool = True,
) -> None:
    """Smart page scoring: OBS_WINDOW=256 + per-head normalized sum + multi-layer voting + [0.5,1,0.5] smoothing."""
    device = q.device
    hq = q.shape[1]
    hkv = k.shape[1]
    qpk = hq // hkv
    n_seq = len(cap.cu_q_cpu) - 1
    if cap.accum_scores is None:
        cap.accum_scores = [None] * n_seq

    kept_chunks: list[torch.Tensor] = []
    new_qlen: list[int] = []
    new_full_pages: list[int] = []
    drop_pages: list[int] = []

    for i in range(n_seq):
        s0 = cap.cu_q_cpu[i]
        s1 = cap.cu_q_cpu[i + 1]
        cur_len = s1 - s0
        rem = cap.orig_rem[i]
        cur_full_pages = (cur_len - rem) // PAGE
        assert cur_full_pages * PAGE + rem == cur_len, (cur_len, rem)

        orig_fp = cap.orig_full_pages[i]
        target_fp = max(MIN_KEPT_PAGES, int(round(orig_fp * cap.target_frac)))
        if target_fp >= cur_full_pages:
            if commit:
                kept_chunks.append(torch.arange(s0, s1, dtype=torch.long, device=device))
                new_qlen.append(cur_len)
                new_full_pages.append(cur_full_pages)
                drop_pages.append(orig_fp - cur_full_pages)
            continue

        obs_w = OBS_WINDOW if cur_len >= OBS_WINDOW else 128
        q_obs = q[s1 - obs_w : s1].transpose(0, 1).contiguous()  # [hq, obs_w, d]
        k_seq = k[s0:s1].transpose(0, 1)  # [hkv, cur_len, d]
        head_tok_scores = torch.zeros((hq, cur_len), dtype=torch.float32, device=device)
        _obs_fused_score_kernel[(hq, obs_w // 16)](
            q_obs, k_seq, head_tok_scores,
            cur_len,
            q_obs.stride(0), q_obs.stride(1), q_obs.stride(2),
            k_seq.stride(0), k_seq.stride(1), k_seq.stride(2),
            head_tok_scores.stride(0), head_tok_scores.stride(1),
            scale, QPK=qpk, OBS_WIN=obs_w, BM=16, BN=64, D=128, num_warps=4, num_stages=2,
        )
        _once("obs_smart", "E476 smart page pruning active (obs=%d, norm_head_sum, multi_layer_vote, neighbor_smooth)", obs_w)

        # Per-head page scores [hq, cur_full_pages], normalized per head on middle region so punctuation heads don't dominate!
        tail_full_start = cur_full_pages - TAIL_PAGES
        h_pg = head_tok_scores[:, : cur_full_pages * PAGE].view(hq, cur_full_pages, PAGE).amax(dim=2)
        h_mid_max = h_pg[:, SINK_PAGES:tail_full_start].amax(dim=1, keepdim=True).clamp(min=1e-6)
        pg_sc = (h_pg / h_mid_max).sum(dim=0)  # [cur_full_pages]

        if cap.accum_scores[i] is None:
            cap.accum_scores[i] = pg_sc
        else:
            cap.accum_scores[i] = cap.accum_scores[i] + pg_sc

        if not commit:
            continue

        full_scores = cap.accum_scores[i]
        # 1D neighbor smoothing [0.5, 1.0, 0.5] on middle pages to preserve multi-page facts
        k_mid = target_fp - SINK_PAGES - TAIL_PAGES
        assert tail_full_start > SINK_PAGES and k_mid > 0
        mid_raw = full_scores[SINK_PAGES:tail_full_start]
        if mid_raw.numel() >= 3:
            mid_scores = mid_raw.clone()
            mid_scores[1:] += 0.5 * mid_raw[:-1]
            mid_scores[:-1] += 0.5 * mid_raw[1:]
        else:
            mid_scores = mid_raw
        _, top_rel = torch.topk(mid_scores, k=k_mid, largest=True, sorted=False)
        top_mid = (top_rel + SINK_PAGES).sort().values

        kept_pages = torch.cat(
            [
                torch.arange(0, SINK_PAGES, device=device),
                top_mid,
                torch.arange(tail_full_start, cur_full_pages, device=device),
            ]
        )
        offsets = torch.arange(PAGE, device=device)
        tok_idx = (kept_pages[:, None] * PAGE + offsets[None, :]).reshape(-1) + s0
        if rem > 0:
            rem_idx = torch.arange(s1 - rem, s1, dtype=torch.long, device=device)
            tok_idx = torch.cat([tok_idx, rem_idx], dim=0)

        kept_chunks.append(tok_idx)
        new_qlen.append(int(tok_idx.numel()))
        new_full_pages.append(int(kept_pages.numel()))
        drop_pages.append(orig_fp - int(kept_pages.numel()))

    if commit:
        cap.kept_token_idx = torch.cat(kept_chunks, dim=0)
        cap.new_qlen = new_qlen
        cap.new_full_pages = new_full_pages
        cap.drop_pages = drop_pages


def build_stage_slot_mapping(
    block_table_gpu: torch.Tensor,
    req_indices: list[int],
    new_full_pages: list[int],
    orig_full_pages: list[int],
    orig_rem: list[int],
) -> torch.Tensor:
    device = block_table_gpu.device
    offsets = torch.arange(PAGE, dtype=torch.int64, device=device)
    chunks: list[torch.Tensor] = []
    for req_idx, kf, orig_fp, rem in zip(
        req_indices, new_full_pages, orig_full_pages, orig_rem
    ):
        blks = block_table_gpu[req_idx, :kf].to(torch.int64)
        slots = (blks[:, None] * PAGE + offsets[None, :]).reshape(-1)
        chunks.append(slots)
        if rem > 0:
            last_blk = block_table_gpu[req_idx, orig_fp].to(torch.int64)
            chunks.append(last_blk * PAGE + torch.arange(rem, dtype=torch.int64, device=device))
    return torch.cat(chunks, dim=0)
