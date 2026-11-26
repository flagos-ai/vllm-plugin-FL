"""Cross-layer bounded Top-K sparse decode state for MetaX C500 INT8 KV.

Budget: MIN_NPG=128, k=min(cdiv(npg, 4), 128), 4 sink + top-K mid + 8 hot pages.
  Protected always-dense layers: 0, 6, 41
  Layer 28: always-dense anchor for bank 3 (Stage 3 = 9% pages)
  Banks:
    0: anchor L4  -> recipients L5, L7..L10
    1: anchor L11 -> recipients L12..L19
    2: anchor L20 -> recipients L21..L27
    3: anchor L28 -> recipients L29..L40
    4: anchor L1  (self-anchor)
    5: anchor L2  (self-anchor)
    6: anchor L3  (self-anchor)
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

NUM_BANKS = 7
SINK_PAGES = 4
HOT_PAGES = 8
K_MAX = 128
MAX_SEL_PAGES = 144
MIN_NPG = 128
MAX_PAGES = 2048    # up to 32,768 tokens; longer rows fall back to dense
EPOCH_STRIDE = 16
DENOM = 4

PROTECTED_DENSE = {0, 6, 41}
ANCHOR_BANK = {
    4: 0,
    11: 1,
    20: 2,
    28: 3,
    1: 4,
    2: 5,
    3: 6,
}

_MARKED = False


def layer_role(li: Optional[int]) -> tuple[int, bool, bool]:
    """Returns (bank_id, is_anchor, always_dense). bank_id == -1 means plain dense."""
    if li is None or li in PROTECTED_DENSE:
        return -1, False, True
    if li in ANCHOR_BANK:
        return ANCHOR_BANK[li], True, (li == 28)
    if 5 <= li <= 10:
        return 0, False, False
    if 12 <= li <= 19:
        return 1, False, False
    if 21 <= li <= 27:
        return 2, False, False
    if 29 <= li <= 40:
        return 3, False, False
    return -1, False, True


@triton.jit
def _prepare_anchor_kernel(
    CU_Q,
    SEQ_LENS,
    TABLE,
    ROW_SLOT,
    CACHED_EPOCH,
    SP_TABLE,
    SP_K_COLD,
    BANK_SP_SEQ,
    NEED_REFRESH,
    nseq,
    s_tab,
    s_sp_slot,
    bank_id,
    max_slots,
    ALWAYS_DENSE: tl.constexpr,
    MIN_NPG_C: tl.constexpr,
    MAX_PAGES_C: tl.constexpr,
    HOT_PAGES_C: tl.constexpr,
    EPOCH_STRIDE_C: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= nseq:
        return
    qrow = tl.load(CU_Q + row)
    if tl.load(CU_Q + row + 1) - qrow != 1:
        tl.store(BANK_SP_SEQ + row, 0)
        tl.store(NEED_REFRESH + row, 0)
        return
    slot = tl.load(ROW_SLOT + row)
    L = tl.load(SEQ_LENS + row)
    npg = tl.cdiv(L, 16)
    invalid = ((slot < 0) | (slot >= max_slots)) | ((npg < MIN_NPG_C) | (npg > MAX_PAGES_C))
    if invalid:
        tl.store(BANK_SP_SEQ + row, 0)
        tl.store(NEED_REFRESH + row, 0)
        return

    cur_ep = (L - 1) // EPOCH_STRIDE_C
    ep_ptr = CACHED_EPOCH + bank_id * max_slots + slot
    old_ep = tl.load(ep_ptr)
    if old_ep != cur_ep:
        tl.store(BANK_SP_SEQ + row, 0)
        tl.store(NEED_REFRESH + row, 1)
        return

    # Cache hit: update the HOT_PAGES tail block IDs in sp_table so newly appended tokens are visible
    k_cold = tl.load(SP_K_COLD + bank_id * max_slots + slot)
    hi_offs = tl.arange(0, HOT_PAGES_C)
    src_pg = npg - HOT_PAGES_C + hi_offs
    blks = tl.load(TABLE + row.to(tl.int64) * s_tab + src_pg)
    sp_base = (bank_id * max_slots + slot).to(tl.int64) * s_sp_slot
    tl.store(SP_TABLE + sp_base + k_cold + hi_offs, blks)
    rem = L - (npg - 1) * 16
    sp_len = (k_cold + HOT_PAGES_C - 1) * 16 + rem
    # If ALWAYS_DENSE (L28), L28 itself runs dense, but BANK_SP_SEQ is set for recipients!
    # Wait: _decode_kernel_i8 checks SP_SEQ; for ALWAYS_DENSE we pass SP_SEQ=None to L28,
    # so writing sp_len to BANK_SP_SEQ here is immediately ready for L29..40!
    tl.store(BANK_SP_SEQ + row, sp_len)
    tl.store(NEED_REFRESH + row, 0)


@triton.jit
def _commit_anchor_kernel(
    SEQ_LENS,
    TABLE,
    ROW_SLOT,
    PAGE_SCORES,
    CACHED_EPOCH,
    SP_TABLE,
    SP_K_COLD,
    BANK_SP_SEQ,
    NEED_REFRESH,
    nseq,
    s_tab,
    s_sp_slot,
    bank_id,
    max_slots,
    MAX_PAGES_C: tl.constexpr,
    SINK_PAGES_C: tl.constexpr,
    HOT_PAGES_C: tl.constexpr,
    K_MAX_C: tl.constexpr,
    DENOM_C: tl.constexpr,
    EPOCH_STRIDE_C: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= nseq:
        return
    if tl.load(NEED_REFRESH + row) == 0:
        return
    slot = tl.load(ROW_SLOT + row)
    L = tl.load(SEQ_LENS + row)
    npg = tl.cdiv(L, 16)
    mid_end = npg - HOT_PAGES_C
    k_mid = tl.minimum(tl.cdiv(npg, DENOM_C), K_MAX_C)
    k_mid = tl.minimum(k_mid, mid_end - SINK_PAGES_C)
    k_mid = tl.maximum(k_mid, 1)

    offs = tl.arange(0, MAX_PAGES_C)
    mid_mask = (offs >= SINK_PAGES_C) & (offs < mid_end)
    ps_base = row.to(tl.int64) * 2 * MAX_PAGES_C
    s0 = tl.load(PAGE_SCORES + ps_base + offs, mask=mid_mask, other=float("-inf"))
    s1 = tl.load(PAGE_SCORES + ps_base + MAX_PAGES_C + offs, mask=mid_mask, other=float("-inf"))
    sc = tl.maximum(s0, s1)
    sc = tl.where(mid_mask, sc, float("-inf"))

    sp_base = (bank_id * max_slots + slot).to(tl.int64) * s_sp_slot
    tab_base = row.to(tl.int64) * s_tab

    # Store sink pages [0..SINK_PAGES_C-1]
    sink_i = tl.arange(0, SINK_PAGES_C)
    sink_blks = tl.load(TABLE + tab_base + sink_i)
    tl.store(SP_TABLE + sp_base + sink_i, sink_blks)

    # Iteratively pick top k_mid middle pages (k_mid <= 48)
    for t in range(K_MAX_C):
        if t < k_mid:
            best_idx = tl.argmax(sc, axis=0)
            blk = tl.load(TABLE + tab_base + best_idx)
            tl.store(SP_TABLE + sp_base + SINK_PAGES_C + t, blk)
            sc = tl.where(offs == best_idx, float("-inf"), sc)

    # Store hot pages [mid_end .. npg-1]
    k_cold = SINK_PAGES_C + k_mid
    hot_i = tl.arange(0, HOT_PAGES_C)
    hot_blks = tl.load(TABLE + tab_base + mid_end + hot_i)
    tl.store(SP_TABLE + sp_base + k_cold + hot_i, hot_blks)

    rem = L - (npg - 1) * 16
    sp_len = (k_cold + HOT_PAGES_C - 1) * 16 + rem
    tl.store(SP_K_COLD + bank_id * max_slots + slot, k_cold)
    tl.store(BANK_SP_SEQ + row, sp_len)
    tl.store(CACHED_EPOCH + bank_id * max_slots + slot, (L - 1) // EPOCH_STRIDE_C)


class SparseDecodeState:
    """Static GPU buffers for cross-layer bounded Top-K sparse decode (CUDA Graph safe)."""

    def __init__(self, device: torch.device, max_reqs: int = 512, max_slots: int = 512):
        self.device = device
        self.max_reqs = int(max_reqs)
        self.max_slots = int(max_slots)
        self.row_slot = torch.full((self.max_reqs,), -1, dtype=torch.int32, device=device)
        self.row_slot_cpu = torch.full((self.max_reqs,), -1, dtype=torch.int32, pin_memory=True)
        self._row_slot_np = self.row_slot_cpu.numpy()
        self._last_slots_tuple = None
        self._last_n = 0
        self.cached_epoch = torch.full((NUM_BANKS, self.max_slots), -1, dtype=torch.int32, device=device)
        self.sp_table = torch.zeros((NUM_BANKS, self.max_slots, MAX_SEL_PAGES), dtype=torch.int32, device=device)
        self.sp_k_cold = torch.zeros((NUM_BANKS, self.max_slots), dtype=torch.int32, device=device)
        self.bank_sp_seq = torch.zeros((NUM_BANKS, self.max_reqs), dtype=torch.int32, device=device)
        self.need_refresh = torch.zeros((self.max_reqs,), dtype=torch.int32, device=device)
        self.page_scores = torch.empty((self.max_reqs, 2, MAX_PAGES), dtype=torch.float32, device=device)

    def set_rows(self, slots: list[int], reset_slots: list[int]) -> None:
        if reset_slots:
            for s in reset_slots:
                if 0 <= s < self.max_slots:
                    self.cached_epoch[:, s].fill_(-1)
        st = tuple(slots)
        if not reset_slots and st == self._last_slots_tuple:
            return
        n = min(len(slots), self.max_reqs)
        if n > 0:
            self._row_slot_np[:n] = slots[:n]
            self.row_slot[:n].copy_(self.row_slot_cpu[:n], non_blocking=True)
        if n < self._last_n:
            self.row_slot[n:self._last_n].fill_(-1)
        self._last_slots_tuple = st
        self._last_n = n

    def set_dummy_rows(self, n: int) -> None:
        self._last_slots_tuple = None
        self._last_n = max(self._last_n, min(n, self.max_reqs))
        self.row_slot[: min(n, self.max_reqs)].fill_(-1)

    def prepare_anchor(self, bank_id: int, always_dense: bool, cu_q, seq_lens, table) -> None:
        global _MARKED
        if not _MARKED:
            _MARKED = True
            logger.warning(
                "E469 sparse decode active (npg>=%d, denom=%d, k_max=%d, hot=%d, banks=%d)",
                MIN_NPG, DENOM, K_MAX, HOT_PAGES, NUM_BANKS,
            )
        nseq = seq_lens.shape[0]
        _prepare_anchor_kernel[(nseq,)](
            cu_q, seq_lens, table, self.row_slot, self.cached_epoch,
            self.sp_table, self.sp_k_cold, self.bank_sp_seq[bank_id], self.need_refresh,
            nseq, table.stride(0), MAX_SEL_PAGES, bank_id, self.max_slots,
            ALWAYS_DENSE=always_dense,
            MIN_NPG_C=MIN_NPG, MAX_PAGES_C=MAX_PAGES, HOT_PAGES_C=HOT_PAGES,
            EPOCH_STRIDE_C=EPOCH_STRIDE, num_warps=4,
        )

    def commit_anchor(self, bank_id: int, seq_lens, table) -> None:
        nseq = seq_lens.shape[0]
        _commit_anchor_kernel[(nseq,)](
            seq_lens, table, self.row_slot, self.page_scores, self.cached_epoch,
            self.sp_table, self.sp_k_cold, self.bank_sp_seq[bank_id], self.need_refresh,
            nseq, table.stride(0), MAX_SEL_PAGES, bank_id, self.max_slots,
            MAX_PAGES_C=MAX_PAGES, SINK_PAGES_C=SINK_PAGES, HOT_PAGES_C=HOT_PAGES,
            K_MAX_C=K_MAX, DENOM_C=DENOM, EPOCH_STRIDE_C=EPOCH_STRIDE, num_warps=4,
        )


_STATE: Optional[SparseDecodeState] = None


def get_state(device: Optional[torch.device] = None) -> Optional[SparseDecodeState]:
    global _STATE
    if _STATE is None and device is not None:
        _STATE = SparseDecodeState(device)
    return _STATE
