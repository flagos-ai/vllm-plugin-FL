# SPDX-License-Identifier: Apache-2.0
"""Vendor-neutral 12-in-1 Fused Step-Preparation, 3-Tier BlockTable Delta Sync,
POD Sub-Batch Horizon Introspection, and 4-Slot Pinned AsyncOutput Ring.

Eliminates CPU/PCIe step-preparation bottlenecks (`T_CPU > T_GPU` launch cliffs)
on both `GPUModelRunner` and `ModelRunnerFL` by:
1. Tracking dirty `BlockTable` rows and single-block decode deltas (`BlockTableSyncState`)
   so steady-state decode steps avoid copying full `[max_reqs, max_cols]` tables.
2. Fusing 12 per-step tensor updates (`num_computed_tokens`, single-block `block_table`
   delta writes, `seq_lens`, `logits_indices`, GPU-resident `prev_sampled_token_ids -> input_ids`
   chaining, `positions`, paged `slot_mapping`, and CUDA-Graph tail zero/pad clearing)
   into a single `_fused_step_prep_and_slot_mapping_kernel` dispatched via
   `launch_compiled_kernel`.
3. Computing Prefill-Over-Decode (`POD`) sub-batch horizons (`num_prefills`,
   `max_prefill_seq_len`, `max_prefill_q_len`, `num_decode_seqs`, `max_decode_seq_len`,
   `mean_decode_seq_len`, and O(1) single-prefill index `prefill_seq_idx = int(pmask.argmax())`)
   in host NumPy memory without GPU synchronizations.
4. Replacing per-step D2H pinned buffer and `torch.Event` allocations in
   `AsyncGPUModelRunnerOutput` with a 4-slot pre-allocated pinned ring buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any

import numpy as np
import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID
from vllm.v1.kv_cache_interface import CrossAttentionSpec, EncoderOnlyAttentionSpec, UniformTypeKVCacheSpecs

from flag_gems.utils.fast_launcher import int_spec, launch_compiled_kernel


@dataclass(frozen=True)
class StepPrepConfig:
    """Static runtime configuration for `vllm_fl.worker.step_prep`.

    Attributes:
        enable_pod_horizon: Populate POD sub-batch horizon metadata (`num_prefills`,
            `prefill_seq_idx`, `max_prefill_seq_len`, `max_decode_seq_len`,
            `mean_decode_seq_len`, etc.) onto attention metadata for Mixed Chunked-Prefill kernels.
        decode_pad_buckets: Optional ascending request-count buckets used to pad
            uniform decode steps; defaults to wave-aligned power/multiple-of-8/16
            buckets when `None`.
    """

    enable_pod_horizon: bool = True
    decode_pad_buckets: tuple[int, ...] | None = None


_ACTIVE_CFG = StepPrepConfig()
_STEP_META = {1: {"num_warps": 1}, 4: {"num_warps": 4}}
# Per-device async D2H state `{device_idx: [ring_idx, pinned_bufs, events]}` capped to <= 16 devices.
_ASYNC_D2H_STATE: dict[int, list[Any]] = {}


class BlockTableSyncState:
    """Inductive dirty-row, single-block delta, and column HWM tracker for `BlockTable`.

    Categorizes per-step `BlockTable` H2D synchronization into three tiers:
    - Tier 0 (Clean): `dirty_rows` is empty -> zero H2D copies.
    - Tier 1 (Single-block delta): Every dirty row appended exactly 1 block ->
      packs `(num_computed_tokens, delta_col, delta_bid)` into a 4-slot pinned
      ring buffer and applies the update inside `_fused_step_prep_and_slot_mapping_kernel`.
    - Tier 2 (Bulk row/slice sync): Multi-block prefill or row move/swap -> copies
      only `:c_hwm` columns for sparse dirty rows or `[r_lo : r_hi + 1]` for dense spans.
    """

    __slots__ = ("max_rows", "max_cols", "raw_max_col", "c_hwm", "dirty_rows", "delta_map", "stale_cpu_rows", "in_add_row")

    def __init__(self, bt: Any) -> None:
        """Scan `bt` to seed initial column HWM (`c_hwm`) and dirty/stale row sets."""
        np_arr = bt.block_table.np
        self.max_rows, self.max_cols = int(np_arr.shape[0]), int(np_arr.shape[1])
        self.dirty_rows, self.delta_map, self.stale_cpu_rows, self.in_add_row = set(), {}, set(), False
        max_n = int(bt.num_blocks_per_row.max()) if self.max_rows > 0 else 0
        if max_n > 0 or bool(np.any(np_arr != 0)):
            nz_cols = np.flatnonzero(np.any(np_arr != 0, axis=0))
            self.raw_max_col = max(max_n, int(nz_cols[-1]) + 1 if nz_cols.size > 0 else 0)
            # Round HWM up to a 64-column boundary to amortize HWM growth and align H2D transfers.
            self.c_hwm = min(self.max_cols, ((self.raw_max_col + 63) // 64) * 64)
            nz_rows = np.flatnonzero(
                (bt.num_blocks_per_row > 0) | np.any(np_arr[:, : max(1, self.c_hwm)] != 0, axis=1)
            ).tolist()
            self.dirty_rows.update(int(r) for r in nz_rows)
            self.stale_cpu_rows.update(int(r) for r in nz_rows)
        else:
            self.raw_max_col = self.c_hwm = 0

    def update_hwm(self, n_cols: int) -> None:
        """Advance the tracked column high-water mark (64-column aligned) if `n_cols` exceeds it."""
        if n_cols > self.raw_max_col:
            self.raw_max_col, self.c_hwm = n_cols, min(self.max_cols, ((n_cols + 63) // 64) * 64)

    def reset_all(self) -> None:
        """Reset all dirty, delta, stale-row, and column HWM tracking state on table clear."""
        self.raw_max_col = self.c_hwm = 0
        self.dirty_rows.clear()
        self.delta_map.clear()
        self.stale_cpu_rows.clear()
        self.in_add_row = False

    def record_gpu_zero_pad(self, bt: Any, row_start: int, row_end: int) -> None:
        """Reconcile dirty state after GPU zero-padding rows `[row_start, row_end)` up to `c_hwm`."""
        if row_start >= row_end or self.c_hwm <= 0:
            return
        nb = bt.num_blocks_per_row
        for r in range(row_start, row_end):
            # Empty rows without stale CPU remnants now match GPU zeros; active/stale rows need resync.
            if int(nb[r]) == 0 and r not in self.stale_cpu_rows:
                self.dirty_rows.discard(r)
            else:
                self.dirty_rows.add(r)
            self.delta_map.pop(r, None)


def _bt_state(bt: Any) -> BlockTableSyncState:
    """Lazily attach and return the `BlockTableSyncState` tracker for `bt`."""
    if (st := getattr(bt, "_fl_bt_sync_state", None)) is None:
        st = BlockTableSyncState(bt)
        bt._fl_bt_sync_state = st
    return st


def _sync_bt_tier02(bt: Any, num_reqs: int) -> int:
    """Synchronize dirty `BlockTable` rows via Tier-0 (no-op) or Tier-2 (sparse/span H2D copy)."""
    st = _bt_state(bt)
    if not (dirty := sorted(r for r in st.dirty_rows if r < num_reqs)):
        return st.c_hwm
    c_hwm = st.c_hwm
    if c_hwm > 0:
        r_lo, r_hi = dirty[0], dirty[-1]
        # Copy sparse rows individually (clamped to `:c_hwm`) when a contiguous span would transfer >2x more data.
        if len(dirty) == 1 or (len(dirty) <= 3 and (r_hi - r_lo + 1) * st.max_cols > len(dirty) * c_hwm * 2):
            rows = dirty
            for r in dirty:
                bt.block_table.gpu[r, :c_hwm].copy_(bt.block_table.cpu[r, :c_hwm], non_blocking=True)
        else:
            bt.block_table.gpu[r_lo : r_hi + 1].copy_(bt.block_table.cpu[r_lo : r_hi + 1], non_blocking=True)
            rows = range(r_lo, r_hi + 1)
    else:
        rows = dirty
    for r in rows:
        st.dirty_rows.discard(r)
        st.delta_map.pop(r, None)
    return c_hwm


def _sync_primary_bt(
    bt0: Any, num_reqs: int, num_comp_np: np.ndarray,
    cpu_ring: torch.Tensor, np_ring: np.ndarray, gpu_ring: torch.Tensor, slot: int,
) -> tuple[bool, torch.Tensor, int]:
    """Synchronize primary `BlockTable` via Tier-1 pinned delta ring when all dirty rows are 1-block appends."""
    st = _bt_state(bt0)
    if not (dirty := [r for r in st.dirty_rows if r < num_reqs]):
        return False, gpu_ring[slot], st.c_hwm
    if all(r in st.delta_map for r in dirty):
        np_s = np_ring[slot]
        np_s[:num_reqs, 0], np_s[:num_reqs, 1], np_s[:num_reqs, 2] = num_comp_np[:num_reqs], -1, 0
        for r in dirty:
            np_s[r, 1], np_s[r, 2] = st.delta_map.pop(r)
            st.dirty_rows.discard(r)
        gpu_s = gpu_ring[slot, :num_reqs]
        gpu_s.copy_(cpu_ring[slot, :num_reqs], non_blocking=True)
        return True, gpu_s, st.c_hwm
    return False, gpu_ring[slot], _sync_bt_tier02(bt0, num_reqs)


def _patch_block_table_cls() -> None:
    """Idempotently instrument `vllm.v1.worker.block_table.BlockTable` mutations for 3-tier delta tracking."""
    import vllm.v1.worker.block_table as bt_mod

    if (bt_cls := getattr(bt_mod, "BlockTable", None)) is None or getattr(bt_cls, "_fl_bt_sync_patched", False):
        return
    o_app, o_add, o_clr_r = bt_cls.append_row, bt_cls.add_row, bt_cls.clear_row
    o_mov, o_swp, o_com, o_clr = bt_cls.move_row, bt_cls.swap_row, bt_cls.commit_block_table, bt_cls.clear

    def append_row(self, block_ids: list[int], row_idx: int) -> None:
        """Track single-block decode appends as Tier-1 deltas; multi-block appends fall back to Tier-2."""
        st, prev_n = _bt_state(self), int(self.num_blocks_per_row[row_idx])
        o_app(self, block_ids, row_idx)
        new_n = int(self.num_blocks_per_row[row_idx])
        st.update_hwm(new_n)
        # Skip nested `append_row` invoked internally by `add_row`.
        if st.in_add_row:
            return
        if new_n - prev_n == 1 and row_idx not in st.dirty_rows:
            st.dirty_rows.add(row_idx)
            st.delta_map[row_idx] = (prev_n, int(self.block_table.np[row_idx, prev_n]))
        elif new_n > prev_n:
            st.dirty_rows.add(row_idx)
            st.delta_map.pop(row_idx, None)

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        """Mark newly added rows dirty (Tier-2) and record `stale_cpu_rows` if row length shrank."""
        st = _bt_state(self)
        prev_n, st.in_add_row = int(self.num_blocks_per_row[row_idx]), True
        try:
            o_add(self, block_ids, row_idx)
        finally:
            st.in_add_row = False
        new_n = int(self.num_blocks_per_row[row_idx])
        st.update_hwm(new_n)
        if prev_n > new_n:
            st.stale_cpu_rows.add(row_idx)
        st.dirty_rows.add(row_idx)
        st.delta_map.pop(row_idx, None)

    def clear_row(self, row_idx: int) -> None:
        """Mark cleared row dirty and invalidate any pending Tier-1 delta."""
        st = _bt_state(self)
        o_clr_r(self, row_idx)
        st.dirty_rows.add(row_idx)
        st.delta_map.pop(row_idx, None)

    def move_row(self, src: int, tgt: int) -> None:
        """Propagate dirty/stale state when relocating a request row from `src` to `tgt`."""
        st, prev_tgt = _bt_state(self), int(self.num_blocks_per_row[tgt])
        o_mov(self, src, tgt)
        new_tgt = int(self.num_blocks_per_row[tgt])
        st.update_hwm(new_tgt)
        if int(self.num_blocks_per_row[src]) > 0:
            st.stale_cpu_rows.add(src)
        if prev_tgt > new_tgt:
            st.stale_cpu_rows.add(tgt)
        st.dirty_rows.add(tgt)
        st.delta_map.pop(tgt, None)

    def swap_row(self, r1: int, r2: int) -> None:
        """Swap `stale_cpu_rows` membership and mark both rows dirty for Tier-2 sync."""
        st = _bt_state(self)
        o_swp(self, r1, r2)
        if (s1 := r1 in st.stale_cpu_rows) != (r2 in st.stale_cpu_rows):
            st.stale_cpu_rows.discard(r1 if s1 else r2)
            st.stale_cpu_rows.add(r2 if s1 else r1)
        st.dirty_rows.update((r1, r2))
        st.delta_map.pop(r1, None)
        st.delta_map.pop(r2, None)

    def commit_block_table(self, num_reqs: int) -> None:
        """Clear dirty/delta markers for `[0, num_reqs)` after an explicit full H2D commit."""
        st = _bt_state(self)
        o_com(self, num_reqs)
        for r in range(num_reqs):
            st.dirty_rows.discard(r)
            st.delta_map.pop(r, None)

    def clear(self) -> None:
        """Reset sync state alongside full `BlockTable.clear()`."""
        o_clr(self)
        if (st := getattr(self, "_fl_bt_sync_state", None)) is not None:
            st.reset_all()

    bt_cls.append_row, bt_cls.add_row, bt_cls.clear_row = append_row, add_row, clear_row
    bt_cls.move_row, bt_cls.swap_row, bt_cls.commit_block_table, bt_cls.clear = move_row, swap_row, commit_block_table, clear
    bt_cls._fl_bt_sync_patched = True


# 12-in-1 fused step-preparation Triton kernel:
# - Programs `0 .. num_reqs - 1` handle one active request each: optionally commit
#   Tier-1 `(num_computed_tokens, delta_col, delta_bid)` updates, write `seq_lens`
#   and `logits_indices`, chain `prev_sampled_token_ids -> input_ids` directly in
#   device memory when `req_idx < num_common_tokens`, and populate `positions` +
#   paged `slot_mapping` (using bit-shifts when `block_size == 16` and `CP == 1`).
# - Optional tail program `req_idx == num_reqs` (`HAS_PAD_CTA=True`) zeroes padded
#   `seq_lens`, `positions`, and `block_table` rows up to `bt_clear_cols` while
#   filling padded `slot_mapping` entries with `PAD_ID` (`-1`) for CUDA Graph safety.
@triton.jit
def _fused_step_prep_and_slot_mapping_kernel(
    num_reqs, max_seq_pad, num_tokens, max_num_tokens, num_common_tokens, pad_reqs, bt_clear_cols,
    query_start_loc_ptr, num_computed_tokens_ptr, step_inputs_ptr, positions_ptr,
    seq_lens_ptr, logits_indices_ptr, input_ids_ptr, prev_sampled_token_ids_ptr, prev_sampled_stride,
    block_table_ptr, block_table_stride, block_size, slot_mapping_ptr,
    TOTAL_CP_WORLD_SIZE: tl.constexpr, TOTAL_CP_RANK: tl.constexpr, CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    PAD_ID: tl.constexpr, BLOCK_SIZE: tl.constexpr, HAS_PAD_CTA: tl.constexpr, HAS_BT_DELTA: tl.constexpr,
):
    """Fused 12-in-1 step-prep, Tier-1 block-table delta commit, slot-mapping, and pad-clearing kernel."""
    req_idx = tl.program_id(0)
    # Tail-padding CTA: clear padded request/token/block-table regions in parallel.
    if HAS_PAD_CTA and req_idx == num_reqs:
        for i in range(num_reqs, max_seq_pad, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            tl.store(seq_lens_ptr + offs, 0, mask=offs < max_seq_pad)
        pad64, zero64 = tl.cast(PAD_ID, tl.int64), tl.cast(0, tl.int64)
        for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):
            offs = i + tl.arange(0, BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offs, pad64, mask=offs < max_num_tokens)
            tl.store(positions_ptr + offs, zero64, mask=offs < max_num_tokens)
        for r in range(num_reqs, pad_reqs):
            r_base = r.to(tl.int64) * block_table_stride
            for c in range(0, bt_clear_cols, BLOCK_SIZE):
                cols = c + tl.arange(0, BLOCK_SIZE)
                tl.store(block_table_ptr + r_base + cols, 0, mask=cols < bt_clear_cols)
        return

    start_idx, end_idx = tl.load(query_start_loc_ptr + req_idx), tl.load(query_start_loc_ptr + req_idx + 1)
    row_offset = req_idx.to(tl.int64) * block_table_stride
    # Tier-1 delta commit: unpack `(num_comp, delta_col, delta_bid)` from pinned ring copy.
    if HAS_BT_DELTA:
        inp = step_inputs_ptr + req_idx * 3
        num_comp, delta_col, delta_bid = tl.load(inp), tl.load(inp + 1), tl.load(inp + 2)
        tl.store(num_computed_tokens_ptr + req_idx, num_comp)
        if delta_col >= 0:
            tl.store(block_table_ptr + row_offset + delta_col.to(tl.int64), delta_bid)
    else:
        num_comp, delta_col, delta_bid = tl.load(num_computed_tokens_ptr + req_idx), -1, 0

    tl.store(seq_lens_ptr + req_idx, num_comp + (end_idx - start_idx))
    tl.store(logits_indices_ptr + req_idx, end_idx - 1)
    if req_idx < num_common_tokens:
        tl.store(input_ids_ptr + req_idx, tl.load(prev_sampled_token_ids_ptr + req_idx * prev_sampled_stride).to(tl.int32))

    # Compute token `positions` and physical KV `slot_mapping` across scheduled tokens.
    s64, e64, nc64, dc64 = start_idx.to(tl.int64), end_idx.to(tl.int64), num_comp.to(tl.int64), tl.cast(delta_col, tl.int64)
    vbs, lanes = block_size * TOTAL_CP_WORLD_SIZE, tl.arange(0, BLOCK_SIZE)
    for i in range(s64, e64, BLOCK_SIZE):
        offs = i + lanes
        mask, pos = offs < e64, nc64 + (offs - s64)
        tl.store(positions_ptr + offs, pos, mask=mask)
        b_idx = (pos >> 4) if (TOTAL_CP_WORLD_SIZE == 1 and block_size == 16) else (pos // vbs)
        b_num = tl.load(block_table_ptr + row_offset + b_idx, mask=mask, other=0)
        if HAS_BT_DELTA:
            b_num = tl.where(b_idx == dc64, delta_bid, b_num)
        if TOTAL_CP_WORLD_SIZE == 1 and block_size == 16:
            slot_ids = (b_num.to(tl.int64) << 4) + (pos & 15)
        elif TOTAL_CP_WORLD_SIZE == 1:
            slot_ids = b_num.to(tl.int64) * block_size + (pos - b_idx * vbs)
        else:
            vbo = pos - b_idx * vbs
            is_loc = ((vbo // CP_KV_CACHE_INTERLEAVE_SIZE) % TOTAL_CP_WORLD_SIZE) == TOTAL_CP_RANK
            lbo = (vbo // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)) * CP_KV_CACHE_INTERLEAVE_SIZE + (vbo % CP_KV_CACHE_INTERLEAVE_SIZE)
            slot_ids = tl.where(is_loc, b_num.to(tl.int64) * block_size + lbo, tl.cast(PAD_ID, tl.int64))
        tl.store(slot_mapping_ptr + offs, slot_ids, mask=mask)


def _pad_decode_reqs(n: int, cap: int, buckets: tuple[int, ...] | None) -> int:
    """Round uniform-decode request count `n` up to the next bucket (clamped to `cap`)."""
    if buckets:
        for b in buckets:
            if b >= n:
                return min(b, cap)
        return min(n, cap)
    return min(
        n if n <= 2 else (4 if n <= 4 else (((n + 7) // 8) * 8 if n <= 256 else ((n + 15) // 16) * 16)),
        cap,
    )


def _make_runner_patches(orig_prep, orig_slots, orig_attn_meta, orig_pre):
    """Build fused `_prepare_inputs`, `_get_slot_mappings`, `_build_attention_metadata`, and `_preprocess` hooks."""

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens: np.ndarray):
        """Fused step-prep replacing multi-kernel CPU/GPU metadata setup and block-table H2D copies."""
        if (
            self.use_async_spec_decode or self.uses_mrope or self.uses_xdrope_dim > 0
            or len(scheduler_output.scheduled_spec_decode_tokens) > 0
            or self.mamba_prev_last_scheduled_idx is not None or self.num_accepted_tokens_event is not None
            or self.enable_prompt_embeds or bool(self.input_batch.req_prompt_embeds)
            or self.lora_config is not None or len(self.input_batch.block_table.block_tables) == 0
        ):
            self._fl_step_prep_padded, self._fl_step_prep_padded_tokens = False, 0
            self._fl_blk_table_padded_upto, self._fl_pod_valid = 0, False
            return orig_prep(self, scheduler_output, num_scheduled_tokens)

        tot_sched, num_reqs = scheduler_output.total_num_scheduled_tokens, self.input_batch.num_reqs
        if (logits_buf := getattr(self, "_fl_logits_indices_gpu", None)) is None or logits_buf.shape[0] < self.max_num_reqs:
            _patch_block_table_cls()
            with torch.inference_mode(False):
                logits_buf = torch.empty(self.max_num_reqs, dtype=torch.int32, device=self.device)
                cpu_r = torch.zeros((4, self.max_num_reqs, 3), dtype=torch.int32, device="cpu", pin_memory=PIN_MEMORY)
                gpu_r = torch.zeros((4, self.max_num_reqs, 3), dtype=torch.int32, device=self.device)
                pref_c = torch.zeros(self.max_num_reqs, dtype=torch.bool, device="cpu")
            self._fl_logits_indices_gpu, self._fl_step_prep_cpu_ring = logits_buf, cpu_r
            self._fl_step_prep_np_ring, self._fl_step_prep_gpu_ring, self._fl_step_prep_ring_idx = cpu_r.numpy(), gpu_r, 0
            self._fl_optimistic_seq_lens_np = self.optimistic_seq_lens_cpu.numpy()
            self._fl_is_prefilling_cpu, self._fl_is_prefilling_np = pref_c, pref_c.numpy()
            self.num_accepted_tokens.np.fill(1)
            self.num_accepted_tokens.gpu.fill_(1)

        bts = self.input_batch.block_table.block_tables
        bt0, slot = bts[0], self._fl_step_prep_ring_idx & 3
        self._fl_step_prep_ring_idx = slot + 1
        has_bt_delta, step_inp_gpu, c_hwm = _sync_primary_bt(
            bt0, num_reqs, self.input_batch.num_computed_tokens_cpu,
            self._fl_step_prep_cpu_ring, self._fl_step_prep_np_ring, self._fl_step_prep_gpu_ring, slot,
        )
        if not has_bt_delta:
            self.num_computed_tokens[:num_reqs].copy_(self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs], non_blocking=True)
        for extra_bt in bts[1:]:
            _sync_bt_tier02(extra_bt, num_reqs)

        prev_sampled, prev_map = self.input_batch.prev_sampled_token_ids, self.input_batch.prev_req_id_to_index
        prev_pos, req_ids = self.prev_positions.np[:num_reqs], self.input_batch.req_ids
        if not prev_map:
            prev_pos.fill(-1)
        else:
            for i in range(num_reqs):
                prev_pos[i] = prev_map.get(req_ids[i], -1)

        is_uniform_decode = tot_sched == num_reqs
        cu_toks = self.arange_np[1 : num_reqs + 1] if is_uniform_decode else np.cumsum(num_scheduled_tokens, dtype=np.int32)
        flat_idx, prev_idx, match_all, max_flat = [], [], True, -1
        if prev_sampled is not None and prev_map:
            for cur_i in range(num_reqs):
                if (pi := int(prev_pos[cur_i])) >= 0:
                    fi = int(cu_toks[cur_i]) - 1
                    prev_idx.append(pi)
                    flat_idx.append(fi)
                    match_all, max_flat = (match_all and pi == fi), max(max_flat, fi)

        # When all continuing requests sit at contiguous indices `[0, n_common)` with
        # `prev_pos[i] == flat_idx[i]`, chain `prev_sampled_token_ids -> input_ids`
        # directly inside the fused GPU kernel; otherwise gather host token IDs and scatter.
        n_common = len(flat_idx)
        direct_copy = n_common > 0 and match_all and max_flat == (n_common - 1)
        if prev_sampled is None or n_common < tot_sched:
            req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
            cu_toks = self._get_cumsum_and_arange(num_scheduled_tokens, self.query_pos.np)
            tok_indices = (
                self.input_batch.num_computed_tokens_cpu[req_indices]
                + self.query_pos.np[: cu_toks[-1]]
                + req_indices * self.input_batch.token_ids_cpu.shape[1]
            )
            torch.index_select(
                self.input_batch.token_ids_cpu_tensor.flatten(),
                0,
                torch.from_numpy(tok_indices),
                out=self.input_ids.cpu[:tot_sched],
            )
            self.input_ids.copy_to_gpu(tot_sched)

        full_max_toks = min(bt0.max_num_batched_tokens, self.positions.shape[0])
        max_cg = getattr(getattr(self, "compilation_config", None), "max_cudagraph_capture_size", 512) or 512
        if is_uniform_decode and num_reqs <= max_cg:
            pad_reqs = _pad_decode_reqs(num_reqs, self.max_num_reqs, _ACTIVE_CFG.decode_pad_buckets)
            max_toks_pad, max_seq_pad, qsl_copy = min(max(pad_reqs, tot_sched), full_max_toks), pad_reqs, pad_reqs + 1
        elif tot_sched <= max_cg:
            pad_reqs, max_toks_pad, max_seq_pad = num_reqs, min(max_cg, full_max_toks), self.max_num_reqs
            qsl_copy = min(self.max_num_reqs, max(num_reqs, max_cg)) + 1
        else:
            pad_reqs, max_toks_pad, max_seq_pad, qsl_copy = num_reqs, tot_sched, num_reqs, num_reqs + 1
        qsl_copy = min(qsl_copy, self.query_start_loc.np.shape[0])

        self.query_start_loc.np[0], self.query_start_loc.np[1 : num_reqs + 1] = 0, cu_toks
        if qsl_copy > num_reqs + 1:
            self.query_start_loc.np[num_reqs + 1 : qsl_copy].fill(cu_toks[-1])
        self.query_start_loc.copy_to_gpu(qsl_copy)
        qsl_gpu = self.query_start_loc.gpu[: num_reqs + 1]

        opt_np = self._fl_optimistic_seq_lens_np
        np.add(self.input_batch.num_computed_tokens_cpu[:num_reqs], num_scheduled_tokens, out=opt_np[:num_reqs])
        if max_seq_pad > num_reqs:
            opt_np[num_reqs:max_seq_pad] = 0

        # POD (Prefill-Over-Decode) sub-batch horizon introspection: compute exact
        # prefill/decode counts and max sequence lengths on CPU views. When exactly
        # one request is prefilling (`n_pref == 1`, standard V1 chunked-prefill steady
        # state), `pmask.argmax()` locates its slot in O(1) C-loop time (`prefill_seq_idx`)
        # so Mixed-step Dual-SME attention kernels can dispatch Decode and Prefill
        # waves without scanning `query_start_loc` on GPU.
        seq_act = opt_np[:num_reqs]
        if _ACTIVE_CFG.enable_pod_horizon:
            if is_uniform_decode:
                self._fl_num_prefills = self._fl_max_prefill_seq_len = self._fl_max_prefill_q_len = 0
                self._fl_prefill_seq_idx = -1
                self._fl_num_decode_seqs = num_reqs
                self._fl_max_decode_seq_len = int(seq_act.max()) if num_reqs > 0 else 0
                self._fl_mean_decode_seq_len = int(seq_act.sum(dtype=np.int64)) // num_reqs if num_reqs > 0 else 0
            else:
                sched_act = num_scheduled_tokens[:num_reqs]
                pmask, dmask = sched_act > 1, sched_act == 1
                n_pref = int(pmask.sum())
                n_dec = int(dmask.sum()) if n_pref < num_reqs else 0
                self._fl_num_prefills, self._fl_num_decode_seqs = n_pref, n_dec
                self._fl_prefill_seq_idx = int(pmask.argmax()) if n_pref == 1 else -1
                self._fl_max_prefill_seq_len = int(seq_act[pmask].max()) if n_pref > 0 else 0
                self._fl_max_prefill_q_len = int(sched_act[pmask].max()) if n_pref > 0 else 0
                self._fl_max_decode_seq_len = int(seq_act[dmask].max()) if n_dec > 0 else 0
                self._fl_mean_decode_seq_len = int(seq_act[dmask].sum(dtype=np.int64)) // n_dec if n_dec > 0 else 0
            self._fl_pod_valid = True
        else:
            self._fl_pod_valid = False

        req_num_toks = np.fromiter(
            (self.requests[r].num_tokens for r in req_ids), dtype=np.int32, count=num_reqs
        )
        self.discard_request_mask.np[:num_reqs] = seq_act < req_num_toks
        if self.speculative_config is not None:
            self.discard_request_mask.copy_to_gpu(num_reqs)

        cp_ws = bt0.pcp_world_size * bt0.dcp_world_size
        cp_rk = bt0.pcp_rank * bt0.dcp_world_size + bt0.dcp_rank
        if direct_copy:
            prev_ptr, prev_s, n_comm_k = prev_sampled, prev_sampled.stride(0), n_common
        else:
            prev_ptr, prev_s, n_comm_k = self.input_ids.gpu, 1, 0
        logits_indices = logits_buf[:num_reqs]
        need_bt_pad = num_reqs < pad_reqs and c_hwm > 0
        need_pad = (num_reqs < max_seq_pad) or (tot_sched < max_toks_pad) or need_bt_pad
        if is_uniform_decode and not need_pad:
            bs_k, nw_k = 64, 1
        else:
            bs_k, nw_k = 256, 4
        gx = num_reqs + 1 if need_pad else num_reqs

        if gx > 0:
            bt_gpu = bt0.block_table.gpu
            bt_s, bs_v = int(bt_gpu.stride(0)), int(bt0.block_size)
            dev_idx = qsl_gpu.device.index or 0
            # Bitwise-OR all tensor data pointers so `(pm_bits & 0xF) == 0` verifies
            # 16-byte alignment required by `launch_compiled_kernel`'s fast dispatch cache.
            pm_bits = (
                qsl_gpu.data_ptr()
                | self.num_computed_tokens.data_ptr()
                | step_inp_gpu.data_ptr()
                | self.positions.data_ptr()
                | self.seq_lens.data_ptr()
                | logits_indices.data_ptr()
                | self.input_ids.gpu.data_ptr()
                | prev_ptr.data_ptr()
                | bt_gpu.data_ptr()
                | bt0.slot_mapping.gpu.data_ptr()
            )
            l_key = (
                "fl_step_prep", dev_idx, prev_ptr.dtype,
                int_spec(num_reqs), int_spec(max_seq_pad), int_spec(tot_sched), int_spec(max_toks_pad),
                int_spec(n_comm_k), int_spec(pad_reqs), int_spec(c_hwm), int_spec(prev_s),
                int_spec(bt_s), int_spec(bs_v), cp_ws, cp_rk, bt0.cp_kv_cache_interleave_size,
                PAD_SLOT_ID, bs_k, need_pad, has_bt_delta, nw_k,
            )
            rt_args = (
                int(num_reqs), int(max_seq_pad), int(tot_sched), int(max_toks_pad), int(n_comm_k),
                int(pad_reqs), int(c_hwm), qsl_gpu, self.num_computed_tokens, step_inp_gpu,
                self.positions, self.seq_lens, logits_indices, self.input_ids.gpu,
                prev_ptr, int(prev_s), bt_gpu, bt_s, bs_v, bt0.slot_mapping.gpu,
            )
            full_args = (
                *rt_args, cp_ws, cp_rk, bt0.cp_kv_cache_interleave_size,
                PAD_SLOT_ID, bs_k, need_pad, has_bt_delta,
            )
            if not launch_compiled_kernel(
                l_key, gx, dev_idx, (pm_bits & 0xF) == 0,
                _fused_step_prep_and_slot_mapping_kernel, rt_args, full_args,
                _STEP_META[nw_k], disallow_when_profiling=True,
            ):
                _fused_step_prep_and_slot_mapping_kernel[(gx,)](
                    *rt_args,
                    TOTAL_CP_WORLD_SIZE=cp_ws, TOTAL_CP_RANK=cp_rk,
                    CP_KV_CACHE_INTERLEAVE_SIZE=bt0.cp_kv_cache_interleave_size,
                    PAD_ID=PAD_SLOT_ID, BLOCK_SIZE=bs_k, HAS_PAD_CTA=need_pad,
                    HAS_BT_DELTA=has_bt_delta, num_warps=nw_k,
                )

        if need_bt_pad:
            _bt_state(bt0).record_gpu_zero_pad(bt0, num_reqs, pad_reqs)
        for bt in bts[1:]:
            bt.compute_slot_mapping(num_reqs, qsl_gpu, self.positions[:tot_sched])
        if n_common > 0 and not direct_copy:
            dst_i = torch.tensor(flat_idx, dtype=torch.int64, pin_memory=PIN_MEMORY).to(self.device, non_blocking=True)
            src_i = torch.tensor(prev_idx, dtype=torch.int64, pin_memory=PIN_MEMORY).to(self.device, non_blocking=True)
            self.input_ids.gpu.scatter_(dim=0, index=dst_i, src=prev_sampled[src_i, 0].to(self.input_ids.gpu.dtype))

        self._fl_step_prep_padded, self._fl_step_prep_padded_tokens, self._fl_blk_table_padded_upto = True, max_toks_pad, pad_reqs
        if is_uniform_decode and not self.cache_config.kv_sharing_fast_prefill and not getattr(self, "is_pooling_model", False) and self.speculative_config is None:
            return slice(0, num_reqs), None
        return logits_indices, None

    def _get_slot_mappings(self, num_tokens_padded: int, num_reqs_padded: int, num_tokens_unpadded: int, ubatch_slices=None):
        """Return pre-computed primary `slot_mapping` slice, topping up pad sentinel `-1` only if needed."""
        if (kv_cfg := getattr(self, "kv_cache_config", None)) is None or len(kv_cfg.kv_cache_groups) == 0:
            return None, None
        kv_groups = kv_cfg.kv_cache_groups
        if len(kv_groups) == 1 and ubatch_slices is None and not isinstance(kv_groups[0].kv_cache_spec, EncoderOnlyAttentionSpec):
            sm = self.input_batch.block_table[0].slot_mapping.gpu[:num_tokens_padded]
            if num_tokens_unpadded < num_tokens_padded:
                padded_upto = getattr(self, "_fl_step_prep_padded_tokens", 0)
                if not getattr(self, "_fl_step_prep_padded", False):
                    sm[num_tokens_unpadded:num_tokens_padded].fill_(-1)
                elif padded_upto < num_tokens_padded:
                    sm[padded_upto:num_tokens_padded].fill_(-1)
                    self.positions[padded_upto:num_tokens_padded].zero_()
            return {0: sm}, dict.fromkeys(kv_groups[0].layer_names, sm)
        return orig_slots(
            self,
            num_tokens_padded=num_tokens_padded,
            num_reqs_padded=num_reqs_padded,
            num_tokens_unpadded=num_tokens_unpadded,
            ubatch_slices=ubatch_slices,
        )

    def _patched_build_attn_meta(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices=None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
        slot_mappings: dict[int, torch.Tensor] | None = None,
        **extra_kwargs: Any,
    ):
        """Fast single-KV-group `CommonAttentionMetadata` builder attaching POD horizon stats."""
        padded_upto, self._fl_blk_table_padded_upto = getattr(self, "_fl_blk_table_padded_upto", 0), 0
        kv_groups = self.kv_cache_config.kv_cache_groups
        if (
            not extra_kwargs
            and len(kv_groups) == 1
            and not for_cudagraph_capture
            and ubatch_slices is None
            and not use_spec_decode
            and not self.routed_experts_initialized
            and not self.use_async_spec_decode
            and not self.is_mm_prefix_lm
            and self.dcp_world_size <= 1
            and not self.cache_config.kv_sharing_fast_prefill
            and self.speculative_config is None
            and len(self.attn_groups[0]) == 1
            and not isinstance(
                kv_groups[0].kv_cache_spec,
                (EncoderOnlyAttentionSpec, CrossAttentionSpec, UniformTypeKVCacheSpecs),
            )
        ):
            nt_pad, nr_pad = (num_tokens_padded or num_tokens), (num_reqs_padded or num_reqs)
            if (
                getattr(self, "_fl_step_prep_padded", False)
                and num_tokens < nt_pad
                and (pt := getattr(self, "_fl_step_prep_padded_tokens", 0)) < nt_pad
            ):
                self.positions[pt:nt_pad].zero_()
            if (opt_np := getattr(self, "_fl_optimistic_seq_lens_np", None)) is None:
                opt_np = self._fl_optimistic_seq_lens_np = self.optimistic_seq_lens_cpu.numpy()
            if num_reqs < nr_pad:
                opt_np[num_reqs:nr_pad] = 0
                self.seq_lens[max(num_reqs, padded_upto) : nr_pad].zero_()
            max_seq_len = int(opt_np[:num_reqs].max()) if num_reqs > 0 else 0

            bt0 = self.input_batch.block_table[0]
            blk_t0 = bt0.get_device_tensor(nr_pad)
            if num_reqs < nr_pad and padded_upto < nr_pad and (st := _bt_state(bt0)).c_hwm > 0:
                r_lo = max(num_reqs, padded_upto)
                blk_t0[r_lo:nr_pad, : st.c_hwm].fill_(NULL_BLOCK_ID)
                st.record_gpu_zero_pad(bt0, r_lo, nr_pad)

            qsl_need = nr_pad + 1
            if qsl_need > num_reqs + 1 and int(self.query_start_loc.np[qsl_need - 1]) != int(self.query_start_loc.np[num_reqs]):
                self.query_start_loc.np[num_reqs + 1 : qsl_need].fill(int(self.query_start_loc.np[num_reqs]))
                self.query_start_loc.copy_to_gpu(qsl_need)

            if (pref_cpu := getattr(self, "_fl_is_prefilling_cpu", None)) is not None:
                np.less(
                    self.input_batch.num_computed_tokens_cpu[:num_reqs],
                    self.input_batch.num_prompt_tokens[:num_reqs],
                    out=self._fl_is_prefilling_np[:num_reqs],
                )
                if num_reqs < nr_pad:
                    self._fl_is_prefilling_np[num_reqs:nr_pad] = False
                is_pref = pref_cpu[:nr_pad]
            else:
                is_pref = (
                    self.input_batch.num_computed_tokens_cpu_tensor[:nr_pad]
                    < self.input_batch.num_prompt_tokens_cpu_tensor[:nr_pad]
                )
                is_pref[num_reqs:] = False

            sl_cpu = self.optimistic_seq_lens_cpu[:nr_pad]
            cm = CommonAttentionMetadata(
                query_start_loc=self.query_start_loc.gpu[: nr_pad + 1],
                query_start_loc_cpu=self.query_start_loc.cpu[: nr_pad + 1],
                seq_lens=self.seq_lens[:nr_pad],
                _seq_lens_cpu=sl_cpu,
                _num_computed_tokens_cpu=self.input_batch.num_computed_tokens_cpu_tensor[:nr_pad],
                seq_lens_cpu_upper_bound=sl_cpu,
                num_reqs=nr_pad,
                num_actual_tokens=nt_pad,
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
                block_table_tensor=blk_t0,
                slot_mapping=slot_mappings[0],
                causal=True,
                is_prefilling=is_pref,
                positions=self.positions[:nt_pad],
                mm_req_doc_ranges=None,
            )
            ag = self.attn_groups[0][0]
            prefix_len = cascade_attn_prefix_lens[0][0] if cascade_attn_prefix_lens else 0
            meta0 = ag.get_metadata_builder(0).build(common_prefix_len=prefix_len, common_attn_metadata=cm)
            if _ACTIVE_CFG.enable_pod_horizon and getattr(self, "_fl_pod_valid", False):
                meta0.num_prefills = self._fl_num_prefills
                meta0.max_prefill_seq_len = self._fl_max_prefill_seq_len
                meta0.max_prefill_q_len = self._fl_max_prefill_q_len
                meta0.prefill_seq_idx = getattr(self, "_fl_prefill_seq_idx", -1)
                meta0.num_decode_seqs = self._fl_num_decode_seqs
                meta0.max_decode_seq_len = self._fl_max_decode_seq_len
                meta0.mean_decode_seq_len = getattr(self, "_fl_mean_decode_seq_len", 0)
            return dict.fromkeys(ag.layer_names, meta0), None

        call_kwargs = dict(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            max_query_len=max_query_len,
            num_tokens_padded=num_tokens_padded,
            num_reqs_padded=num_reqs_padded,
            ubatch_slices=ubatch_slices,
            logits_indices=logits_indices,
            use_spec_decode=use_spec_decode,
            for_cudagraph_capture=for_cudagraph_capture,
            num_scheduled_tokens=num_scheduled_tokens,
            cascade_attn_prefix_lens=cascade_attn_prefix_lens,
            slot_mappings=slot_mappings,
        )
        if extra_kwargs:
            call_kwargs.update(extra_kwargs)
        res = orig_attn_meta(self, **call_kwargs)
        if num_reqs < (eff_pad := num_reqs_padded or num_reqs):
            for bt in self.input_batch.block_table.block_tables:
                _bt_state(bt).record_gpu_zero_pad(bt, num_reqs, eff_pad)
        return res

    def _preprocess(self, scheduler_output, num_input_tokens: int, intermediate_tensors=None):
        """Return pre-padded `(input_ids, positions)` slices directly when `_prepare_inputs` handled padding."""
        was_padded = getattr(self, "_fl_step_prep_padded", False)
        padded_toks = getattr(self, "_fl_step_prep_padded_tokens", 0)
        self._fl_step_prep_padded, self._fl_step_prep_padded_tokens = False, 0
        if (
            was_padded
            and not self.supports_mm_inputs
            and not self.enable_prompt_embeds
            and not self.model_config.is_encoder_decoder
            and not self.uses_mrope
            and self.uses_xdrope_dim == 0
            and intermediate_tensors is None
        ):
            unpadded = scheduler_output.total_num_scheduled_tokens
            if unpadded < num_input_tokens and padded_toks < num_input_tokens:
                self.positions[max(unpadded, padded_toks) : num_input_tokens].zero_()
            return self.input_ids.gpu[:num_input_tokens], None, self.positions[:num_input_tokens], None, {}, None
        return orig_pre(self, scheduler_output, num_input_tokens, intermediate_tensors)

    return _prepare_inputs, _get_slot_mappings, _patched_build_attn_meta, _preprocess


def _make_async_output_init(orig_init):
    """Wrap `AsyncGPUModelRunnerOutput.__init__` with a per-device 4-slot pinned D2H ring.

    Overlaps sampled-token D2H copies on `async_output_copy_stream` with the next
    step's CPU scheduling while avoiding per-step pinned memory and CUDA event
    allocations for standard 1-token-per-request batches (`<= 1024` requests).
    """

    def _async_init(
        self,
        model_runner_output,
        sampled_token_ids: torch.Tensor,
        logprobs_tensors,
        invalid_req_indices: list[int],
        async_output_copy_stream,
        vocab_size: int,
        routed_experts=None,
    ):
        """Use a pre-allocated 4-slot pinned D2H ring for standard 1-token batches, else delegate to `orig_init`."""
        if (
            logprobs_tensors is None
            and routed_experts is None
            and sampled_token_ids.ndim == 2
            and sampled_token_ids.shape[1] == 1
            and sampled_token_ids.shape[0] <= 1024
        ):
            dev_idx = sampled_token_ids.device.index or 0
            state = _ASYNC_D2H_STATE.get(dev_idx)
            if state is None:
                if len(_ASYNC_D2H_STATE) >= 16:
                    _ASYNC_D2H_STATE.pop(next(iter(_ASYNC_D2H_STATE)))
                state = [0, {}, [torch.Event() for _ in range(4)]]
                _ASYNC_D2H_STATE[dev_idx] = state
            slot = state[0] & 3
            state[0] = slot + 1
            pinned_bufs, events = state[1], state[2]
            buf_key = (sampled_token_ids.dtype, slot)
            pbuf = pinned_bufs.get(buf_key)
            if pbuf is None:
                pbuf = torch.empty(
                    (1024, 1),
                    dtype=sampled_token_ids.dtype,
                    device="cpu",
                    pin_memory=PIN_MEMORY,
                )
                pinned_bufs[buf_key] = pbuf
            self._model_runner_output = model_runner_output
            self._invalid_req_indices = invalid_req_indices
            self.async_copy_ready_event = events[slot]
            self._sampled_token_ids = sampled_token_ids
            self.vocab_size = vocab_size
            self._logprobs_tensors = None
            self._routed_experts = None
            self._logprobs_tensors_cpu = None
            self._routed_experts_cpu = None
            cpu_slice = pbuf[: sampled_token_ids.shape[0]]
            def_s = torch.cuda.current_stream()
            with torch.cuda.stream(async_output_copy_stream):
                async_output_copy_stream.wait_stream(def_s)
                cpu_slice.copy_(sampled_token_ids, non_blocking=True)
                self.async_copy_ready_event.record()
            self.sampled_token_ids_cpu = cpu_slice
            return
        orig_init(
            self,
            model_runner_output,
            sampled_token_ids,
            logprobs_tensors,
            invalid_req_indices,
            async_output_copy_stream,
            vocab_size,
            routed_experts,
        )

    return _async_init


def install_step_prep_on_runner_module(mod: Any) -> None:
    """Idempotently patch `GPUModelRunner` / `ModelRunnerFL` and `AsyncGPUModelRunnerOutput` on `mod`."""
    if mod is None:
        return
    _patch_block_table_cls()
    for cls_name in ("GPUModelRunner", "ModelRunnerFL"):
        if (r_cls := getattr(mod, cls_name, None)) is not None and not getattr(r_cls, "_fl_step_prep_patched", False):
            r_cls._prepare_inputs, r_cls._get_slot_mappings, r_cls._build_attention_metadata, r_cls._preprocess = _make_runner_patches(
                r_cls._prepare_inputs, r_cls._get_slot_mappings, r_cls._build_attention_metadata, r_cls._preprocess
            )
            r_cls._fl_step_prep_patched = True
    if (async_cls := getattr(mod, "AsyncGPUModelRunnerOutput", None)) is not None and not getattr(async_cls, "_fl_async_out_patched", False):
        async_cls.__init__ = _make_async_output_init(async_cls.__init__)
        async_cls._fl_async_out_patched = True


def install_step_prep(config: StepPrepConfig | None = None, *, import_runners: bool = True) -> None:
    """Idempotently bind fused step-prep onto both `GPUModelRunner` and `ModelRunnerFL`."""
    global _ACTIVE_CFG
    if config is not None:
        _ACTIVE_CFG = config
    _patch_block_table_cls()
    for mod_name in ("vllm.v1.worker.gpu_model_runner", "vllm_fl.worker.model_runner"):
        mod = sys.modules.get(mod_name)
        if mod is None and import_runners:
            try:
                __import__(mod_name)
                mod = sys.modules.get(mod_name)
            except Exception:
                continue
        if mod is not None:
            install_step_prep_on_runner_module(mod)


__all__ = [
    "BlockTableSyncState",
    "StepPrepConfig",
    "install_step_prep",
    "install_step_prep_on_runner_module",
]
