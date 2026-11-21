"""Isolated request-indexed stage banks and ordered active-only publication.

Caller guarantees unique active RequestSlots and unique writable pages,
nonnegative bounded query intervals, valid slot IDs and Bank values in {0,1}.
Both kernels use the same current stream: publication follows all page CTAs.
Concurrent calls on different streams sharing these banks are unsupported.
"""
import torch
import triton
import triton.language as tl
from vllm_fl.dispatch.backends.flaggems.impl.quantization import _round_half_to_even


@triton.jit
def _request_bank_fullcover_BF16_write(NewK, NewV, StageBanks, Bank, RequestSlots,
                            K8, KS, V, CuQuery, SeqLens, Table, SlotMapping, COLS: tl.constexpr, V_TOKEN_STRIDE: tl.constexpr):
    page_offset = tl.program_id(0)
    batch = tl.program_id(1)
    head = tl.program_id(2)
    q_begin = tl.load(CuQuery + batch)
    q_end = tl.load(CuQuery + batch + 1)
    query_count = q_end - q_begin
    valid_slot = tl.load(SlotMapping + q_begin, mask=query_count > 0, other=-1) >= 0
    if query_count > 0 and valid_slot:
        seq_end = tl.load(SeqLens + batch)
        seq_start = seq_end - query_count
        first_page = seq_start // 16
        last_page = (seq_end - 1) // 16
        logical_page = first_page + page_offset
        if logical_page <= last_page:
            slot = tl.load(RequestSlots + batch)
            source_bank = tl.load(Bank + slot)
            dest_bank = 1 - source_bank
            physical = tl.load(Table + batch * COLS + logical_page)
            t = tl.arange(0, 16)
            d = tl.arange(0, 128)
            position = logical_page * 16 + t
            previous = (logical_page == first_page) & (position < seq_start)
            incoming = (position >= seq_start) & (position < seq_end)
            seed_ptr = StageBanks + ((((slot * 2 + source_bank) * 16 + t[:, None]) * 2 + head) * 128 + d[None, :])
            seed = tl.load(seed_ptr, mask=previous[:, None], other=0).to(tl.float32)
            token = q_begin + position - seq_start
            input_offset = (token[:, None] * 2 + head) * 128 + d[None, :]
            new_k = tl.load(NewK + input_offset, mask=incoming[:, None], other=0).to(tl.float32)
            x = tl.where(previous[:, None], seed, new_k)
            peak = tl.max(tl.max(tl.abs(x), 1), 0)
            scale = tl.maximum(peak / 127., tl.full((), 1.1754943508222875e-38, tl.float32)).to(tl.bfloat16)
            scale = tl.where(peak == 0, 1., scale).to(tl.bfloat16)
            quantized = _round_half_to_even(tl.div_rn(x, scale.to(tl.float32)))
            quantized = tl.minimum(tl.maximum(quantized, -127.), 127.).to(tl.int8)
            output_offset = ((physical * 16 + t[:, None]) * 2 + head) * 128 + d[None, :]
            tl.store(K8 + output_offset, quantized)
            tl.store(KS + physical * 2 + head, scale)
            # Touch every BF16 field in the existing contiguous physical
            # head page. Preserve nonincoming bytes by direct BF16 load/select.
            vi = tl.arange(0, 2048)
            word = vi // 4
            vc = 2 * (word // 8) + (word & 1)
            tg = ((word // 2) & 3) ^ ((vc >> 3) & 3)
            vt = 4 * tg + (vi & 3)
            incoming_v = vt == (seq_start % 16)
            vp = V + (physical * 2 + head) * 2048 + vi
            old_v = tl.load(vp)
            new_v = tl.load(NewV + q_begin * V_TOKEN_STRIDE + head * 128 + vc, mask=incoming_v, other=0)
            value_v = tl.where(incoming_v, new_v, old_v)
            tl.store(vp, value_v)
            if logical_page == last_page:
                dest_ptr = StageBanks + ((((slot * 2 + dest_bank) * 16 + t[:, None]) * 2 + head) * 128 + d[None, :])
                tl.store(dest_ptr, x.to(tl.bfloat16))


@triton.jit
def _publish_active_banks(Bank, RequestSlots, CuQuery, SlotMapping, BATCH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    begin = tl.load(CuQuery + row, mask=row < BATCH, other=0)
    end = tl.load(CuQuery + row + 1, mask=row < BATCH, other=0)
    active = (row < BATCH) & (end > begin)
    valid_slot = tl.load(SlotMapping + begin, mask=active, other=-1) >= 0
    active = active & valid_slot
    slot = tl.load(RequestSlots + row, mask=active, other=0)
    bank = tl.load(Bank + slot, mask=active, other=0)
    tl.store(Bank + slot, 1 - bank, mask=active)


def request_bank_write(new_k, new_v, stages, banks, request_slots, k8, scales, v,
                       cu_query, seq_lens, table, slot_mapping, *, max_query_tokens):
    if type(max_query_tokens) is not int or max_query_tokens < 1:
        raise ValueError('max_query_tokens must be a positive host integer')
    tensors = (new_k, new_v, stages, banks, request_slots, k8, scales, v, cu_query, seq_lens, table, slot_mapping)
    if not new_k.is_cuda or any(t.device != new_k.device for t in tensors):
        raise ValueError('all tensors must reside on the same GPU')
    if any(not t.is_contiguous() for t in tensors if t is not new_v):
        raise ValueError('all tensors except token-strided NewV must be contiguous')
    if new_v.stride(1) != 128 or new_v.stride(2) != 1 or new_v.stride(0) < 256:
        raise ValueError('NewV requires dense KV-head/channel axes and a nonoverlapping token stride')
    if seq_lens.ndim != 1 or seq_lens.numel() < 1:
        raise ValueError('seq_lens must be a nonempty request vector')
    if slot_mapping.dtype != torch.int64 or slot_mapping.ndim != 1 or slot_mapping.numel() < 1 or slot_mapping.numel() > new_k.shape[0]:
        raise ValueError('native slot_mapping extent must fit within padded input capacity')
    if max_query_tokens != 1:
        raise ValueError('Experimental 1D V writer requires declared uniform qlen<=1 bound')
    batch = seq_lens.numel()
    if table.ndim != 2 or table.shape[0] != batch or table.shape[1] < 1:
        raise ValueError('table must have B rows and positive width')
    if new_k.ndim != 3 or new_k.shape[1:] != (2, 128) or new_v.shape != new_k.shape:
        raise ValueError('incoming KV shape must be [token_capacity,2,128]')
    if stages.ndim != 5 or stages.shape[1:] != (2, 16, 2, 128) or stages.shape[0] < 1:
        raise ValueError('stage shape must be [request_slots,2,16,2,128]')
    if banks.shape != (stages.shape[0],) or request_slots.shape != (batch,) or cu_query.shape != (batch + 1,):
        raise ValueError('invalid bank, request slot or cumulative query dimensions')
    if k8.ndim != 4 or k8.shape[0] < 1 or k8.shape[1:] != (16, 2, 128) or v.shape != k8.shape:
        raise ValueError('cache shape must be [P,16,2,128]')
    if scales.shape != (k8.shape[0], 2):
        raise ValueError('scale shape must be [P,2]')
    if any(t.dtype != torch.bfloat16 for t in (new_k, new_v, stages, scales, v)) or k8.dtype != torch.int8:
        raise ValueError('expected BF16 data/stages/scales/V and INT8 K')
    if any(t.dtype != torch.int32 for t in (banks, request_slots, cu_query, seq_lens, table)):
        raise ValueError('metadata must be INT32')
    _request_bank_fullcover_BF16_write[(triton.cdiv(max_query_tokens + 15, 16), batch, 2)](
        new_k, new_v, stages, banks, request_slots, k8, scales, v, cu_query, seq_lens, table, slot_mapping,
        table.shape[1], new_v.stride(0), num_warps=4, num_stages=1)
    _publish_active_banks[(triton.cdiv(batch, 128),)](
        banks, request_slots, cu_query, slot_mapping, batch, 128, num_warps=1, num_stages=1)

# ThismodulewritesONLYphysicalVlayout846;notcompatiblewithtokenmajorreaders.
V_CACHE_LAYOUT = 846
