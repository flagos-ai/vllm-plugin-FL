"""Disjoint request-level M128 prefill and S16 decode, shared representation/math."""
import torch
import triton
import triton.language as tl
from .v846_readers import PreparedReader


@triton.jit(do_not_specialize=[10,11,12])
def _request_phase_metadata(Cu, Ends, PO, PB, PC, PV, DO, DB, DC, DV,
                            PCAP, DCAP,
                            BATCH, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    po = tl.full((BLOCK,), 0, tl.int32)
    pb = tl.full((BLOCK,), 0, tl.int32)
    pc = tl.full((BLOCK,), 0, tl.int32)
    pv = tl.full((BLOCK,), 0, tl.int32)
    do = tl.full((BLOCK,), 0, tl.int32)
    db = tl.full((BLOCK,), 0, tl.int32)
    dc = tl.full((BLOCK,), 0, tl.int32)
    dv = tl.full((BLOCK,), 0, tl.int32)
    pp = 0
    dp = 0
    for request in range(BATCH):
        begin = tl.load(Cu + request)
        finish = tl.load(Cu + request + 1)
        n = finish - begin
        end = tl.load(Ends + request)
        tiles = tl.where(n > 1, tl.cdiv(n, 128), 0)
        activep = (i >= PCAP - pp - tiles) & (i < PCAP - pp) & (i < PCAP)
        base = begin + (i - (PCAP - pp - tiles)) * 128
        po = tl.where(activep, request, po)
        pb = tl.where(activep, base, pb)
        pc = tl.where(activep, tl.minimum(128, finish - base), pc)
        pv = tl.where(activep, end - n + base - begin + 1, pv)
        actived = (n == 1) & (i == dp) & (i < DCAP)
        do = tl.where(actived, request, do)
        db = tl.where(actived, begin, db)
        dc = tl.where(actived, 1, dc)
        dv = tl.where(actived, end, dv)
        pp += tiles
        dp += tl.where(n == 1, 1, 0)
    tl.store(PO + i, po, mask=i < PCAP)
    tl.store(PB + i, pb, mask=i < PCAP)
    tl.store(PC + i, pc, mask=i < PCAP)
    tl.store(PV + i, pv, mask=i < PCAP)
    tl.store(DO + i, do, mask=i < DCAP)
    tl.store(DB + i, db, mask=i < DCAP)
    tl.store(DC + i, dc, mask=i < DCAP)
    tl.store(DV + i, dv, mask=i < DCAP)


class RequestPhaseReader:
    """Caller allocates decode metadata during profiled auxiliary allocation."""
    V_CACHE_LAYOUT = 846
    tile = 128

    def __init__(self):
        self.prefill = PreparedReader('prefill')
        self.decode = PreparedReader('decode', single_query=True)
        self.decode_metadata = None

    def _metadata(self, batch):
        if self.decode_metadata is None or len(self.decode_metadata) != 4:
            raise ValueError('request-phase decode metadata has not been profiled and bound')
        if any(x.numel() < batch or x.dtype != torch.int32 for x in self.decode_metadata):
            raise ValueError('request-phase decode metadata capacity/dtype mismatch')
        return [x[:batch] for x in self.decode_metadata]

    def prepare(self, cu, ends, owners, bases, counts, visible):
        decode = self._metadata(ends.numel())
        cap = max(owners.numel(), ends.numel())
        _request_phase_metadata[(triton.cdiv(cap, 128),)](
            cu, ends, owners, bases, counts, visible, *decode,
            owners.numel(), ends.numel(), ends.numel(), 128,
            num_warps=1, num_stages=1)

    def into(self, q, k, v, qs, ks, cu, owners, bases, counts, visible, table,
             part, maxima, sums, out, *, max_len, splits):
        if splits != 1:
            raise ValueError('request-phase entry requires its prefill split marker1')
        self.prefill.into(q, k, v, qs, ks, cu, owners, bases, counts, visible, table,
                          part, maxima, sums, out, max_len=max_len, splits=1)
        self.decode.into(q, k, v, qs, ks, cu, *self._metadata(cu.numel() - 1), table,
                         part, maxima, sums, out, max_len=max_len, splits=16)
